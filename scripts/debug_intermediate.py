#!/usr/bin/env python3
"""
Extract intermediate output from ONNX model and compare ONNXRT vs TRT.
"""
import struct
import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH = "/workspace/tensorrt/A800_full.onnx"
ENGINE_PATH = "/workspace/tensorrt/A800_full.trt"
META_PATH = "/tmp/dsvt_input_meta.bin"

# Read meta
with open(META_PATH, "rb") as f:
    meta = struct.unpack("iii", f.read(12))
valid_voxel_num, valid_set_num_0, valid_set_num_1 = meta
print(f"Meta: valid_voxel_num={valid_voxel_num}, valid_set_num_0={valid_set_num_0}, valid_set_num_1={valid_set_num_1}")

MAX_VOXEL = 20000
MAX_SET_NUM = 1200

def load_bin(path, dtype, shape):
    return np.fromfile(path, dtype=dtype).reshape(shape)

pillar_features = load_bin("/tmp/dsvt_input_pillar_features.bin", np.float32, (MAX_VOXEL, 20, 4))
pillar_coors = load_bin("/tmp/dsvt_input_pillar_coors.bin", np.int32, (MAX_VOXEL, 4))
voxel_num_points = load_bin("/tmp/dsvt_input_voxel_num_points.bin", np.int32, (MAX_VOXEL,))
set_voxel_inds_0 = load_bin("/tmp/dsvt_input_set_voxel_inds_shift_0.bin", np.int32, (2, MAX_SET_NUM, 36))
set_voxel_inds_1 = load_bin("/tmp/dsvt_input_set_voxel_inds_shift_1.bin", np.int32, (2, MAX_SET_NUM, 36))
set_voxel_masks_0 = load_bin("/tmp/dsvt_input_set_voxel_masks_shift_0.bin", np.bool_, (2, MAX_SET_NUM, 36))
set_voxel_masks_1 = load_bin("/tmp/dsvt_input_set_voxel_masks_shift_1.bin", np.bool_, (2, MAX_SET_NUM, 36))
coors_in_win = load_bin("/tmp/dsvt_input_coors_in_win.bin", np.int32, (2, MAX_VOXEL, 3))

# Pad to 6000 for A800_full.trt
MIN_VOXEL = 6000
voxel_num = max(valid_voxel_num, MIN_VOXEL)

def pad_arr(arr, target_size, axis=0):
    if arr.shape[axis] >= target_size:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_size - arr.shape[axis])
    return np.pad(arr, pad_width, mode='constant', constant_values=0)

inputs = {
    "pillar_features": pad_arr(pillar_features[:valid_voxel_num], voxel_num, axis=0),
    "pillar_coors": pad_arr(pillar_coors[:valid_voxel_num], voxel_num, axis=0),
    "voxel_num_points": pad_arr(voxel_num_points[:valid_voxel_num], voxel_num, axis=0),
    "set_voxel_inds_tensor_shift_0": set_voxel_inds_0[:, :valid_set_num_0, :],
    "set_voxel_inds_tensor_shift_1": set_voxel_inds_1[:, :valid_set_num_1, :],
    "set_voxel_masks_tensor_shift_0": set_voxel_masks_0[:, :valid_set_num_0, :],
    "set_voxel_masks_tensor_shift_1": set_voxel_masks_1[:, :valid_set_num_1, :],
    "coors_in_win": pad_arr(coors_in_win[:, :valid_voxel_num, :], voxel_num, axis=1),
}

inputs_onnx = {k: v for k, v in inputs.items()}
inputs_onnx["set_voxel_inds_tensor_shift_0"] = inputs_onnx["set_voxel_inds_tensor_shift_0"].astype(np.int64)
inputs_onnx["set_voxel_inds_tensor_shift_1"] = inputs_onnx["set_voxel_inds_tensor_shift_1"].astype(np.int64)

INTERMEDIATE_NAMES = [
    "/TopK_1_output_0",
    "/Reshape_1_output_0",
    "/Reshape_output_0",
]

def run_onnx_with_intermediate(onnx_path, intermediate_name):
    m = onnx.load(onnx_path)
    # Add intermediate as output
    found = False
    for node in m.graph.node:
        if intermediate_name in node.output:
            found = True
            break
    if not found:
        print(f"WARNING: {intermediate_name} not found as node output in model")
        # It may be an existing graph output
    else:
        # Check if already output
        already = any(o.name == intermediate_name for o in m.graph.output)
        if not already:
            value_info = onnx.helper.make_tensor_value_info(intermediate_name, onnx.TensorProto.FLOAT, None)
            m.graph.output.append(value_info)
            print(f"Added {intermediate_name} as graph output")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m.SerializeToString(), sess_options, providers=["CPUExecutionProvider"])
    outputs = sess.run([intermediate_name], inputs_onnx)
    return outputs[0]

def run_trt_with_intermediate(engine_path, intermediate_name):
    # For prebuilt engine, we cannot add intermediate outputs.
    # We need to build a new engine from modified ONNX.
    m = onnx.load(ONNX_PATH)
    found = False
    for node in m.graph.node:
        if intermediate_name in node.output:
            found = True
            break
    if not found:
        print(f"WARNING: {intermediate_name} not found as node output in model")
    else:
        already = any(o.name == intermediate_name for o in m.graph.output)
        if not already:
            value_info = onnx.helper.make_tensor_value_info(intermediate_name, onnx.TensorProto.FLOAT, None)
            m.graph.output.append(value_info)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(m.SerializeToString()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30

    profile = builder.create_optimization_profile()
    profile.set_shape("pillar_features", (6000,20,4), (30000,20,4), (30000,20,4))
    profile.set_shape("pillar_coors", (6000,4), (30000,4), (30000,4))
    profile.set_shape("voxel_num_points", (6000,), (30000,), (30000,))
    profile.set_shape("set_voxel_inds_tensor_shift_0", (2,30,36), (2,1200,36), (2,1200,36))
    profile.set_shape("set_voxel_inds_tensor_shift_1", (2,30,36), (2,1200,36), (2,1200,36))
    profile.set_shape("set_voxel_masks_tensor_shift_0", (2,30,36), (2,1200,36), (2,1200,36))
    profile.set_shape("set_voxel_masks_tensor_shift_1", (2,30,36), (2,1200,36), (2,1200,36))
    profile.set_shape("coors_in_win", (2,6000,3), (2,30000,3), (2,30000,3))
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed")

    context = engine.create_execution_context()
    bindings = []
    host_buffers = []
    device_buffers = []

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        if engine.binding_is_input(i):
            shape = inputs[name].shape
            context.set_binding_shape(i, shape)
            host_mem = inputs[name].astype(dtype)
        else:
            shape = tuple(context.get_binding_shape(i))
            if any(s <= 0 for s in shape):
                shape = (1, 500, 11)
            host_mem = np.empty(shape, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        host_buffers.append(host_mem)
        device_buffers.append(device_mem)

    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            cuda.memcpy_htod(device_buffers[i], host_buffers[i])

    context.execute_v2(bindings=bindings)

    out_idx = -1
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i) and engine.get_binding_name(i) == intermediate_name:
            out_idx = i
            break
    if out_idx == -1:
        raise RuntimeError(f"Intermediate {intermediate_name} not found in engine outputs")

    cuda.memcpy_dtoh(host_buffers[out_idx], device_buffers[out_idx])
    out = host_buffers[out_idx]

    for d in device_buffers:
        d.free()
    return out

for name in INTERMEDIATE_NAMES:
    print(f"\n========== Checking intermediate: {name} ==========")
    try:
        onnx_out = run_onnx_with_intermediate(ONNX_PATH, name)
        print(f"ONNX  shape={onnx_out.shape} min={np.nanmin(onnx_out):.6f} max={np.nanmax(onnx_out):.6f} mean={np.nanmean(onnx_out):.6f} nan={np.isnan(onnx_out).sum()}")
    except Exception as e:
        print(f"ONNX failed: {e}")
        onnx_out = None

    try:
        trt_out = run_trt_with_intermediate(ENGINE_PATH, name)
        print(f"TRT   shape={trt_out.shape} min={np.nanmin(trt_out):.6f} max={np.nanmax(trt_out):.6f} mean={np.nanmean(trt_out):.6f} nan={np.isnan(trt_out).sum()}")
    except Exception as e:
        print(f"TRT failed: {e}")
        trt_out = None

    if onnx_out is not None and trt_out is not None:
        diff = np.abs(onnx_out.astype(np.float64) - trt_out.astype(np.float64))
        print(f"Diff  max={np.nanmax(diff):.6f} mean={np.nanmean(diff):.6f}")
