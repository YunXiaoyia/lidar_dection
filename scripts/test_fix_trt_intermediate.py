#!/usr/bin/env python3
import struct
import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH = "/workspace/tensorrt/A800_full.onnx"
META_PATH = "/tmp/dsvt_input_meta.bin"

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
inputs["voxel_num_points"][valid_voxel_num:] = 1

inputs_onnx = {k: v for k, v in inputs.items()}
inputs_onnx["set_voxel_inds_tensor_shift_0"] = inputs_onnx["set_voxel_inds_tensor_shift_0"].astype(np.int64)
inputs_onnx["set_voxel_inds_tensor_shift_1"] = inputs_onnx["set_voxel_inds_tensor_shift_1"].astype(np.int64)

for intermediate_name in ["/Reshape_output_0", "/TopK_1_output_0", "/Add_3_output_0"]:
    print(f"\n========== {intermediate_name} ==========")
    m = onnx.load(ONNX_PATH)
    for node in m.graph.node:
        if intermediate_name in node.output:
            break
    already = any(o.name == intermediate_name for o in m.graph.output)
    if not already:
        value_info = onnx.helper.make_tensor_value_info(intermediate_name, onnx.TensorProto.FLOAT, None)
        m.graph.output.append(value_info)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m.SerializeToString(), sess_options, providers=["CPUExecutionProvider"])
    onnx_out = sess.run([intermediate_name], inputs_onnx)[0]
    print(f"ONNX: shape={onnx_out.shape} nan={np.isnan(onnx_out).sum()}")

    # TensorRT
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(m.SerializeToString()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        continue

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
        print("TRT: engine build failed")
        continue

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
        print("TRT: output not found")
        continue
    cuda.memcpy_dtoh(host_buffers[out_idx], device_buffers[out_idx])
    trt_out = host_buffers[out_idx]
    print(f"TRT:  shape={trt_out.shape} nan={np.isnan(trt_out).sum()}")
    for d in device_buffers:
        d.free()
