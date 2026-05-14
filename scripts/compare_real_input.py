#!/usr/bin/env python3
"""
Load real dumped inputs from /tmp/dsvt_input_*.bin and compare ONNX vs TensorRT.
"""
import struct
import numpy as np
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

# Config shapes (max shapes from toml, but we will slice to valid)
MAX_VOXEL = 20000  # because we dumped with e14 config
MAX_SET_NUM = 1200

def load_bin(path, dtype, shape):
    arr = np.fromfile(path, dtype=dtype).reshape(shape)
    return arr

pillar_features = load_bin("/tmp/dsvt_input_pillar_features.bin", np.float32, (MAX_VOXEL, 20, 4))
pillar_coors = load_bin("/tmp/dsvt_input_pillar_coors.bin", np.int32, (MAX_VOXEL, 4))
voxel_num_points = load_bin("/tmp/dsvt_input_voxel_num_points.bin", np.int32, (MAX_VOXEL,))
set_voxel_inds_0 = load_bin("/tmp/dsvt_input_set_voxel_inds_shift_0.bin", np.int32, (2, MAX_SET_NUM, 36))
set_voxel_inds_1 = load_bin("/tmp/dsvt_input_set_voxel_inds_shift_1.bin", np.int32, (2, MAX_SET_NUM, 36))
set_voxel_masks_0 = load_bin("/tmp/dsvt_input_set_voxel_masks_shift_0.bin", np.bool_, (2, MAX_SET_NUM, 36))
set_voxel_masks_1 = load_bin("/tmp/dsvt_input_set_voxel_masks_shift_1.bin", np.bool_, (2, MAX_SET_NUM, 36))
coors_in_win = load_bin("/tmp/dsvt_input_coors_in_win.bin", np.int32, (2, MAX_VOXEL, 3))

# Slice to actual valid shapes for A800_full (max 30000)
# We need to pad to at least 6000 for A800_full.trt if valid < 6000
# But our saved frame has valid_voxel_num=6360, so it's fine.
# However A800_full.trt was compiled with max_voxel=30000, so we can pad to 30000 or use 6360.
# Let's try 6360 first (since A800_full profile min=6000, max=30000)

# Pad voxel_num to at least 6000 to satisfy A800_full.trt profile min=6000
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

# ONNX Runtime (disable optimizations to avoid MatMulBnFusion bug)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=["CPUExecutionProvider"])

# ONNX model expects int64 for set_voxel_inds
inputs_onnx = {k: v for k, v in inputs.items()}
inputs_onnx["set_voxel_inds_tensor_shift_0"] = inputs_onnx["set_voxel_inds_tensor_shift_0"].astype(np.int64)
inputs_onnx["set_voxel_inds_tensor_shift_1"] = inputs_onnx["set_voxel_inds_tensor_shift_1"].astype(np.int64)

output_names = [o.name for o in sess.get_outputs()]
onnx_outputs = sess.run(output_names, inputs_onnx)
onnx_pred = onnx_outputs[0]

print(f"\n=== ONNX Runtime Output ===")
print(f"shape: {onnx_pred.shape}")
print(f"min: {np.nanmin(onnx_pred):.6f}, max: {np.nanmax(onnx_pred):.6f}, mean: {np.nanmean(onnx_pred):.6f}")
print(f"nan count: {np.isnan(onnx_pred).sum()}, inf count: {np.isinf(onnx_pred).sum()}")
print(f"score (col 8) max: {np.nanmax(onnx_pred[0,:,8]):.6f}")
print(f"iou (col 10) max: {np.nanmax(onnx_pred[0,:,10]):.6f}")

# TensorRT: load prebuilt engine
with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Set binding shapes
bindings = []
host_buffers = []
device_buffers = []

for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    dtype = trt.nptype(engine.get_binding_dtype(i))
    if engine.binding_is_input(i):
        if name in inputs:
            shape = inputs[name].shape
        else:
            raise RuntimeError(f"Unknown input: {name}")
        context.set_binding_shape(i, shape)
        host_mem = inputs[name].astype(dtype)
    else:
        shape = tuple(context.get_binding_shape(i))
        if any(s <= 0 for s in shape):
            # Output shape may not be resolved yet; try after all inputs set
            shape = (1, 500, 11)
        host_mem = np.empty(shape, dtype=dtype)

    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    host_buffers.append(host_mem)
    device_buffers.append(device_mem)

# H2D
for i in range(engine.num_bindings):
    if engine.binding_is_input(i):
        cuda.memcpy_htod(device_buffers[i], host_buffers[i])

# Execute
print("\n=== TensorRT Execute ===")
status = context.execute_v2(bindings=bindings)
print(f"execute_v2 returned: {status}")

# D2H
trt_outputs = {}
for i in range(engine.num_bindings):
    if not engine.binding_is_input(i):
        cuda.memcpy_dtoh(host_buffers[i], device_buffers[i])
        name = engine.get_binding_name(i)
        trt_outputs[name] = host_buffers[i]

# Free
for d in device_buffers:
    d.free()

trt_pred = trt_outputs["pred_box"]
print(f"\n=== TensorRT Output ===")
print(f"shape: {trt_pred.shape}")
print(f"min: {np.nanmin(trt_pred):.6f}, max: {np.nanmax(trt_pred):.6f}, mean: {np.nanmean(trt_pred):.6f}")
print(f"nan count: {np.isnan(trt_pred).sum()}, inf count: {np.isinf(trt_pred).sum()}")
print(f"score (col 8) max: {np.nanmax(trt_pred[0,:,8]):.6f}")
print(f"iou (col 10) max: {np.nanmax(trt_pred[0,:,10]):.6f}")

# Compare first few rows
print("\n=== Row-by-row comparison (first 5 rows) ===")
for i in range(5):
    a = onnx_pred[0, i]
    b = trt_pred[0, i]
    diff = np.abs(a - b)
    print(f"Row {i}: ONNX score={a[8]:.6f} iou={a[10]:.6f} | TRT score={b[8]:.6f} iou={b[10]:.6f} | max_diff={np.nanmax(diff):.6f}")
