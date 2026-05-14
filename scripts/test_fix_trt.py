#!/usr/bin/env python3
"""
Test fixed input on A800_full.trt
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

# Use actual valid voxel num (no pad needed if >= 6000)
# But A800_full.trt profile min=6000, so we need at least 6000.
# Our saved valid_voxel_num=5624 < 6000, so we still need pad.
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

# FIX: set pad voxel_num_points to 1
inputs["voxel_num_points"][valid_voxel_num:] = 1

# ONNX Runtime
inputs_onnx = {k: v for k, v in inputs.items()}
inputs_onnx["set_voxel_inds_tensor_shift_0"] = inputs_onnx["set_voxel_inds_tensor_shift_0"].astype(np.int64)
inputs_onnx["set_voxel_inds_tensor_shift_1"] = inputs_onnx["set_voxel_inds_tensor_shift_1"].astype(np.int64)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=["CPUExecutionProvider"])
onnx_pred = sess.run(None, inputs_onnx)[0]

print(f"\nONNX: shape={onnx_pred.shape} nan={np.isnan(onnx_pred).sum()} score_max={np.nanmax(onnx_pred[0,:,8]):.6f}")

# TensorRT
with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

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

for i in range(engine.num_bindings):
    if not engine.binding_is_input(i):
        cuda.memcpy_dtoh(host_buffers[i], device_buffers[i])
        trt_pred = host_buffers[i]

for d in device_buffers:
    d.free()

print(f"TRT:  shape={trt_pred.shape} nan={np.isnan(trt_pred).sum()} score_max={np.nanmax(trt_pred[0,:,8]):.6f}")
