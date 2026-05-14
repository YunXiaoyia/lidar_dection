#!/usr/bin/env python3
"""
Test if setting pad voxel_num_points to 1 removes NaN in ONNX.
"""
import struct
import numpy as np
import onnxruntime as ort

ONNX_PATH = "/workspace/tensorrt/A800_full.onnx"
META_PATH = "/tmp/dsvt_input_meta.bin"

with open(META_PATH, "rb") as f:
    meta = struct.unpack("iii", f.read(12))
valid_voxel_num, valid_set_num_0, valid_set_num_1 = meta

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

# Pad to 6000
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

inputs_onnx = {k: v for k, v in inputs.items()}
inputs_onnx["set_voxel_inds_tensor_shift_0"] = inputs_onnx["set_voxel_inds_tensor_shift_0"].astype(np.int64)
inputs_onnx["set_voxel_inds_tensor_shift_1"] = inputs_onnx["set_voxel_inds_tensor_shift_1"].astype(np.int64)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=["CPUExecutionProvider"])

output_names = [o.name for o in sess.get_outputs()]
onnx_outputs = sess.run(output_names, inputs_onnx)
onnx_pred = onnx_outputs[0]

print(f"ONNX Output (with fix):")
print(f"shape: {onnx_pred.shape}")
print(f"min: {np.nanmin(onnx_pred):.6f}, max: {np.nanmax(onnx_pred):.6f}, mean: {np.nanmean(onnx_pred):.6f}")
print(f"nan count: {np.isnan(onnx_pred).sum()}, inf count: {np.isinf(onnx_pred).sum()}")
print(f"score (col 8) max: {np.nanmax(onnx_pred[0,:,8]):.6f}")
print(f"iou (col 10) max: {np.nanmax(onnx_pred[0,:,10]):.6f}")
