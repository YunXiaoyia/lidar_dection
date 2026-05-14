#!/usr/bin/env python3
"""
Compare ONNX Runtime vs TensorRT output for A800_full model.
"""
import sys
import numpy as np

# ONNX Runtime
import onnxruntime as ort

# TensorRT
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ONNX_PATH = "/workspace/tensorrt/A800_full.onnx"

INPUT_SHAPES = {
    "pillar_features": (30000, 20, 4),
    "pillar_coors": (30000, 4),
    "voxel_num_points": (30000,),
    "set_voxel_inds_tensor_shift_0": (2, 1200, 36),
    "set_voxel_inds_tensor_shift_1": (2, 1200, 36),
    "set_voxel_masks_tensor_shift_0": (2, 1200, 36),
    "set_voxel_masks_tensor_shift_1": (2, 1200, 36),
    "coors_in_win": (2, 30000, 3),
}

INPUT_DTYPES = {
    "pillar_features": np.float32,
    "pillar_coors": np.int32,
    "voxel_num_points": np.int32,
    "set_voxel_inds_tensor_shift_0": np.int64,
    "set_voxel_inds_tensor_shift_1": np.int64,
    "set_voxel_masks_tensor_shift_0": np.bool_,
    "set_voxel_masks_tensor_shift_1": np.bool_,
    "coors_in_win": np.int32,
}


def generate_random_inputs(seed=42):
    rng = np.random.default_rng(seed)
    inputs = {}
    for name, shape in INPUT_SHAPES.items():
        dtype = INPUT_DTYPES[name]
        if dtype == np.bool_:
            inputs[name] = rng.integers(0, 2, size=shape, dtype=np.bool_)
        elif np.issubdtype(dtype, np.integer):
            inputs[name] = rng.integers(0, 100, size=shape, dtype=dtype)
        else:
            inputs[name] = rng.normal(0, 1, size=shape).astype(dtype)
    return inputs


def run_onnx(inputs):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=["CPUExecutionProvider"])

    ort_inputs = {k: v for k, v in inputs.items()}
    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, ort_inputs)
    return dict(zip(output_names, outputs))


def build_engine(onnx_path, explicit_batch=True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30  # 4GB

    profile = builder.create_optimization_profile()
    for name, shape in INPUT_SHAPES.items():
        min_shape = tuple(max(1, s // 10) if s > 1 else s for s in shape)
        opt_shape = shape
        max_shape = shape
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed")
    return engine


def run_trt(inputs):
    engine = build_engine(ONNX_PATH)
    context = engine.create_execution_context()

    # Set binding shapes
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
            shape = context.get_binding_shape(i)
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
    context.execute_v2(bindings=bindings)

    # D2H
    outputs = {}
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            cuda.memcpy_dtoh(host_buffers[i], device_buffers[i])
            name = engine.get_binding_name(i)
            outputs[name] = host_buffers[i]

    # Free
    for d in device_buffers:
        d.free()

    return outputs


def print_stats(arr, label):
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    print(f"[{label}] shape={arr.shape} min={np.nanmin(arr):.6f} max={np.nanmax(arr):.6f} mean={np.nanmean(arr):.6f} nan={nan_count} inf={inf_count}")


def main():
    print("Generating random inputs...")
    inputs = generate_random_inputs()

    print("\n=== ONNX Runtime (CPU, no optimization) ===")
    onnx_outputs = run_onnx(inputs)
    for name, arr in onnx_outputs.items():
        print_stats(arr, f"ONNX-{name}")

    print("\n=== TensorRT ===")
    trt_outputs = run_trt(inputs)
    for name, arr in trt_outputs.items():
        print_stats(arr, f"TRT-{name}")

    print("\n=== Diff ===")
    for name in onnx_outputs:
        if name in trt_outputs:
            a = onnx_outputs[name]
            b = trt_outputs[name]
            diff = np.abs(a - b)
            print(f"[{name}] max_diff={np.nanmax(diff):.6f} mean_diff={np.nanmean(diff):.6f}")


if __name__ == "__main__":
    main()
