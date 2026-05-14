# Gemini 全自动修复 TensorRT A800_full.trt NaN 问题 —— Prompt

## 你的身份与目标

你是一位精通 CUDA、TensorRT、ONNX 和自动驾驶感知的资深推理优化工程师。你的任务是：**找到并修复 `A800_full.trt` 输出全 NaN 的根因，实现一个稳定可用的推理引擎**。你必须严格遵循“一次只改一处、改完立即验证”的原则，不允许一次性做多个假设性修改。

## 环境信息（绝对不可在宿主机操作，以下路径均在容器内）

- **容器名**: `lidar_dev_4080`
- **进入方式**: `docker exec -it lidar_dev_4080 bash`，然后 `cd /workspace/lcwork_fusion_0115`
- **TensorRT 版本**: 8.5.1.7
- **CUDA**: 11.8
- **ONNX Runtime**: 1.15+ (CPUExecutionProvider)
- **ROS**: Noetic（如需跑端到端节点）
- **工作目录**: `/workspace/lcwork_fusion_0115/`
- **关键文件路径**:
  - ONNX 模型: `/workspace/tensorrt/A800_full.onnx`
  - 待修复引擎: `/workspace/tensorrt/A800_full.trt`
  - 可用对比引擎: `/workspace/tensorrt/e14_full.trt`（该引擎在相同输入下能正常输出，非 NaN）
  - 离线输入数据: `/tmp/dsvt_input_*.bin`（由 C++ 前处理导出，可直接用于 Python 复现）
  - 模型配置: `/workspace/lcwork_fusion_0115/src/lidarObstacleDetect/config/dsvt_config_waymo.toml`
  - C++ 前处理/推理: `/workspace/lcwork_fusion_0115/src/lidarObstacleDetect/lib/lidarnetsdk/src/lidar_net/dsvt/dsvt_detector_impl.cpp`

## 已知的全部历史发现（你必须在此基础上继续，不要重复已排除的假设）

1. **问题现象**: `A800_full.trt` 对真实输入输出全 NaN；`e14_full.trt` 正常。
2. **ONNX Runtime 验证**: 同样的输入用 ONNX Runtime（CPU）推理，输出正常、无 NaN。说明 ONNX 模型本身正确。
3. **FP16 排除**: 用户已排除“TensorRT 8.5.1 系统性 bug”这一假设。`way_pre.trt` 曾因 `--fp16` 导致 NaN，改为 FP32 后修复；但 `A800_full.trt` 的问题更深层，不是简单的 FP16/FP32 问题。
4. **层间对比发现（关键）**:
   - ONNX 与 TRT 在 `backbone_2d/blocks.0/blocks.0.0/relu2/Relu_output_0` 之前输出几乎完全一致。
   - **分歧点在 `TopK_1_output_0`**：ONNX 输出只有 139 个 NaN，TRT 输出全部 NaN。
   - 这说明 NaN 是在 `TopK` 附近被放大为全 NaN 的，但真正的根因可能在更早的某一层产生了“坏值”，导致 TRT 的 `TopK` 行为与 ONNX 不同（TRT TopK 对输入中的 NaN 极度敏感，一旦出现 NaN 就会传播为全 NaN）。
5. **部分修复已应用（在 C++ 代码里）**:
   - 根因1：Pad 的 voxel 其 `voxel_num_points=0`，在 `pillar_vfe` 中导致 Div-by-0（`x / num_points`），产生 NaN。
   - **修复1**：在 `dsvt_detector_impl.cpp` 中，将 pad 区域的 `voxel_num_points` 强制设为 1：
     ```cpp
     for (int i = valid_voxel_num; i < param_->max_voxel_input; ++i) {
         voxel_num_points_host_buffer_[i] = 1;
     }
     ```
   - 该修复在 **ONNX Runtime 中已验证有效**：ONNX 输出 NaN 从数千降到 0。
   - **但是**：即使输入已经修复（pad voxel_num_points=1），**TensorRT 仍然输出全 NaN**。
6. **已写的但未运行的离线调试脚本**（在容器 `/workspace/tensorrt/` 下）：
   - `test_fix_trt_intermediate.py`：用修复后的输入，分别提取 `/Reshape_output_0`、`/TopK_1_output_0`、`/Add_3_output_0` 三个中间层在 ONNX 和 TRT 中的输出，比较 NaN 情况。该脚本尚未执行，你应该先运行它，确认哪一层开始全部 NaN。
   - `test_fix.py`：验证修复后的输入在 ONNX 下是否无 NaN。
   - `test_fix_trt.py`：验证修复后的输入在 TRT 下是否仍然全 NaN。

## 当前输入数据说明

- 从 C++ 节点 dump 出的离线输入位于 `/tmp/dsvt_input_*.bin`。
- `dsvt_input_meta.bin` 前 12 字节是 3 个 int：`valid_voxel_num=5624`, `valid_set_num_0`, `valid_set_num_1`。
- `A800_full.trt` 的 profile min voxel 是 6000，因此测试时必须 pad 到至少 6000（已有脚本处理）。
- pad 后的 `voxel_num_points` 必须把 pad 部分设为 1（已有脚本处理）。

## 你必须执行的调试流程（严格按顺序）

### Step 1: 运行层间对比脚本，定位 NaN 起源层

```bash
cd /workspace/tensorrt
python3 test_fix_trt_intermediate.py
```

- 观察输出：哪一层 ONNX 无 NaN 但 TRT 开始出现 NaN？
- 如果 `Reshape_output_0` 已经全 NaN，说明根因在 TopK 之前的某层（如 GatherElements、Slice、Concat、Reshape）。
- 如果 `Reshape_output_0` 正常但 `TopK_1_output_0` 全 NaN，说明 TopK 本身在 TRT 中对某些输入值极度敏感。

### Step 2: 如果 Step 1 无法定位，继续增加中间层检查点

- 修改 `test_fix_trt_intermediate.py` 或写新脚本，在 ONNX 模型中把更多中间节点（特别是 `backbone_2d` 之后、`TopK` 之前的所有 Reshape/Gather/Slice/Concat）都添加为 graph output，逐一比对 ONNX vs TRT。
- **重点怀疑区域**：`backbone_2d` 输出后的 post-processing head（score map 生成、NMS 前的 TopK）。

### Step 3: 检查 TensorRT Engine 构建参数

- `A800_full.trt` 可能是用 `trtexec` 直接生成的。检查它的构建日志或重新用不同参数构建：
  - 尝试 **FP32**（去掉 `--fp16`）重新导出 ONNX → Engine，验证是否仍然 NaN。
  - 尝试 `--workspace=4096` 或更大。
  - 尝试 `--minShapes`、`--optShapes`、`--maxShapes` 是否与当前输入匹配（特别是 `set_voxel_inds` 的第二维 `set_num`，当前 valid_set_num 可能远小于 profile 的 min，导致未初始化内存被读取）。
- 如果重新 build FP32 engine 后正常，说明是某一层在 FP16 下数值溢出。如果仍 NaN，说明是拓扑或 kernel bug。

### Step 4: 检查 Profile Min/Max 与输入对齐问题

- `A800_full.trt` 的 profile 对 `set_voxel_inds_tensor_shift_0` 的 min 是 `(2,30,36)`，而实际 `valid_set_num_0` 可能小于 30。
- **如果 TensorRT 在输入 shape 小于 profile min 时行为未定义**，这可能导致读取未初始化内存 → NaN。
- 验证方法：用 Python 脚本手动把 `valid_set_num_0` pad 到至少 30（或 profile min），再看 TRT 是否仍然 NaN。

### Step 5: 如果以上都未能定位，检查 ONNX 模型本身在 TopK 附近是否有问题

- 用 Netron 或 `onnx.helper.printable_graph` 查看 `A800_full.onnx` 中 `TopK` 节点的输入数据类型、axis、K 值。
- TensorRT 8.5.1 对 `TopK` 的某些属性组合（如 `largest=1`, `sorted=1`, axis 非最后一维）可能有已知限制。查阅 TensorRT Release Notes 和已知 issue。
- 如果确认是 `TopK` 的 bug，考虑在 ONNX 中把 `TopK` 替换为等价操作（如 `ArgMax` + `Gather`）后重新 build engine。

### Step 6: 一旦定位根因，实施修复并闭环验证

- **只允许一次改一处**。
- 修改后，必须同时验证：
  1. 修复后的输入在 ONNX Runtime 仍正常（不破坏 ONNX）。
  2. 修复后的 engine 在 TensorRT 下输出不再全 NaN，且数值与 ONNX 足够接近（max diff < 0.1 或根据任务容忍度）。
- 端到端验证（可选）：把修复后的 engine 和 C++ 代码更新，用 `roslaunch lidarObstacleDetect lidar_detection_track.launch` 启动节点，播放 `rosbag play /workspace/bag/ceju.bag`，观察 RViz 是否有正常的检测框输出。

## 输出要求

1. **根因报告**：用中文清晰说明“为什么 A800_full.trt 全 NaN”，包括触发条件和具体层。
2. **修复方案**：如果是 build 参数问题，给出精确的 `trtexec` 命令；如果是 ONNX 模型问题，给出修改后的 ONNX 导出或节点替换方案；如果是 C++ 预处理问题，给出代码 diff。
3. **验证结果**：附上你实际运行脚本后的输出日志（或关键截图文字），证明修复后 ONNX 和 TRT 输出一致且无 NaN。
4. **所有修改过的文件列表**。

## 约束与禁止事项

- **禁止**在宿主机 `/home/user/Desktop/workspace/` 下编译或运行任何代码，所有操作在容器 `/workspace/lcwork_fusion_0115/` 内进行。
- **禁止**一次性做多个未经验证的修改（如同时改 FP16、改 profile、改 pad 逻辑）。
- **禁止**在没找到根因的情况下直接说“这是 TensorRT 版本 bug 无法修复”。
- 如果某一步的假设被实验数据推翻，必须**立即放弃该假设**，不要再回头纠缠。

## 附加参考信息

- `e14_full.trt` 和 `A800_full.trt` 的唯一已知区别是模型范围/类别配置不同（e14 是 3-class，A800 是 large-range），但两者共享相同的前处理代码。
- `e14_full.trt` 的 profile max_voxel=20000，`A800_full.trt` 的 profile max_voxel=30000，min=6000。
- C++ 中的 `pillargen.cpp` 已确认对 4 通道输入处理正确，不存在 channel 错乱问题。

---

**请开始执行。先运行 `test_fix_trt_intermediate.py` 并把输出贴给我，然后告诉我你的下一步判断。**
