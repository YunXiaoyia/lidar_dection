# LidarNet SDK

## 1. Introduction

本项目实现了lidar 深度学习的推理部分，包括数据预处理、模型推理、后处理等。


## 2. Compilation and Installation

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
make package
```

```bash
cd packages
sudo dpkg -i lib*.deb
```

## 4. 使用

### Logger
LidarNetSDK支持使用用户的Logger。用户可以自定义Logger的实现，只要实现了LidarNetLogger接口即可。如果用户没有自定义Logger，LidarNetSDK会使用其默认的Logger。

1. spdlog
```cpp
class UserLogger : public lidar_net::LidarNetLogger {
    void log(lidar_net::LidarNetLogger::Severity severity, const std::string_view msg) noexcept override {
        try {
            switch (severity) {
                case lidar_net::LidarNetLogger::Severity::kINTERNAL_ERROR:
                    spdlog::critical("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kERROR:
                    spdlog::error("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kWARNING:
                    spdlog::warn("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kINFO:
                    spdlog::info("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kVERBOSE:
                    spdlog::debug("[lidar_net] {}", msg);
                    break;
            }
        } catch (const std::exception& exc) {
            std::cerr << exc.what();
        }
    }
};
```

2. 伟康spdlog_ws
```cpp
class UserLogger : public lidar_net::LidarNetLogger {
    void log(lidar_net::LidarNetLogger::Severity severity, const std::string_view msg) noexcept override {
        try {
            switch (severity) {
                case lidar_net::LidarNetLogger::Severity::kINTERNAL_ERROR:
                    TLOGF("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kERROR:
                    TLOGE("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kWARNING:
                    TLOGW("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kINFO:
                    TLOGI("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kVERBOSE:
                    TLOGD("[lidar_net] {}", msg);
                    break;
            }
        } catch (const std::exception& exc) {
            std::cerr << exc.what();
        }
    }
};
```

3. glog
```cpp
class GlogUserLogger : public lidar_net::LidarNetLogger {
    void log(lidar_net::LidarNetLogger::Severity severity, const std::string_view msg) noexcept override {
        try {
            switch (severity) {
                case lidar_net::LidarNetLogger::Severity::kINTERNAL_ERROR:
                    LOG(FATAL) << "[lidar_net] " << msg;
                    break;
                case lidar_net::LidarNetLogger::Severity::kERROR:
                    LOG(ERROR) << "[lidar_net] " << msg;
                    break;
                case lidar_net::LidarNetLogger::Severity::kWARNING:
                    LOG(WARNING) << "[lidar_net] " << msg;
                    break;
                case lidar_net::LidarNetLogger::Severity::kINFO:
                    LOG(INFO) << "[lidar_net] " << msg;
                    break;
                case lidar_net::LidarNetLogger::Severity::kVERBOSE:
                    VLOG(1) << "[lidar_net] " << msg;
                    break;
            }
        } catch (const std::exception& exc) {
            std::cerr << exc.what();
        }
    }
};
```

4. jian
TODO

### 配置文件模板
LidarNetSDK需要一个配置文件来初始化。配置文件的格式为toml, json或者yaml格式的配置文件。

配置文件需要固定的结构，如下：

```toml
#################################################
# detector 检测算法
#################################################
[dsvt]
out_types = ["VEHICLE", "VEHICLE", "VEHICLE", "VEHICLE", "VEHICLE"]

[dsvt.params]
max_num_pillars = 12000
max_num_points_per_pillar = 16
pillar_x_resolution = 0.35
pillar_y_resolution = 0.35
pillar_z_resolution = 7.0
range_x_max = 117.0
range_x_min = 5.0
range_y_max = 28.0
range_y_min = -28.0
range_z_max = 5.0
range_z_min = -2.0

iou_threshold = 0.1
num_classes = 5
score_threshold = 0.5

input_points_channel = 3
max_input_points = 50000

[dsvt.network]
engine = "/opt/deepways_dsvt_model/model/livox_20211217_epoch99_sim_VoxelGenerator.trt"

[dsvt.network.inputs]
voxelgen_input_num_points.dtype = 'int32'
voxelgen_input_num_points.shape = [1]
voxelgen_input_points.dtype = 'float32'
voxelgen_input_points.shape = [1, 50000, 3]

[dsvt.network.outputs]
output.dtype = 'float32'
output.shape = [1, 1050, 12]
```

同理，有yaml
```yaml
dsvt:
    network:
    engine: /opt/deepways_dsvt_model/model/livox_20211217_epoch99_sim_VoxelGenerator.trt
    inputs:
        voxelgen_input_num_points:
        dtype: int32
        shape:
        - 1
        voxelgen_input_points:
        dtype: float32
        shape:
        - 1
        - 50000
        - 3
    outputs:
        output:
        dtype: float32
        shape:
        - 1
        - 1050
        - 12
    params:
        input_points_channel: 3
        iou_threshold: 0.1
        max_input_points: 50000
        max_num_pillars: 12000
        max_num_points_per_pillar: 16
        num_classes: 5
        pillar_x_resolution: 0.35
        pillar_y_resolution: 0.35
        pillar_z_resolution: 7.0
        range_x_max: 117.0
        range_x_min: 5.0
        range_y_max: 28.0
        range_y_min: -28.0
        range_z_max: 5.0
        range_z_min: -2.0
        score_threshold: 0.5
    out_types:
    - VEHICLE
    - VEHICLE
    - VEHICLE
    - VEHICLE
    - VEHICLE
```

以及，json
```json
{
    "dsvt": {
        "network": {
            "engine": "/opt/deepways_dsvt_model/model/livox_20211217_epoch99_sim_VoxelGenerator.trt",
            "inputs": {
                "voxelgen_input_num_points": {
                    "dtype": "int32",
                    "shape": [
                        1
                    ]
                },
                "voxelgen_input_points": {
                    "dtype": "float32",
                    "shape": [
                        1,
                        50000,
                        3
                    ]
                }
            },
            "outputs": {
                "output": {
                    "dtype": "float32",
                    "shape": [
                        1,
                        1050,
                        12
                    ]
                }
            }
        },
        "params": {
            "input_points_channel": 3,
            "iou_threshold": 0.1,
            "max_input_points": 50000,
            "max_num_pillars": 12000,
            "max_num_points_per_pillar": 16,
            "num_classes": 5,
            "pillar_x_resolution": 0.35,
            "pillar_y_resolution": 0.35,
            "pillar_z_resolution": 7.0,
            "range_x_max": 117.0,
            "range_x_min": 5.0,
            "range_y_max": 28.0,
            "range_y_min": -28.0,
            "range_z_max": 5.0,
            "range_z_min": -2.0,
            "score_threshold": 0.5,
        },
        "out_types": [
            "VEHICLE",
            "VEHICLE",
            "VEHICLE",
            "VEHICLE",
            "VEHICLE"
        ]
    }
}
```

### 配置文件解析

含义解析:
1. `dsvt`：配置文件的入口，所有的配置都在这个下面
2. `network`：神经网络的配置
3. `params`：超参配置
4. `out_types`：输出类型配置，影响用户得到的结果中目标类型

network段：
1. `engine`：神经网络的模型文件，trt格式
2. `inputs`：输入张量的配置，包括输入张量的名字，数据类型，形状
3. `outputs`：输出张量的配置，包括输出张量的名字，数据类型，形状

params段：
1. `input_points_channel`：输入点云的通道数，一般为3；带有intensity为4；最多支持到5
2. `iou_threshold`：nms时的iou阈值
3. `max_input_points`：输入点云的最大点数
4. `max_num_pillars`：VoxelGenerator生成最大的pillar数
5. `max_num_points_per_pillar`：VoxelGenerator生成的pillar中最大的点数
6. `num_classes`：目标类型的数量
7. `pillar_x_resolution`：pillar在x方向的分辨率
8. `pillar_y_resolution`：pillar在y方向的分辨率
9. `pillar_z_resolution`：pillar在z方向的分辨率
10. `range_x_max`：输入点云的x方向的最大值
11. `range_x_min`：输入点云的x方向的最小值
12. `range_y_max`：输入点云的y方向的最大值
13. `range_y_min`：输入点云的y方向的最小值
14. `range_z_max`：输入点云的z方向的最大值
15. `range_z_min`：输入点云的z方向的最小值
16. `score_threshold`：目标置信度的阈值

out_types段：
列表，逐个指明输出的目标类型，目前支持的目标类型见自定义类别系统


## 推理接口

```cpp
// Raw C data input
std::vector<base::BoundingBox> Processing(void* points, const size_t points_count);
void Processing(void* points, const size_t points_count, std::vector<base::BoundingBox>& bboxes);

// Eigen data input (RowMajor)
std::vector<base::BoundingBox> ProcessingEigenRow(ConstRefRow points);
void ProcessingEigenRow(ConstRefRow points, std::vector<base::BoundingBox>& bboxes);

// Eigen data input (ColMajor)
std::vector<base::BoundingBox> ProcessingEigenCol(ConstRef points);
void ProcessingEigenCol(ConstRef points, std::vector<base::BoundingBox>& bboxes);
```

### 输入
1. Raw C data input
输入：指针、点数
2. Eigen data input (RowMajor)
输入：eigen数组（行优先）
3. Eigen data input (ColMajor)
输入：eigen数组（列优先）

### 自定义类别系统
```cpp
UNKNOWN = 0,  ///< 未知; rule-based 方法结果
/********************************************************/
VEHICLE = 1,     ///< 车
BYCYCLE = 2,     ///< 自行车、摩托车
PEDESTRIAN = 3,  ///< 行人
/********************************************************/
ART = 10,               ///< ART (通用类型)
ART_NO_TRAILER = 11,    ///< 不带箱ART
ART_SEMI_TRAILER = 12,  ///< 带20吋箱ART
ART_FULL_TRAILER = 13,  ///< 带40吋箱ART
/********************************************************/
TRUCK = 20,               ///< 卡车 (通用类型)
TRUCK_HEAD = 21,          ///< 卡车头
TRAILER = 22,             ///< 挂车
TRUCK_NO_TRAILER = 23,    ///< 不带箱挂车
TRUCK_SEMI_TRAILER = 24,  ///< 带20吋箱挂车
TRUCK_FULL_TRAILER = 25,  ///< 带40吋箱挂车
/********************************************************/
ALIEN_VEHICLE = 30,         ///< 异形车 (通用类型)
FORKLIFT = 31,              ///< 叉车
ECCENTRIC_TRUCK_HEAD = 32,  ///< 偏头车
MOBILE_CRANE = 33,          ///< 流机
```
在config文件的 `out_types` 中，指明网络输出的结果中各类对应的ObjectType(使用字符串列表)，输出的结果中物体将按照config中的定义类别进行赋值。

注意：out_types中的类别必须是上述定义的类别; out_types列表长度必须和网络输出的类别数一致。

### 返回

1. `std::vector<base::BoundingBox>`
可以传入用户所有的BBox列表，或者返回一个新的BBox列表
2. `enum class ObjType`
目标类型枚举型

