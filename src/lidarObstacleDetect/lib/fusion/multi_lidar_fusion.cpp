#include "multi_lidar_fusion.h"
#include <pcl_conversions/pcl_conversions.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace lidar_fusion {

namespace {
constexpr double kPi = 3.14159265358979323846;

double normalizeDeg(double deg) {
    while (deg > 180.0) deg -= 360.0;
    while (deg <= -180.0) deg += 360.0;
    return deg;
}

bool isAngleInSector(double angle_deg, double min_deg, double max_deg) {
    const double angle = normalizeDeg(angle_deg);
    const double min_angle = normalizeDeg(min_deg);
    const double max_angle = normalizeDeg(max_deg);

    if (min_angle <= max_angle) {
        return angle >= min_angle && angle <= max_angle;
    }

    return angle >= min_angle || angle <= max_angle;
}
} // namespace

// [辅助函数] 方便添加盒子
void addBox(std::vector<ParamsBounds>& list, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
    ParamsBounds b;
    b.x_min = x_min; b.x_max = x_max;
    b.y_min = y_min; b.y_max = y_max;
    b.z_min = z_min; b.z_max = z_max;
    list.push_back(b);
}

MultiLidarFusion::MultiLidarFusion(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
    // 1. 初始化融合后的点云指针
    fused_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);

    // 2. 读取话题参数
    pnh.param("parent_pc_topic", parent_pc_topic_, std::string("/rslidar/middle/rslidar_points"));
    pnh.param("child_pc_topic1", child1_pc_topic_, std::string("/rslidar/right/rslidar_points"));
    pnh.param("child_pc_topic2", child2_pc_topic_, std::string("/rslidar/left/rslidar_points"));
    pnh.param("child_pc_topic3", child3_pc_topic_, std::string("/rslidar/back/rslidar_points"));
    pnh.param("use_child1_lidar", use_child1_lidar_, false);

    // 3. 读取变换参数 (从 lidar_detection_track.yaml)
    // (注意：您的YAML文件中的参数名与代码中的不完全一致，这里以代码为准)
    pnh.param("ppc_to_vehicle/x", ppc_to_vehicle_.x, 5.6);
    pnh.param("ppc_to_vehicle/y", ppc_to_vehicle_.y, 0.0);
    pnh.param("ppc_to_vehicle/z", ppc_to_vehicle_.z, 2.0);
    pnh.param("ppc_to_vehicle/roll", ppc_to_vehicle_.roll, 0.04);
    pnh.param("ppc_to_vehicle/pitch", ppc_to_vehicle_.pitch, 0.02);
    pnh.param("ppc_to_vehicle/yaw", ppc_to_vehicle_.yaw, 1.65);

    pnh.param("cpc1_to_ppc/x", cpc1_to_ppc_.x, -0.9);
    pnh.param("cpc1_to_ppc/y", cpc1_to_ppc_.y, 3.7);
    pnh.param("cpc1_to_ppc/z", cpc1_to_ppc_.z, -1.95);
    pnh.param("cpc1_to_ppc/roll", cpc1_to_ppc_.roll, 1.54);
    pnh.param("cpc1_to_ppc/pitch", cpc1_to_ppc_.pitch, -1.938);
    pnh.param("cpc1_to_ppc/yaw", cpc1_to_ppc_.yaw, -1.593);

    pnh.param("cpc2_to_ppc/x", cpc2_to_ppc_.x, 0.0);
    pnh.param("cpc2_to_ppc/y", cpc2_to_ppc_.y, 0.0);
    pnh.param("cpc2_to_ppc/z", cpc2_to_ppc_.z, 0.0);
    pnh.param("cpc2_to_ppc/roll", cpc2_to_ppc_.roll, 0.0);
    pnh.param("cpc2_to_ppc/pitch", cpc2_to_ppc_.pitch, 0.0);
    pnh.param("cpc2_to_ppc/yaw", cpc2_to_ppc_.yaw, 0.0);

    pnh.param("cpc3_to_ppc/x", cpc3_to_ppc_.x, 0.0);
    pnh.param("cpc3_to_ppc/y", cpc3_to_ppc_.y, 0.0);
    pnh.param("cpc3_to_ppc/z", cpc3_to_ppc_.z, 0.0);
    pnh.param("cpc3_to_ppc/roll", cpc3_to_ppc_.roll, 0.0);
    pnh.param("cpc3_to_ppc/pitch", cpc3_to_ppc_.pitch, 0.0);
    pnh.param("cpc3_to_ppc/yaw", cpc3_to_ppc_.yaw, 0.0);

    // +++  读取滤波参数 +++
    pnh.param("use_external_bounds_filter", use_external_bounds_filter_, true);
    pnh.param("use_internal_bounds_filter", use_internal_bounds_filter_, true);
    loadChildFilterParams(pnh, "child1", child1_filter_);
    loadChildFilterParams(pnh, "child2", child2_filter_);
    loadChildFilterParams(pnh, "child3", child3_filter_);

    // (从YAML加载外部边界)
    ParamsBounds ext_temp;
    pnh.param("external_bounds/x_min", ext_temp.x_min, -20.0f);
    pnh.param("external_bounds/x_max", ext_temp.x_max, 50.0f);
    pnh.param("external_bounds/y_min", ext_temp.y_min, -20.0f);
    pnh.param("external_bounds/y_max", ext_temp.y_max, 20.0f);
    pnh.param("external_bounds/z_min", ext_temp.z_min, -2.0f);
    pnh.param("external_bounds/z_max", ext_temp.z_max, 2.0f);
    external_bounds_list_.push_back(ext_temp);

    // (从YAML加载内部边界)
    ParamsBounds int_temp;
    pnh.param("internal_bounds/x_min", int_temp.x_min, -6.0f);
    pnh.param("internal_bounds/x_max", int_temp.x_max, 1.5f);
    pnh.param("internal_bounds/y_min", int_temp.y_min, -1.0f);
    pnh.param("internal_bounds/y_max", int_temp.y_max, 1.0f);
    pnh.param("internal_bounds/z_min", int_temp.z_min, -2.0f);
    pnh.param("internal_bounds/z_max", int_temp.z_max, 2.5f);
    internal_bounds_list_.push_back(int_temp);

    //单独添加车身轮廓框（float x_min, float x_max, float y_min, float y_max, float z_min, float z_max）
    // addBox(internal_bounds_list_, 6.0, 6.3, 1.2, 1.5, 1.5, 2.5);//后视镜1
    // addBox(internal_bounds_list_, 6.0, 6.3, -1.5, -1.2, 1.5, 2.5);//后视镜2

    if (!internal_bounds_list_.empty()) {
        CUDA_CHECK(cudaMalloc(&d_internal_bounds_, internal_bounds_list_.size() * sizeof(ParamsBounds)));
        CUDA_CHECK(cudaMemcpy(d_internal_bounds_, internal_bounds_list_.data(), 
                   internal_bounds_list_.size() * sizeof(ParamsBounds), cudaMemcpyHostToDevice));
    }
    
    if (!external_bounds_list_.empty()) {
        CUDA_CHECK(cudaMalloc(&d_external_bounds_, external_bounds_list_.size() * sizeof(ParamsBounds)));
        CUDA_CHECK(cudaMemcpy(d_external_bounds_, external_bounds_list_.data(), 
                   external_bounds_list_.size() * sizeof(ParamsBounds), cudaMemcpyHostToDevice));
    }

    // 4. 初始化CUDA
    CUDA_CHECK(cudaStreamCreate(&stream_));
    ROS_INFO("CUDA Stream created for MultiLidarFusion.");

    // 5. 预先计算变换矩阵 (Matrix4f)
    auto tf_parent_to_vehicle = createTransformMatrix(ppc_to_vehicle_);
    auto tf_child1_to_parent = createTransformMatrix(cpc1_to_ppc_);
    auto tf_child2_to_parent = createTransformMatrix(cpc2_to_ppc_);
    auto tf_child3_to_parent = createTransformMatrix(cpc3_to_ppc_);

    // 最终变换：全部到车辆坐标系 (base_link)
    transform_parent_ = tf_parent_to_vehicle;
    transform_child1_ = tf_parent_to_vehicle * tf_child1_to_parent;
    transform_child2_ = tf_parent_to_vehicle * tf_child2_to_parent;
    transform_child3_ = tf_parent_to_vehicle * tf_child3_to_parent;

    // 6. 初始化消息同步器
    parent_pc_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, parent_pc_topic_, 10));
    child2_pc_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, child2_pc_topic_, 10));
    child3_pc_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, child3_pc_topic_, 10));

    if (use_child1_lidar_) {
        child1_pc_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, child1_pc_topic_, 10));
        sync_all_.reset(new message_filters::Synchronizer<SyncPolicyAll>(SyncPolicyAll(10),
            *parent_pc_sub_, *child1_pc_sub_, *child2_pc_sub_, *child3_pc_sub_));
        sync_all_->registerCallback(boost::bind(&MultiLidarFusion::syncedCallback, this, _1, _2, _3, _4));
        ROS_INFO("MultiLidarFusion (CUDA) initialized with child1 lidar enabled. Waiting for 4 synchronized topics...");
    } else {
        sync_without_child1_.reset(new message_filters::Synchronizer<SyncPolicyWithoutChild1>(SyncPolicyWithoutChild1(10),
            *parent_pc_sub_, *child2_pc_sub_, *child3_pc_sub_));
        sync_without_child1_->registerCallback(boost::bind(&MultiLidarFusion::syncedCallbackWithoutChild1, this, _1, _2, _3));
        ROS_WARN("MultiLidarFusion (CUDA) initialized with child1/right lidar disabled. Skipping topic: %s",
                 child1_pc_topic_.c_str());
    }
}


// +++ 添加析构函数 +++
MultiLidarFusion::~MultiLidarFusion() {
    ROS_INFO("Destroying CUDA stream for MultiLidarFusion.");
    CUDA_CHECK(cudaStreamDestroy(stream_));
    if (d_internal_bounds_) cudaFree(d_internal_bounds_);
    if (d_external_bounds_) cudaFree(d_external_bounds_);
}

Eigen::Matrix4f MultiLidarFusion::createTransformMatrix(const TransformParams& params) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
    // 1. 初始化欧拉角 (Z-Y-X，即RPY)
    // 注意: TransformParams 使用 double，而 Eigen::Vector3f 通常使用 float
    Eigen::Vector3f rpy(static_cast<float>(params.roll), 
                          static_cast<float>(params.pitch), 
                          static_cast<float>(params.yaw));

    // 2. 欧拉角转换为旋转矩阵 (Z-Y-X 顺序)
    Eigen::Matrix3f rotation_matrix3;
    rotation_matrix3 = Eigen::AngleAxisf(rpy[2], Eigen::Vector3f::UnitZ()) * // Yaw
                       Eigen::AngleAxisf(rpy[1], Eigen::Vector3f::UnitY()) * // Pitch
                       Eigen::AngleAxisf(rpy[0], Eigen::Vector3f::UnitX());  // Roll

    // 3. 将 3x3 旋转矩阵赋值给 4x4 变换矩阵的旋转部分
    transform.block<3,3>(0,0) = rotation_matrix3;

    // 4. 设置平移部分 
    transform(0,3) = params.x;
    transform(1,3) = params.y;
    transform(2,3) = params.z;
    
    return transform;
}

void MultiLidarFusion::loadChildFilterParams(ros::NodeHandle& pnh,
                                             const std::string& prefix,
                                             ChildFilterParams& params) {
    pnh.param(prefix + "_use_sector_filter", params.use_sector_filter, false);
    pnh.param(prefix + "_sector/min_deg", params.min_deg, -180.0);
    pnh.param(prefix + "_sector/max_deg", params.max_deg, 180.0);
    pnh.param(prefix + "_use_distance_filter", params.use_distance_filter, false);
    pnh.param(prefix + "_distance/min_dist", params.min_dist, 0.0);
    pnh.param(prefix + "_distance/max_dist", params.max_dist, 1.0e9);

    if (params.min_dist > params.max_dist) {
        ROS_WARN("%s_distance min_dist %.3f is greater than max_dist %.3f, swapping values.",
                 prefix.c_str(), params.min_dist, params.max_dist);
        std::swap(params.min_dist, params.max_dist);
    }
}

void MultiLidarFusion::applyChildFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                        const ChildFilterParams& params) const {
    if (!cloud || cloud->empty() || (!params.use_sector_filter && !params.use_distance_filter)) {
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
    filtered->points.reserve(cloud->points.size());

    for (const auto& point : cloud->points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }

        if (params.use_sector_filter) {
            const double angle_deg = std::atan2(point.y, point.x) * 180.0 / kPi;
            if (!isAngleInSector(angle_deg, params.min_deg, params.max_deg)) {
                continue;
            }
        }

        if (params.use_distance_filter) {
            if (point.x < params.min_dist || point.x > params.max_dist) {
                continue;
            }
        }

        filtered->points.push_back(point);
    }

    filtered->width = static_cast<uint32_t>(filtered->points.size());
    filtered->height = 1;
    filtered->is_dense = true;
    filtered->header = cloud->header;
    cloud = filtered;
}

// --- 移除 transformCloud (CPU) ---
// (该函数现在由 cuda_transform.cu 中的 transformKernel 替代)

// --- 移除所有单独的 *CloudCallback 函数 ---

// +++ 实现同步回调函数 +++
void MultiLidarFusion::syncedCallback(
    const sensor_msgs::PointCloud2::ConstPtr& parent_msg,
    const sensor_msgs::PointCloud2::ConstPtr& child1_msg,
    const sensor_msgs::PointCloud2::ConstPtr& child2_msg,
    const sensor_msgs::PointCloud2::ConstPtr& child3_msg)
{
    processClouds(parent_msg, child1_msg, child2_msg, child3_msg);
}

void MultiLidarFusion::syncedCallbackWithoutChild1(
    const sensor_msgs::PointCloud2::ConstPtr& parent_msg,
    const sensor_msgs::PointCloud2::ConstPtr& child2_msg,
    const sensor_msgs::PointCloud2::ConstPtr& child3_msg)
{
    processClouds(parent_msg, sensor_msgs::PointCloud2::ConstPtr(), child2_msg, child3_msg);
}

void MultiLidarFusion::processClouds(
    const sensor_msgs::PointCloud2::ConstPtr& parent_msg,
    const sensor_msgs::PointCloud2::ConstPtr& child1_msg,
    const sensor_msgs::PointCloud2::ConstPtr& child2_msg,
    const sensor_msgs::PointCloud2::ConstPtr& child3_msg)
{
    // 1. ROS Msg -> PCL CPU Cloud (临时)
    // (在GPU管线中，我们也可以直接操作ROS Msg的data，但fromROSMsg更安全)
    pcl::PointCloud<pcl::PointXYZI>::Ptr cpu_pc1(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cpu_pc2(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cpu_pc3(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cpu_pc4(new pcl::PointCloud<pcl::PointXYZI>());
    
    pcl::fromROSMsg(*parent_msg, *cpu_pc1);
    if (child1_msg) {
        pcl::fromROSMsg(*child1_msg, *cpu_pc2);
        applyChildFilter(cpu_pc2, child1_filter_);
    }
    pcl::fromROSMsg(*child2_msg, *cpu_pc3);
    applyChildFilter(cpu_pc3, child2_filter_);
    pcl::fromROSMsg(*child3_msg, *cpu_pc4);
    applyChildFilter(cpu_pc4, child3_filter_);

    size_t n_pc1 = cpu_pc1->points.size();
    size_t n_pc2 = cpu_pc2->points.size();
    size_t n_pc3 = cpu_pc3->points.size();
    size_t n_pc4 = cpu_pc4->points.size();
    size_t n_fused = n_pc1 + n_pc2 + n_pc3 + n_pc4;

    if (n_fused == 0) {
        ROS_WARN("Empty cloud in syncedCallback, skipping fusion.");
        return;
    }

    // --- 2. GPU 管线开始 ---

    // 2.1 定义设备指针
    pcl::PointXYZI *d_pc1 = nullptr, *d_pc2 = nullptr, *d_pc3 = nullptr, *d_pc4 = nullptr; // 原始
    pcl::PointXYZI *d_trans_1 = nullptr, *d_trans_2 = nullptr, *d_trans_3 = nullptr, *d_trans_4 = nullptr; // 变换后
    float *d_mat1 = nullptr, *d_mat2 = nullptr, *d_mat3 = nullptr, *d_mat4 = nullptr; // 矩阵
    pcl::PointXYZI *d_fused = nullptr; // 融合后

    // 2.2 分配设备内存
    // (注意：为简化，这里使用同步 Malloc。更优化的版本会使用内存池)
    if (n_pc1 > 0) CUDA_CHECK(cudaMalloc(&d_pc1, n_pc1 * sizeof(pcl::PointXYZI)));
    if (n_pc2 > 0) CUDA_CHECK(cudaMalloc(&d_pc2, n_pc2 * sizeof(pcl::PointXYZI)));
    if (n_pc3 > 0) CUDA_CHECK(cudaMalloc(&d_pc3, n_pc3 * sizeof(pcl::PointXYZI)));
    if (n_pc4 > 0) CUDA_CHECK(cudaMalloc(&d_pc4, n_pc4 * sizeof(pcl::PointXYZI)));

    if (n_pc1 > 0) CUDA_CHECK(cudaMalloc(&d_trans_1, n_pc1 * sizeof(pcl::PointXYZI)));
    if (n_pc2 > 0) CUDA_CHECK(cudaMalloc(&d_trans_2, n_pc2 * sizeof(pcl::PointXYZI)));
    if (n_pc3 > 0) CUDA_CHECK(cudaMalloc(&d_trans_3, n_pc3 * sizeof(pcl::PointXYZI)));
    if (n_pc4 > 0) CUDA_CHECK(cudaMalloc(&d_trans_4, n_pc4 * sizeof(pcl::PointXYZI)));
    
    if (n_pc1 > 0) CUDA_CHECK(cudaMalloc(&d_mat1, 16 * sizeof(float)));
    if (n_pc2 > 0) CUDA_CHECK(cudaMalloc(&d_mat2, 16 * sizeof(float)));
    if (n_pc3 > 0) CUDA_CHECK(cudaMalloc(&d_mat3, 16 * sizeof(float)));
    if (n_pc4 > 0) CUDA_CHECK(cudaMalloc(&d_mat4, 16 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_fused, n_fused * sizeof(pcl::PointXYZI)));

    // 2.3 准备行优先(RowMajor)的矩阵
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> mat1_row_major = transform_parent_;
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> mat2_row_major = transform_child1_;
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> mat3_row_major = transform_child2_;
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> mat4_row_major = transform_child3_;

    // 2.4 [异步] 上传所有数据 (HtoD)
    if (n_pc1 > 0) CUDA_CHECK(cudaMemcpyAsync(d_pc1, cpu_pc1->points.data(), n_pc1 * sizeof(pcl::PointXYZI), cudaMemcpyHostToDevice, stream_));
    if (n_pc2 > 0) CUDA_CHECK(cudaMemcpyAsync(d_pc2, cpu_pc2->points.data(), n_pc2 * sizeof(pcl::PointXYZI), cudaMemcpyHostToDevice, stream_));
    if (n_pc3 > 0) CUDA_CHECK(cudaMemcpyAsync(d_pc3, cpu_pc3->points.data(), n_pc3 * sizeof(pcl::PointXYZI), cudaMemcpyHostToDevice, stream_));
    if (n_pc4 > 0) CUDA_CHECK(cudaMemcpyAsync(d_pc4, cpu_pc4->points.data(), n_pc4 * sizeof(pcl::PointXYZI), cudaMemcpyHostToDevice, stream_));

    if (n_pc1 > 0) CUDA_CHECK(cudaMemcpyAsync(d_mat1, mat1_row_major.data(), 16 * sizeof(float), cudaMemcpyHostToDevice, stream_));
    if (n_pc2 > 0) CUDA_CHECK(cudaMemcpyAsync(d_mat2, mat2_row_major.data(), 16 * sizeof(float), cudaMemcpyHostToDevice, stream_));
    if (n_pc3 > 0) CUDA_CHECK(cudaMemcpyAsync(d_mat3, mat3_row_major.data(), 16 * sizeof(float), cudaMemcpyHostToDevice, stream_));
    if (n_pc4 > 0) CUDA_CHECK(cudaMemcpyAsync(d_mat4, mat4_row_major.data(), 16 * sizeof(float), cudaMemcpyHostToDevice, stream_));

    // 2.5 [异步] 执行GPU坐标变换 (DtoD)
    transformPointCloudCUDA_async(d_pc1, d_trans_1, d_mat1, n_pc1, stream_);
    transformPointCloudCUDA_async(d_pc2, d_trans_2, d_mat2, n_pc2, stream_);
    transformPointCloudCUDA_async(d_pc3, d_trans_3, d_mat3, n_pc3, stream_);
    transformPointCloudCUDA_async(d_pc4, d_trans_4, d_mat4, n_pc4, stream_);

    // 2.6 [异步] 执行GPU融合 (DtoD)
    launch_fuse_clouds(d_fused, 
                       d_trans_1, n_pc1, d_trans_2, n_pc2, d_trans_3, n_pc3, d_trans_4, n_pc4,
                       4, stream_); // 禁用的雷达以 0 个点参与，保持输入槽位固定。

    // 2.7 [异步] 释放不再需要的中间缓冲区
    if (d_pc1) CUDA_CHECK(cudaFreeAsync(d_pc1, stream_));
    if (d_pc2) CUDA_CHECK(cudaFreeAsync(d_pc2, stream_));
    if (d_pc3) CUDA_CHECK(cudaFreeAsync(d_pc3, stream_));
    if (d_pc4) CUDA_CHECK(cudaFreeAsync(d_pc4, stream_));
    if (d_trans_1) CUDA_CHECK(cudaFreeAsync(d_trans_1, stream_));
    if (d_trans_2) CUDA_CHECK(cudaFreeAsync(d_trans_2, stream_));
    if (d_trans_3) CUDA_CHECK(cudaFreeAsync(d_trans_3, stream_));
    if (d_trans_4) CUDA_CHECK(cudaFreeAsync(d_trans_4, stream_));
    if (d_mat1) CUDA_CHECK(cudaFreeAsync(d_mat1, stream_));
    if (d_mat2) CUDA_CHECK(cudaFreeAsync(d_mat2, stream_));
    if (d_mat3) CUDA_CHECK(cudaFreeAsync(d_mat3, stream_));
    if (d_mat4) CUDA_CHECK(cudaFreeAsync(d_mat4, stream_));

    //2.8 边界滤波管线
    pcl::PointXYZI* d_current_in = d_fused;     // 当前处理的输入
    size_t    n_current_in = n_fused;     // 当前处理的输入点数
    pcl::PointXYZI* d_current_out = nullptr;    // 当前处理的输出

    unsigned int* d_out_index; // 原子计数器
    CUDA_CHECK(cudaMalloc(&d_out_index, sizeof(unsigned int)));

    unsigned int n_after_filter = n_current_in; // 跟踪CPU上的点数

        // 2.8.1 [异步] 外部边界滤波 (保留内部)
    if (use_external_bounds_filter_ && n_current_in > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_current_out, n_current_in * sizeof(pcl::PointXYZI))); 
        CUDA_CHECK(cudaMemsetAsync(d_out_index, 0, sizeof(unsigned int), stream_)); // 重置计数器

        launch_filter_points(d_current_in, n_current_in, d_current_out, d_out_index,
                             d_external_bounds_, external_bounds_list_.size(), false, stream_); // false = 保留内部

        // [同步] 获取输出点数
        CUDA_CHECK(cudaMemcpyAsync(&n_after_filter, d_out_index, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_)); // *** 需要同步以获取 n_after_filter

        CUDA_CHECK(cudaFreeAsync(d_current_in, stream_)); // 释放上一阶段的内存
        d_current_in = d_current_out;       // 输出成为下一阶段的输入
        n_current_in = n_after_filter;
    }
    
    // 2.8.2 [异步] 内部边界滤波 (保留外部)
    if (use_internal_bounds_filter_ && n_current_in > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_current_out, n_current_in * sizeof(pcl::PointXYZI))); 
        CUDA_CHECK(cudaMemsetAsync(d_out_index, 0, sizeof(unsigned int), stream_)); // 重置计数器

        launch_filter_points(d_current_in, n_current_in, d_current_out, d_out_index,
                             d_internal_bounds_, internal_bounds_list_.size(), true, stream_); // true = 保留外部 (滤除内部)

        // [同步] 获取输出点数
        CUDA_CHECK(cudaMemcpyAsync(&n_after_filter, d_out_index, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_)); // *** 需要同步

        CUDA_CHECK(cudaFreeAsync(d_current_in, stream_)); // 释放上一阶段
        d_current_in = d_current_out;       
        n_current_in = n_after_filter;
    }

    // 释放计数器
    CUDA_CHECK(cudaFreeAsync(d_out_index, stream_));

    // 2.9 [异步] 下载最终结果 (DtoH)
    pcl::PointCloud<pcl::PointXYZI>::Ptr new_fused_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    // 使用滤波后的点数 n_current_in
    new_fused_cloud->points.resize(n_current_in);

    if (n_current_in > 0)
    {
        // 从 d_current_in (滤波管线的最终输出) 下载
        CUDA_CHECK(cudaMemcpyAsync(new_fused_cloud->points.data(), d_current_in, n_current_in * sizeof(pcl::PointXYZI), cudaMemcpyDeviceToHost, stream_));
    }

    // 2.10 [异步] 释放最后的GPU点云内存
    CUDA_CHECK(cudaFreeAsync(d_current_in, stream_)); // 释放 d_current_in

    // 2.11 [同步] 等待所有操作完成
    CUDA_CHECK(cudaStreamSynchronize(stream_)); 

    // 3. 更新供外部获取的点云
    new_fused_cloud->width = n_current_in; // 使用滤波后的点数
    new_fused_cloud->height = 1;
    new_fused_cloud->is_dense = true;

    {
        // 线程安全地替换掉旧的点云
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        fused_cloud_ = new_fused_cloud;
    }
}

} // namespace lidar_fusion
