#ifndef MULTI_LIDAR_FUSION_H_
#define MULTI_LIDAR_FUSION_H_

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <sensor_msgs/PointCloud2.h> 

// --- 添加同步器和CUDA头文件 ---
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cuda_runtime.h>
#include "cuda_transform.h" // 包含自己的CUDA头文件
#include <Eigen/Geometry>


namespace lidar_fusion {

struct TransformParams {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

struct ChildFilterParams {
    bool use_sector_filter = false;
    double min_deg = -180.0;
    double max_deg = 180.0;
    bool use_distance_filter = false;
    double min_dist = 0.0;
    double max_dist = 1.0e9;
};

// 定义消息同步策略
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, 
    sensor_msgs::PointCloud2, 
    sensor_msgs::PointCloud2, 
    sensor_msgs::PointCloud2
> SyncPolicyAll;

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2,
    sensor_msgs::PointCloud2,
    sensor_msgs::PointCloud2
> SyncPolicyWithoutChild1;

class MultiLidarFusion {
public:
    MultiLidarFusion(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~MultiLidarFusion();

    pcl::PointCloud<pcl::PointXYZI>::Ptr getFusedCloud() const {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        return fused_cloud_;
    }

private:

	void parentCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
	void child1CloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
	void child2CloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
	void child3CloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
    	// +++ 添加同步回调函数 +++
	    	void syncedCallback(
	        	const sensor_msgs::PointCloud2::ConstPtr& parent_msg,
	        	const sensor_msgs::PointCloud2::ConstPtr& child1_msg,
	        	const sensor_msgs::PointCloud2::ConstPtr& child2_msg,
	        	const sensor_msgs::PointCloud2::ConstPtr& child3_msg
	    	);
            void syncedCallbackWithoutChild1(
                const sensor_msgs::PointCloud2::ConstPtr& parent_msg,
                const sensor_msgs::PointCloud2::ConstPtr& child2_msg,
                const sensor_msgs::PointCloud2::ConstPtr& child3_msg
            );
            void processClouds(
                const sensor_msgs::PointCloud2::ConstPtr& parent_msg,
                const sensor_msgs::PointCloud2::ConstPtr& child1_msg,
                const sensor_msgs::PointCloud2::ConstPtr& child2_msg,
                const sensor_msgs::PointCloud2::ConstPtr& child3_msg
            );
            void loadChildFilterParams(ros::NodeHandle& pnh,
                                       const std::string& prefix,
                                       ChildFilterParams& params);
            void applyChildFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                  const ChildFilterParams& params) const;
    Eigen::Matrix4f createTransformMatrix(const TransformParams& params);
    void transformCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_cloud,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& out_cloud,
                       const Eigen::Matrix4f& transform);

// +++ 添加消息过滤器 +++
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> parent_pc_sub_;
	    std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> child1_pc_sub_;
	    std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> child2_pc_sub_;
	    std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> child3_pc_sub_;
	    std::unique_ptr<message_filters::Synchronizer<SyncPolicyAll>> sync_all_;
	    std::unique_ptr<message_filters::Synchronizer<SyncPolicyWithoutChild1>> sync_without_child1_;

    // 融合后的点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr fused_cloud_;

// +++ 添加CUDA Stream +++
    cudaStream_t stream_;
    
    // +++ 添加预先计算的变换矩阵 +++
    Eigen::Matrix4f transform_parent_;
    Eigen::Matrix4f transform_child1_;
    Eigen::Matrix4f transform_child2_;
    Eigen::Matrix4f transform_child3_;
    
    // 变换参数
    TransformParams ppc_to_vehicle_;
    TransformParams cpc1_to_ppc_;
    TransformParams cpc2_to_ppc_;
    TransformParams cpc3_to_ppc_;

    // 话题名称
    std::string parent_pc_topic_;
    std::string child1_pc_topic_;
    std::string child2_pc_topic_;
    std::string child3_pc_topic_;

    // +++ 滤波参数 +++
	    bool use_external_bounds_filter_;
	    bool use_internal_bounds_filter_;
        bool use_child1_lidar_;
        ChildFilterParams child1_filter_;
        ChildFilterParams child2_filter_;
        ChildFilterParams child3_filter_;

    // 使用 vector 存储多个盒子 (CPU端)
    std::vector<ParamsBounds> external_bounds_list_;
    std::vector<ParamsBounds> internal_bounds_list_;

    ParamsBounds* d_external_bounds_ = nullptr;
    ParamsBounds* d_internal_bounds_ = nullptr;
    // ++++++++++++++++++++++++++

    // 保护点云数据的互斥锁
    mutable std::mutex cloud_mutex_;
};

} // namespace lidar_fusion

#endif // MULTI_LIDAR_FUSION_H_
