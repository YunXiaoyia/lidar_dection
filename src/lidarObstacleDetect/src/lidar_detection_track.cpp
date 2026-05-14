#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float64.h> 
#include <sensor_msgs/point_cloud2_iterator.h> 
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/package.h>

// 标准�?
#include <fstream>
#include <sstream>
#include <chrono> 
#include <cmath>
#include <iostream>
#include <numeric>
#include <ctime>
#include <sys/time.h> 
#include <future> 
#include <thread>
#include <mutex>

// === 消息头文�?===
#include <msg_gen/PerceptionObjects.h>
#include <msg_gen/PerceptionLocalization.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <autoware_msgs/CloudClusterArray.h> 

// === SDK & Tracker 头文�?===
#include "lidar_net/lidar_net_det.h"
#include "lidar_net/object.h"
#include "tracker/tracker_manager.h"
#include "tracker/common/base.h"

// === Fusion & Cluster 头文�?===
#include "lib/fusion/multi_lidar_fusion.h" 
#include "cluster/euclideanCluster/euclideanCluster.h" 
#include "ground_detector/patchwork/patchwork.h"       
#include "bounding_box/bounding_box.h"                 

const float MAX_INTENSITY = 255.0f;
const int CHANNEL = 4; 

// 辅助计时
int64_t gtm() {
    struct timeval tm;
    gettimeofday(&tm, 0);
    int64_t re = (((int64_t)tm.tv_sec) * 1000 * 1000 + tm.tv_usec);
    return re;
}

class CustomLogger : public lidar_net::LidarNetLogger {
public:
    void log(Severity severity, const std::string_view msg) noexcept override {
        switch (severity) {
            case Severity::kERROR:   ROS_ERROR("[SDK] %s", msg.data()); break;
            case Severity::kWARNING: ROS_WARN("[SDK] %s", msg.data()); break;
            case Severity::kINFO:    ROS_INFO("[SDK] %s", msg.data()); break;
            default:                 ROS_DEBUG("[SDK] %s", msg.data()); break;
        }
    }
};

class LidarDetectorNode {
public:
    LidarDetectorNode() : 
        nh_(), 
        pnh_("~"),
        cluster_(nh_, pnh_),        
        boundingBox_(pnh_),         
        patchWork_(pnh_)            
    {
        // 1. 初始化检测器
        std::string config_file;
        pnh_.param<std::string>("config_path", config_file, "");
        if (!config_file.empty()) processConfigFile(config_file);

        detector_ = std::make_unique<lidar_net::LidarNetDetector>();
        auto logger = std::make_shared<CustomLogger>();
        if (!config_file.empty() && !detector_->Init(config_file, logger, "toml")) {
            ROS_ERROR("Failed to init detector");
            exit(-1);
        }

        std::string tracker_type;
        // 获取参数，默认使�?l_shape_track,支持通过 launch 文件动态覆盖此参数
        pnh_.param<std::string>("tracker_type", tracker_type, "l_shape_track"); 

        highway::perception::track::TrackMethod track_method;

        if (tracker_type == "l_shape_track") {
            track_method = highway::perception::track::TrackMethod::L_SHAPE;
            ROS_INFO("[Init] Selected Tracker Strategy: L_SHAPE (Best for detections without velocity)");
        } else {

            track_method = highway::perception::track::TrackMethod::SIMPLE_TRACK;
            ROS_INFO("[Init] Selected Tracker Strategy: SIMPLE_TRACK");
        }
        
        tracker_ = highway::perception::track::TargetTrackManager::create(track_method);

        if (!tracker_) {
            ROS_ERROR("Failed to create tracker instance for type: %s", tracker_type.c_str());
            exit(-1);
        }
        loadTrackerConfig(pnh_); // 加载 tracker_config.toml

        // 3. 初始�?Fusion
        fusion_ptr_.reset(new lidar_fusion::MultiLidarFusion(nh_, pnh_));

        // 4. 定时�?
        timer_ = nh_.createTimer(ros::Duration(0.1), &LidarDetectorNode::fusionCloudCallback, this);

        // 5. Publishers
        pub_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("/visualize/fusion_detection_result", 1);
        _pub_fusion_cloud = nh_.advertise<sensor_msgs::PointCloud2>("/fusion/points_fusion", 1);
        _pub_objects = nh_.advertise<msg_gen::PerceptionObjects>("/fusion/lidar_obstacle", 1);
        _pub_detected_objects = nh_.advertise<autoware_msgs::DetectedObjectArray>("/fusion/lidar_detector/objects", 1);
        _pub_noground_cloud = nh_.advertise<sensor_msgs::PointCloud2>("/fusion/nopoints_ground", 1);
        _pub_cluster_cloud = nh_.advertise<sensor_msgs::PointCloud2>("/fusion/cluster_cloud", 1);
        pnh_.param("publish_ground_debug", publish_ground_debug_, true);

        points_buffer_.resize(CHANNEL, 200000); 
        ROS_INFO("Node Started: Hybrid Mode (DSVT->Tracker + Cluster->Visual).");
    }
    // 加载追踪配置文件
    void loadTrackerConfig(ros::NodeHandle& nh) {
        std::string tracker_config_file;
        nh.param<std::string>("tracker_config_file", tracker_config_file, "");
        if (tracker_config_file.empty()) {
            std::string detector_config;
            nh.param<std::string>("config_path", detector_config, "");
            if (!detector_config.empty()) {
                size_t last_slash = detector_config.find_last_of("/\\");
                std::string config_dir = (last_slash != std::string::npos) ? detector_config.substr(0, last_slash + 1) : "";
                tracker_config_file = config_dir + "tracker_config.toml";
            }
        }
        if (tracker_config_file.empty()) return;
        processConfigFile(tracker_config_file);
        try {
            std::ifstream ifs(tracker_config_file);
            if (ifs.good()) {
                auto tracker_config = toml::parse_file(tracker_config_file);
                tracker_->Init(toml::node_view<const toml::node>(tracker_config));
            }
        } catch (...) {}
    }

    void fusionCloudCallback(const ros::TimerEvent& event) {
        int64_t tm1 = gtm();
        auto fused_cloud_ptr = fusion_ptr_->getFusedCloud(); 
        if (fused_cloud_ptr && !fused_cloud_ptr->empty()) {
            std_msgs::Header header;
            header.frame_id = "base_link"; 
            header.stamp = ros::Time::now();

            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*fused_cloud_ptr, cloud_msg);
            cloud_msg.header = header;
            _pub_fusion_cloud.publish(cloud_msg);

            processParallel(fused_cloud_ptr, cloud_msg, header);
        }
        int64_t tm2 = gtm();
        total_time_ += tm2 - tm1;
        counter_++;
        if (counter_ % 100 == 0) {
            ROS_INFO("Avg Time: %ld ms", total_time_ / 100000);
            total_time_ = 0.;
        }
    }

    // === 核心逻辑：深度学习与聚类检测并行与融合 ===
    void processParallel(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_cloud, 
                         const sensor_msgs::PointCloud2& ros_cloud,
                         const std_msgs::Header& header) 
    {

        // 1. 准备 DSVT 数据
        Eigen::MatrixXf eigen_points = prepareDSVTInput(ros_cloud);
        
        // 2. 并行启动推理
        auto future_dsvt = std::async(std::launch::async, [&]() {
            return runDSVT(eigen_points);
        });
        auto future_cluster = std::async(std::launch::async, [&]() {
            return runCluster(pcl_cloud, header);
        });

        // 3. 获取结果
        std::vector<lidar_net::base::BoundingBox> dsvt_results = future_dsvt.get();
        std::vector<autoware_msgs::CloudCluster> cluster_results = future_cluster.get();

        // 4. 运行 Tracker
        auto frame = std::make_shared<highway::perception::track::common::LidarFrame>();
        frame->timestamp = header.stamp.toSec();
        
        std::vector<lidar_net::base::BoundingBox> valid_dsvt_boxes; // 用于后续比对

        for (const auto& box : dsvt_results) {
            if (box.confidence < 0.3) continue;
            // 过滤�?Unknown，只送入�?�?骑行�?
            if (box.type == lidar_net::base::ObjectType::UNKNOWN) continue;
            
            frame->detected_objects.push_back(ConvertDetToTrackObj(box, frame->timestamp));
            valid_dsvt_boxes.push_back(box);
        }

        if (tracker_) {
            tracker_->Track(frame); 
        }

        // 5. 构建最终显示列�?
        // A. 首先加入 Tracker 成功输出的目�?
        std::vector<highway::perception::track::common::ObjectPtr> final_display_objects;
        final_display_objects = frame->tracked_objects;

        // B.  检查是否有 DSVT 目标�?Tracker "吞掉" (未输�?
        // 如果 Tracker 没输出，把原�?DSVT 捞回来显�?

        // for (const auto& raw_box : valid_dsvt_boxes) {
        //     bool is_tracked = false;
            
        //     // 检查这个原始框是否与任何已跟踪的物体重�?
        //     for (const auto& tracked_obj : final_display_objects) {
        //         // 简单的 IOU/距离 检�?
        //         double dist = (raw_box.center.head(2) - tracked_obj->bbox.center.head(2)).norm();
        //         // 阈值设宽一点，只要附近有跟踪目标，就认为它被跟踪了
        //         if (dist < 2.0) { 
        //             is_tracked = true;
        //             break;
        //         }
        //     }

        //     // 如果 Tracker 没有输出�?(可能是因�?age 不够，或者关联失�?
        //     // 强制把它加回�?
        //     if (!is_tracked) {
        //         auto fallback_obj = ConvertDetToTrackObj(raw_box, frame->timestamp);
        //         fallback_obj->track_id = -1; // -1 代表这是一个“未稳定跟踪”的纯检测结�?
        //         // 为了区分，可以把速度�?，或者保留原始速度
        //         // fallback_obj->velocity = ... 
        //         final_display_objects.push_back(fallback_obj);
        //     }
        // }

        // 6. 融合 Cluster (只保留那些既没被 Tracker 覆盖)
        std::vector<autoware_msgs::CloudCluster> secondary_clusters;
        
        for (const auto& cluster : cluster_results) {
            lidar_net::base::BoundingBox c_box;
            c_box.center << cluster.centroid_point.point.x, cluster.centroid_point.point.y, cluster.centroid_point.point.z;
            c_box.size << cluster.dimensions.x, cluster.dimensions.y, cluster.dimensions.z;
            
            if (!isValidCluster(c_box)) continue;

            bool is_duplicate = false;
            // 跟最终显示列表比�?
            for (const auto& display_obj : final_display_objects) {
                // �?TrackObject 转为 Box 做比�?
                lidar_net::base::BoundingBox d_box;
                d_box.center = display_obj->bbox.center;
                d_box.size = display_obj->bbox.size;
                
                if (isOverlap(d_box, c_box)) {
                    is_duplicate = true;
                    break;
                }
            }
            
            if (!is_duplicate) {
                secondary_clusters.push_back(cluster);
            }
        }

        // 7. 统一发布
        
        // 使用合并后的 final_display_objects 进行发布
        publishUnifiedObjects(final_display_objects, secondary_clusters, header);
        publishTrackedMarkers(final_display_objects, secondary_clusters, header);
        
    }

    // 聚类流程
    std::vector<autoware_msgs::CloudCluster> runCluster(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_cloud_ptr, 
        const std_msgs::Header& header) 
    {   
	    pcl::PointCloud<pcl::PointXYZI>::Ptr clip_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
	    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr noground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        // 地面分割
        patchWork_.estimate_ground(in_cloud_ptr, ground_cloud_ptr, noground_cloud_ptr);

        pcl::PointCloud<pcl::PointXYZI>::Ptr outCloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pointsVector;
        cluster_.segmentByDistance(noground_cloud_ptr, outCloudPtr, pointsVector);

        autoware_msgs::CloudClusterArray inOutClusters;
        boundingBox_.getBoundingBox(header, pointsVector, inOutClusters);

        if (publish_ground_debug_) {
            noground_cloud_ptr->width = static_cast<uint32_t>(noground_cloud_ptr->points.size());
            noground_cloud_ptr->height = 1;
            noground_cloud_ptr->is_dense = false;
            outCloudPtr->width = static_cast<uint32_t>(outCloudPtr->points.size());
            outCloudPtr->height = 1;
            outCloudPtr->is_dense = false;

            sensor_msgs::PointCloud2 debug_msg;
            sensor_msgs::PointCloud2 debug_msg_2;
            pcl::toROSMsg(*noground_cloud_ptr, debug_msg);
            pcl::toROSMsg(*outCloudPtr, debug_msg_2);
            debug_msg.header = header;
            debug_msg_2.header = header;
            _pub_noground_cloud.publish(debug_msg);
            _pub_cluster_cloud.publish(debug_msg_2);
        }

        return inOutClusters.clusters;
    }

    // 发布函数
    void publishUnifiedObjects(
        const std::vector<highway::perception::track::common::ObjectPtr>& tracked_objects, 
        const std::vector<autoware_msgs::CloudCluster>& untracked_clusters,
        const std_msgs::Header& header) 
    {
         autoware_msgs::DetectedObjectArray detected_objects_msg;
         msg_gen::PerceptionObjects perception_objects_msg; 

         detected_objects_msg.header = header;
         perception_objects_msg.header = header;

         // A. 跟踪物体 (DSVT) - 输出规则�?
         for (const auto& obj_ptr : tracked_objects) {
            autoware_msgs::DetectedObject detected_object;
            msg_gen::Object obj_gen;
            
            uint8_t gen_type = 4; 
            std::string label = "Unknown";
            switch (obj_ptr->type) {
                case highway::perception::track::common::ObjectType::VEHICLE:    gen_type = 0; label="vehicle"; break;
                case highway::perception::track::common::ObjectType::PEDESTRIAN: gen_type = 2; label="pedestrian"; break;
                case highway::perception::track::common::ObjectType::CYCLIST:    gen_type = 3; label="cyclist"; break;
                default: break;
            }

            detected_object.header = header; 
            detected_object.label = label; 
            detected_object.score = 1.0; 
            detected_object.valid = true; 
            detected_object.id = obj_ptr->track_id;
            detected_object.space_frame = header.frame_id;
            detected_object.pose.position.x = obj_ptr->bbox.center.x(); 
            detected_object.pose.position.y = obj_ptr->bbox.center.y(); 
            detected_object.pose.position.z = obj_ptr->bbox.center.z(); 
            detected_object.pose.orientation = tf::createQuaternionMsgFromYaw(obj_ptr->bbox.theta);
            detected_object.dimensions.x = obj_ptr->bbox.size.x(); 
            detected_object.dimensions.y = obj_ptr->bbox.size.y(); 
            detected_object.dimensions.z = obj_ptr->bbox.size.z();
            
            obj_gen.id = obj_ptr->track_id; 
            obj_gen.type = gen_type;
            obj_gen.confidence = 1.0;
            obj_gen.x = obj_ptr->bbox.center.x(); obj_gen.y = obj_ptr->bbox.center.y(); obj_gen.z = obj_ptr->bbox.center.z();
            obj_gen.width = obj_ptr->bbox.size.x(); obj_gen.length = obj_ptr->bbox.size.y(); obj_gen.height = obj_ptr->bbox.size.z();
            // Tracker 算出来的速度
            obj_gen.vxrel = obj_ptr->velocity.x(); obj_gen.vyrel = obj_ptr->velocity.y(); 
            obj_gen.speed = obj_ptr->velocity.head(2).norm();
            obj_gen.heading = obj_ptr->bbox.theta * 180.0 / M_PI; 
            
            // DSVT 结果为规则矩�?(4�?
            obj_gen.corner_points.clear();
            for(int i=0; i<4; ++i) {
                geometry_msgs::Point32 pt32;
                pt32.x = obj_ptr->bbox.corners2d(0, i); pt32.y = obj_ptr->bbox.corners2d(1, i);
                pt32.z = detected_object.pose.position.z - detected_object.dimensions.z/2.0;
                detected_object.convex_hull.polygon.points.push_back(pt32);

                geometry_msgs::Point pt; pt.x = pt32.x; pt.y = pt32.y; pt.z = pt32.z;
                obj_gen.corner_points.push_back(pt);
            }
            detected_objects_msg.objects.push_back(detected_object);
            perception_objects_msg.objs.push_back(obj_gen);
         }

         // B. 聚类物体 (Cluster) - 输出多凸�?
         for (size_t i = 0; i < untracked_clusters.size(); ++i) {
            const auto& cluster = untracked_clusters[i];
            autoware_msgs::DetectedObject detected_object;
            msg_gen::Object obj_gen;

            detected_object.header = header;
            detected_object.label = "unknown"; 
            detected_object.score = cluster.score;
            detected_object.valid = true;
            detected_object.space_frame = header.frame_id;
            detected_object.id = 10000 + i; 

            detected_object.pose = cluster.bounding_box.pose;
            detected_object.dimensions = cluster.bounding_box.dimensions;
            detected_object.convex_hull = cluster.convex_hull; // 保留原始凸包

            detected_objects_msg.objects.push_back(detected_object);

            obj_gen.id = detected_object.id;
            obj_gen.type = 4; 
            obj_gen.confidence = cluster.score;
            obj_gen.x = cluster.centroid_point.point.x; obj_gen.y = cluster.centroid_point.point.y; obj_gen.z = cluster.centroid_point.point.z;
            obj_gen.width = cluster.dimensions.x; obj_gen.length = cluster.dimensions.y; obj_gen.height = cluster.dimensions.z;
            obj_gen.vxrel = 0; obj_gen.vyrel = 0; obj_gen.speed = 0;
            
            tf::Quaternion q; tf::quaternionMsgToTF(cluster.bounding_box.pose.orientation, q);
            obj_gen.heading = tf::getYaw(q) * 180.0 / M_PI;

            // 将所有凸包点填入 corner_points
            obj_gen.corner_points.clear();
            for(const auto& pt32 : cluster.convex_hull.polygon.points) {
                geometry_msgs::Point pt; pt.x = pt32.x; pt.y = pt32.y; pt.z = pt32.z;
                obj_gen.corner_points.push_back(pt);
            }
            perception_objects_msg.objs.push_back(obj_gen);
         }
         
         _pub_detected_objects.publish(detected_objects_msg);
         _pub_objects.publish(perception_objects_msg);
    }

    void publishTrackedMarkers(
        const std::vector<highway::perception::track::common::ObjectPtr>& tracked_objects, 
        const std::vector<autoware_msgs::CloudCluster>& untracked_clusters,
        const std_msgs::Header& header) 
    {
        visualization_msgs::MarkerArray markers;
        
        // 1. 清除�?Marker
        std::vector<std::string> namespaces = {"objects", "ids", "cluster_polygons", "cluster_texts"};
        for(const auto& ns : namespaces) {
            visualization_msgs::Marker clear_marker;
            clear_marker.header = header; 
            clear_marker.ns = ns; 
            clear_marker.action = visualization_msgs::Marker::DELETEALL;
            markers.markers.push_back(clear_marker);
        }

        // 2. Tracked Objects (DSVT) -> 彩色实心�?+ ID/速度信息
        for (const auto& obj : tracked_objects) {
            float r=0, g=1, b=0; 
            std::string label_str = "Unk";
            
            // 根据类别定色
            if(obj->type==highway::perception::track::common::ObjectType::VEHICLE) {
                r=0; g=1; b=0; label_str="Veh"; // 绿色
            } else if(obj->type==highway::perception::track::common::ObjectType::PEDESTRIAN) {
                r=1; g=1; b=0; label_str="Ped"; // 黄色
            } else if(obj->type==highway::perception::track::common::ObjectType::CYCLIST) {
                r=0; g=1; b=1; label_str="Cyc"; // 青色
            }

            // 延长生命周期�?0.2秒，防止因系统延迟导致文字闪�?消失
            // 因为开头有 DELETEALL，所以延长时间是安全�?
            ros::Duration marker_lifetime(0.2);

            // A. 方框 Marker
            visualization_msgs::Marker marker;
            marker.header = header; 
            marker.ns = "objects"; 
            // 使用 track_id 作为 Marker ID，确保稳定对�?
            marker.id = obj->track_id; 
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD; 
            marker.lifetime = marker_lifetime;
            
            marker.pose.position.x = obj->bbox.center.x(); 
            marker.pose.position.y = obj->bbox.center.y(); 
            marker.pose.position.z = obj->bbox.center.z();
            marker.pose.orientation = tf::createQuaternionMsgFromYaw(obj->bbox.theta);
            
            marker.scale.x = obj->bbox.size.x(); 
            marker.scale.y = obj->bbox.size.y(); 
            marker.scale.z = obj->bbox.size.z();
            
            marker.color.a = 0.6; // 半透明
            marker.color.r = r; marker.color.g = g; marker.color.b = b;
            markers.markers.push_back(marker);

            // B. 文字 Marker (Text)
            visualization_msgs::Marker text;
            text.header = header; 
            text.ns = "ids"; 
            text.id = obj->track_id;
            text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::Marker::ADD; 
            text.lifetime = marker_lifetime; // 关键：生命周期同�?
            
            text.scale.z = 0.8; // 字体改大一点，确保显眼
            text.color.a = 1.0; 
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; // 纯白�?
            
            text.pose.position = marker.pose.position; 
            // 抬高文字高度，确保不被方框遮�?
            text.pose.position.z += std::max(1.2f, (float)obj->bbox.size.z()/2.0f + 0.8f);
            
            float vel = obj->velocity.head(2).norm();
    // 检查是否是兜底目标 (ID = -1)
    if (obj->track_id == -1) {
        // 让兜底目标的文字显示�?"Det" 而不�?ID
        std::stringstream ss;
        ss << "[Det] " << label_str; // 只显示检测，没ID
        text.text = ss.str();
        
        // 给兜底目标换个透明度，表示这是不稳定的
        marker.color.a = 0.3; 
    } else {
        // 正常�?Tracker 目标
        std::stringstream ss;
        ss << "ID:" << obj->track_id << " (" << label_str << ")\n" 
           << std::fixed << std::setprecision(1) << vel << "m/s";
        text.text = ss.str();
    }
            
            markers.markers.push_back(text);
        }

        // 3. Untracked Clusters -> 白色线框柱体 + "Unknown" 文字
        int cluster_id_counter = 0;

        for (const auto& cluster : untracked_clusters) {
            ros::Duration cluster_lifetime(0.2); 

            // A. 线框柱体
            visualization_msgs::Marker line_list;
            line_list.header = header; 
            line_list.ns = "cluster_polygons"; 
            line_list.id = cluster_id_counter++;
            line_list.type = visualization_msgs::Marker::LINE_LIST; 
            line_list.action = visualization_msgs::Marker::ADD; 
            line_list.lifetime = cluster_lifetime;
            line_list.scale.x = 0.05; 
            
            line_list.color.a = 1.0; 
            line_list.color.r = 1.0; line_list.color.g = 1.0; line_list.color.b = 1.0; // 白色

            // 获取 Cluster 高度
            double min_z = cluster.min_point.point.z;
            double max_z = cluster.max_point.point.z;
            if (max_z - min_z < 0.1) max_z = min_z + 1.0;

            const auto& points = cluster.convex_hull.polygon.points;
            size_t num_points = points.size();

            if (num_points > 1) {
                for (size_t i = 0; i < num_points; ++i) {
                    geometry_msgs::Point p1, p2;
                    p1.x = points[i].x; p1.y = points[i].y; 
                    p2.x = points[(i + 1) % num_points].x; p2.y = points[(i + 1) % num_points].y;

                    // 底面
                    p1.z = min_z; p2.z = min_z;
                    line_list.points.push_back(p1); line_list.points.push_back(p2);
                    // 顶面
                    p1.z = max_z; p2.z = max_z;
                    line_list.points.push_back(p1); line_list.points.push_back(p2);
                    // 侧棱
                    geometry_msgs::Point p_bottom, p_top;
                    p_bottom.x = points[i].x; p_bottom.y = points[i].y; p_bottom.z = min_z;
                    p_top.x = points[i].x;    p_top.y = points[i].y;    p_top.z = max_z;
                    line_list.points.push_back(p_bottom); line_list.points.push_back(p_top);
                }
            }
            markers.markers.push_back(line_list);

            // B. Cluster 文字标签
            visualization_msgs::Marker text;
            text.header = header; 
            text.ns = "cluster_texts";   
            text.id = cluster_id_counter; 
            text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::Marker::ADD; 
            text.lifetime = cluster_lifetime;
            
            text.scale.z = 0.5; 
            text.color.a = 0.8; 
            text.color.r = 0.9; text.color.g = 0.9; text.color.b = 0.9;

            text.pose.position.x = cluster.centroid_point.point.x;
            text.pose.position.y = cluster.centroid_point.point.y;
            text.pose.position.z = max_z + 0.5;

            text.text = "Unknown";
            markers.markers.push_back(text);
        }

        pub_markers_.publish(markers);
    }


    bool isValidCluster(const lidar_net::base::BoundingBox& obj) {
        // if (obj.size.x() < 0.15 || obj.size.y() < 0.15) return false; 
        // if (obj.size.x() > 20.0 || obj.size.y() > 20.0) return false; 
        // if (obj.center.z() > 5.0) return false; 
        // if (obj.center.z() + obj.size.z() / 2.0 < 0.0) return false; 
        return true;
    }

    bool isOverlap(const lidar_net::base::BoundingBox& dl_box, const lidar_net::base::BoundingBox& cluster_box) {
        double dist = (dl_box.center.head(2) - cluster_box.center.head(2)).norm();
        double max_r_dl = std::max(dl_box.size.x(), dl_box.size.y()) / 2.0;
        double max_r_cl = std::max(cluster_box.size.x(), cluster_box.size.y()) / 2.0;
        return dist < (max_r_dl + max_r_cl) * 0.6;
    }
    // 准备dsvt输入
    Eigen::MatrixXf prepareDSVTInput(const sensor_msgs::PointCloud2& msg) {
        size_t num_points = msg.height * msg.width;
        if (num_points == 0) return Eigen::MatrixXf(0, 0);
        if (points_buffer_.cols() < num_points) points_buffer_.resize(CHANNEL, num_points + 10000);
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x"), iter_y(msg, "y"), iter_z(msg, "z");
        std::string i_name = "intensity";
        for (const auto& f : msg.fields) if(f.name=="i") i_name="i";
        sensor_msgs::PointCloud2ConstIterator<float> iter_i(msg, i_name);
        int valid_count = 0;
        for(size_t i=0; i<num_points; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
             float x = *iter_x; float y = *iter_y; float z = *iter_z;
             if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
             points_buffer_(0, valid_count) = x; points_buffer_(1, valid_count) = y; points_buffer_(2, valid_count) = z; 
             float raw_intensity = (iter_i != iter_i.end()) ? *iter_i : 0.0f; // 强度归一�?
             points_buffer_(3, valid_count) = std::tanh(raw_intensity / MAX_INTENSITY);
             if (CHANNEL >= 5) points_buffer_(4, valid_count) = 0.0f; 
             valid_count++;
        }
        return points_buffer_.leftCols(valid_count);
    }

    std::vector<lidar_net::base::BoundingBox> runDSVT(const Eigen::MatrixXf& input_points) {
        std::vector<lidar_net::base::BoundingBox> results;
        if (input_points.cols() > 0 && detector_) {
            lidar_net::ProcessStatus status(lidar_net::StatusCode::Succeed);
            detector_->ProcessingEigenRow(input_points, results, &status);
        }
        return results;
    }
    
    // 辅助转换函数
    highway::perception::track::common::ObjectPtr 
    ConvertDetToTrackObj(const lidar_net::base::BoundingBox& box, double timestamp) {
        auto obj = std::make_shared<highway::perception::track::common::Object>();
        obj->timestamp = timestamp;
        using TrackType = highway::perception::track::common::ObjectType;
        using DetType = lidar_net::base::ObjectType;
        if (box.type == DetType::VEHICLE || box.type == DetType::TRUCK || box.type == DetType::BUS) obj->type = TrackType::VEHICLE;    
        else if (box.type == DetType::PEDESTRIAN) obj->type = TrackType::PEDESTRIAN; 
        else if (box.type == DetType::CYCLIST) obj->type = TrackType::CYCLIST;    
        else obj->type = TrackType::UNKNOWN;    

        obj->bbox.center = box.center;
        obj->bbox.size = box.size;
        obj->bbox.theta = box.theta;
        obj->bbox.corners2d = box.corners2d;
        obj->velocity = Eigen::Vector3f::Zero(); 

        float min_dist_sq = std::numeric_limits<float>::max();
        int min_idx = 0;
        for (int i = 0; i < 4; ++i) {
            float dist_sq = box.corners2d.col(i).squaredNorm();
            if (dist_sq < min_dist_sq) { min_dist_sq = dist_sq; min_idx = i; }
        }
        obj->lshape_box.reference_point = Eigen::Vector3d(box.corners2d(0, min_idx), box.corners2d(1, min_idx), 0);
        obj->lshape_box.l_shape = Eigen::Vector3d(box.size.x(), box.size.y(), box.theta);
        obj->points->resize(10); 
        return obj;
    }
    
    void processConfigFile(std::string& config_file) {
        std::ifstream file(config_file);
        if (!file.is_open()) return;
        std::stringstream buffer; buffer << file.rdbuf(); 
        std::string content = buffer.str(); file.close();
        size_t pos = 0;
        while ((pos = content.find("$(find ", pos)) != std::string::npos) {
            size_t end = content.find(")", pos);
            if (end != std::string::npos) {
                std::string full_path = content.substr(pos, end - pos + 1);
                std::string resolved_path = resolvePackagePath(full_path);
                content.replace(pos, end - pos + 1, resolved_path);
                pos += resolved_path.length();
            } else { break; }
        }
        std::ofstream out_file(config_file); if (out_file.is_open()) out_file << content;
    }

    std::string resolvePackagePath(const std::string& path) {
        if (path.find("$(find") != std::string::npos) {
            size_t start = path.find("$(find ") + 7; 
            size_t end = path.find(")", start);
            return ros::package::getPath(path.substr(start, end - start)) + path.substr(end + 1);
        }
        return path;
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    std::unique_ptr<lidar_net::LidarNetDetector> detector_;
    std::shared_ptr<highway::perception::track::BaseTargetTrack> tracker_;
    std::unique_ptr<lidar_fusion::MultiLidarFusion> fusion_ptr_; 
    
    EuclideanCluster cluster_;
    BoundingBox boundingBox_;
    PatchWork patchWork_;

    ros::Timer timer_;
    ros::Publisher pub_markers_;
    ros::Publisher _pub_fusion_cloud;
    ros::Publisher _pub_objects;
    ros::Publisher _pub_detected_objects;
    ros::Publisher _pub_noground_cloud;
    ros::Publisher _pub_cluster_cloud;

    Eigen::MatrixXf points_buffer_;
    int64_t total_time_ = 0;
    int counter_ = 0;
    bool publish_ground_debug_ = true;
};

int main(int argc, char ** argv)
{
    ros::init(argc, argv, "lidar_detection_track");
    LidarDetectorNode node;
    ros::spin();
    return 0;
}
