#include "patchwork.h"
#include <algorithm>
#include <cmath>
#include <limits>



PatchWork::PatchWork(ros::NodeHandle& nh){  

    nh.param("sensor_height", sensor_height_, 0.4);// 【关键】传感器（坐标系原点）离地面的高度。
    nh.param("verbose", verbose_, false);// 是否在终端打印详细的调试信息
    nh.param("num_iter", num_iter_, 3);// RANSAC 迭代次数：在每个 Patch (扇区) 中，迭代多少次来拟合地平面

    // 算法会取 Z 轴最低的 num_lpr_ 个点来计算一个平均高度，作为初始地面高度的粗略估计。
    nh.param("num_lpr", num_lpr_, 20);

    // 最小点数：一个 Patch (扇区) 中最少包含多少点才进行处理，否则该扇区将被忽略。
    nh.param("num_min_pts", num_min_pts_, 10);

    // 种子点阈值：在 LPR 估计的平均地面高度之上多高范围内的点，被认为是初始的地面“种子点”。
    nh.param("th_seeds", th_seeds_, 0.6);

    // 距离阈值：在 RANSAC 迭代中，一个点距离拟合平面的最大距离。
    // 小于此距离的点被认为是“内点”（地面点）。
    nh.param("th_dist", th_dist_, 0.125);


    nh.param("max_range", max_range_, 60.0);// 最大处理范围
    nh.param("min_range", min_range_, 1.0);// 最小处理范围

    // 【关键】直立性阈值：拟合出的地平面的法向量 Z 轴分量的最小值。
    nh.param("uprightness_thr", uprightness_thr_, 0.7);

    // 自适应种子选择边际：一个负数，用于在近距离区域更鲁棒地选择种子点。
    // 通常是 sensor_height_ 的一个负比例。
    nh.param("adaptive_seed_selection_margin", adaptive_seed_selection_margin_, -1.3);
    nh.param("use_ground_temporal_stabilizer", use_ground_temporal_stabilizer_, true);
    nh.param("ground_plane_ema_alpha", ground_plane_ema_alpha_, 0.25);
    nh.param("ground_cache_max_miss", ground_cache_max_miss_, 10);
    nh.param("ground_enter_dist", ground_enter_dist_, 0.15);
    nh.param("nonground_exit_dist", nonground_exit_dist_, 0.25);
    nh.param("small_patch_fallback_z", small_patch_fallback_z_, 0.25);
    ground_plane_ema_alpha_ = std::clamp(ground_plane_ema_alpha_, 0.0, 1.0);
    if (nonground_exit_dist_ < ground_enter_dist_) {
        ROS_WARN("nonground_exit_dist %.3f is smaller than ground_enter_dist %.3f. Swapping values.",
                 nonground_exit_dist_, ground_enter_dist_);
        std::swap(nonground_exit_dist_, ground_enter_dist_);
    }

    // 是否使用全局高程阈值
    nh.param("using_global_thr", using_global_thr_, false);
    // 全局高程阈值：如果 using_global_thr_ 为 true，在远距离区域，
    // 拟合出的地面平均高度不能超过这个值。
    nh.param("global_elevation_thr", global_elevation_thr_, -0.3);


    nh.param("num_zone", num_zones_, 4);// 将点云划分为多少个同心圆区域 (Zone)
    nh.param("num_zones", num_zones_, num_zones_);
    // 每个 Zone 内部划分的扇区 (Sector) 数量。
    nh.param("num_sectors_each_zone", num_sectors_each_zone_, {16, 32, 48, 56});

    // 每个 Zone 内部划分的环 (Ring) 数量
    nh.param("num_rings_each_zone", num_rings_each_zone_, {4, 6, 6, 8});
    
    // 每个 Zone 的最大半径（外边界）。
    // 【重要】最后一个值必须与 max_range_ 匹配！
    nh.param("min_ranges", min_ranges_, {12.0, 24.0, 40.0, 60.0});
    
    // 【关键】高程阈值 (Elevation Thresholds):
    nh.param("elevation_thr", elevation_thr_, {-0.2, -0.2, -0.2, -0.2});

    // 【关键】平坦度阈值 (Flatness Thresholds):
    // 这是一个数组。只有当上述 elevation_thr_ 检查被触发时，才会启用这个更严格的平坦度检查。
    nh.param("flatness_thr", flatness_thr_, {0.000125, 0.000125, 0.000185, 0.000185});
    
    
    // 动态体素滤波：每个 Zone 使用的体素大小（Leaf Size）。
    nh.param("leaf_sizes", leaf_sizes_, {0.2, 0.3, 0.5, 0.8});

    if (using_global_thr_) {
            ROS_WARN("Global elevation threshold is turned on: %f", global_elevation_thr_);
        }

    num_rings_of_interest_ = std::min(elevation_thr_.size(), flatness_thr_.size());
    if (elevation_thr_.size() != flatness_thr_.size()) {
        ROS_WARN("elevation_thr size (%zu) != flatness_thr size (%zu). Using %d near rings.",
                 elevation_thr_.size(), flatness_thr_.size(), num_rings_of_interest_);
    }

    revert_pc.reserve(3000);
    ground_pc_.reserve(3000);
    non_ground_pc_.reserve(3000);
    regionwise_ground_.reserve(3000);
    regionwise_nonground_.reserve(3000);

    min_range_z1_ = min_ranges_[0];
    min_range_z2_ = min_ranges_[1];
    min_range_z3_ = min_ranges_[2];
    min_range_z4_ = min_ranges_[3];

    min_range_sq_ = min_range_ * min_range_;
    max_range_sq_ = max_range_ * max_range_;

    min_range_z1_sq_ = min_range_z1_ * min_range_z1_;
    min_range_z2_sq_ = min_range_z2_ * min_range_z2_;
    min_range_z3_sq_ = min_range_z3_ * min_range_z3_;
    min_range_z4_sq_ = min_range_z4_ * min_range_z4_;

    ring_sizes_   = {(min_range_z1_ - min_range_) / num_rings_each_zone_.at(0),
                        (min_range_z2_ - min_range_z1_) / num_rings_each_zone_.at(1),
                        (min_range_z3_ - min_range_z2_) / num_rings_each_zone_.at(2),
                        (min_range_z4_ - min_range_z3_) / num_rings_each_zone_.at(3)};
    sector_sizes_ = {2 * M_PI / num_sectors_each_zone_.at(0), 2 * M_PI / num_sectors_each_zone_.at(1),
                        2 * M_PI / num_sectors_each_zone_.at(2),
                        2 * M_PI / num_sectors_each_zone_.at(3)};
    //std::cout << "INITIALIZATION COMPLETE" << std::endl;

    for (int iter = 0; iter < num_zones_; ++iter) {
        Zone z;
        initialize_zone(z, num_sectors_each_zone_.at(iter), num_rings_each_zone_.at(iter));
        ConcentricZoneModel_.push_back(z);
    }
    plane_cache_.resize(num_zones_);
    for (int zone_idx = 0; zone_idx < num_zones_; ++zone_idx) {
        plane_cache_[zone_idx].resize(num_rings_each_zone_.at(zone_idx));
        for (int ring_idx = 0; ring_idx < num_rings_each_zone_.at(zone_idx); ++ring_idx) {
            plane_cache_[zone_idx][ring_idx].resize(num_sectors_each_zone_.at(zone_idx));
        }
    }
}

PatchWork::~PatchWork(){}    



void PatchWork::clip_cloud_roi(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud,
                                  pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud,
                                  const float clip_x_min, const float clip_x_max,
                                  const float clip_y_min, const float clip_y_max,
                                  const float clip_z_min, const float clip_z_max, const double in_dist)

{
  out_cloud->points.clear();
  for (unsigned int i = 0; i < in_cloud->points.size(); i++)
  {
    if (in_cloud->points[i].x >= clip_x_min && in_cloud->points[i].x <= clip_x_max &&
        in_cloud->points[i].y >= clip_y_min && in_cloud->points[i].y <= clip_y_max &&
        in_cloud->points[i].z >= clip_z_min && in_cloud->points[i].z <= clip_z_max &&
        sqrt(pow(in_cloud->points[i].x, 2) + pow(in_cloud->points[i].y, 2)) > in_dist)
    {
      out_cloud->points.push_back(in_cloud->points[i]);
    }
  }
}
void PatchWork::groundfilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr &in_cloud,
                                  pcl::PointCloud<pcl::PointXYZI>::Ptr &out_ground_points,
                                  pcl::PointCloud<pcl::PointXYZI>::Ptr &out_groundless_points,
                                  float threshold, float floor_max_angle)

{ 

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr idx(new pcl::PointIndices);

  pcl::SACSegmentation<pcl::PointXYZI> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(threshold);
  seg.setMaxIterations(100);
  seg.setAxis(Eigen::Vector3f(0, 0, 1));
  seg.setEpsAngle(floor_max_angle);

  seg.setInputCloud(in_cloud);
  seg.segment(*idx, *coefficients);

  if (idx->indices.size() == 0)
  {
    //std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud(in_cloud);
  extract.setIndices(idx);
  extract.setNegative(false);  // true removes the indices, false leaves only the indices
  extract.filter(*out_ground_points);

  extract.setNegative(true);
  extract.filter(*out_groundless_points);  

}

void PatchWork::ransac_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr &in_cloud,
                                  pcl::PointCloud<pcl::PointXYZI>::Ptr &out_ground,
                                  pcl::PointCloud<pcl::PointXYZI>::Ptr &out_unground,
                                  float threshold)
{
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr idx(new pcl::PointIndices);

  pcl::SACSegmentation<pcl::PointXYZI> seg;
  seg.setOptimizeCoefficients(true);
  // seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(threshold);
  // seg.setMaxIterations(100);
  // seg.setAxis(Eigen::Vector3f(0, 0, 1));
  // seg.setEpsAngle(0.3);
  seg.setInputCloud(in_cloud);
  seg.segment(*idx, *coefficients);

  if (idx->indices.size() == 0)
  {
    //std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }
  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud(in_cloud);
  extract.setIndices(idx);
  extract.setNegative(false);  // true removes the indices, false leaves only the indices
  extract.filter(*out_ground);
  extract.setNegative(true);
  extract.filter(*out_unground);
}


void PatchWork::extract_initial_seeds_(
        const int zone_idx, const pcl::PointCloud<pcl::PointXYZI> &p_sorted,
        pcl::PointCloud<pcl::PointXYZI> &init_seeds) {
    init_seeds.points.clear();

    // LPR is the mean of low point representative
    double sum = 0;
    int    cnt = 0;

    int init_idx = 0;
    // Empirically, adaptive seed selection applying to Z1 is fine
    static double lowest_h_margin_in_close_zone = (sensor_height_ == 0.0)? -0.1 : adaptive_seed_selection_margin_ * sensor_height_;
    if (zone_idx == 0) {
        for (int i = 0; i < p_sorted.points.size(); i++) {
            if (p_sorted.points[i].z < lowest_h_margin_in_close_zone) {
                ++init_idx;
            } else {
                break;
            }
        }
    }

    // Calculate the mean height value.
    for (int i          = init_idx; i < p_sorted.points.size() && cnt < num_lpr_; i++) {
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double   lpr_height = cnt != 0 ? sum / cnt : 0;// in case divide by 0

    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < p_sorted.points.size(); i++) {
        if (p_sorted.points[i].z < lpr_height + th_seeds_) {
            init_seeds.points.push_back(p_sorted.points[i]);
        }
    }
}

void PatchWork::estimate_plane_(const pcl::PointCloud<pcl::PointXYZI> &ground) {
    pcl::computeMeanAndCovarianceMatrix(ground, cov_, pc_mean_);
    // Singular Value Decomposition: SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);
    singular_values_ = svd.singularValues();

    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean_.head<3>();

    // according to normal.T*[x,y,z] = -d
    d_ = -(normal_.transpose() * seeds_mean)(0, 0);
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;
}
void PatchWork::initialize_zone(Zone &z, int num_sectors, int num_rings)
{
    z.clear();
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.reserve(1000);
    Ring     ring;
    for (int i = 0; i < num_sectors; i++) {
        ring.emplace_back(cloud);
    }
    for (int j = 0; j < num_rings; j++) {
        z.emplace_back(ring);
    }

}

void PatchWork::flush_patches_in_zone(Zone &patches, int num_sectors, int num_rings)
{
  for (int i = 0; i < num_sectors; i++)
  {
    for (size_t j = 0; j < num_rings; j++)
    {
      if(!patches[j][i].points.empty()) patches[j][i].points.clear();
    }
    
  }
  

}
double PatchWork::xy2theta(const double &x, const double &y) { // 0 ~ 2 * PI
    if (y >= 0) {
        return atan2(y, x); // 1, 2 quadrant
    } else {
        return 2 * M_PI + atan2(y, x);// 3, 4 quadrant
    }
}


template<typename PointT> bool point_z_cmp(PointT a, PointT b)
{
  return a.z < b.z;
}
void PatchWork::pc2czm(const pcl::PointCloud<pcl::PointXYZI> &src, std::vector<Zone> &czm) {

    // 1. 创建临时的点云容器，用于按距离分区 (Zone-level)
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_zones[num_zones_];
    for (int i=0; i<num_zones_; ++i) {
        cloud_zones[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
    }
    // 2. 先只按距离分区 (Zone)，不区分扇区 (Sector)
    for (auto const &pt : src.points) {
        // 【距离计算】
        double r_squared = pow(pt.x , 2) + pow(pt.y, 2);

        if ((r_squared <= max_range_sq_) && (r_squared > min_range_sq_)) {
            // 【按距离分区】
            if (r_squared < min_range_z1_sq_) { // Zone 0
                cloud_zones[0]->points.emplace_back(pt);
            } else if (r_squared < min_range_z2_sq_) { // Zone 1
                cloud_zones[1]->points.emplace_back(pt);
            } else if (r_squared < min_range_z3_sq_) { // Zone 2
                cloud_zones[2]->points.emplace_back(pt);
            }
              else { // Zone 3
                cloud_zones[3]->points.emplace_back(pt);
            }
        }
    }
    // 3. 对每个距离分区 (Zone) 的点云进行动态体素滤波
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    pcl::PointCloud<pcl::PointXYZI> filtered_cloud; // 用于存放合并后的点云
    for (int i=0; i<num_zones_; ++i) {
        if (cloud_zones[i]->points.empty()) continue;
        
        voxel_filter.setInputCloud(cloud_zones[i]);
        // 使用在构造函数中设置的 leaf_sizes_
        voxel_filter.setLeafSize(leaf_sizes_[i], leaf_sizes_[i], leaf_sizes_[i]);
        voxel_filter.filter(*(cloud_zones[i])); // 滤波结果写回原处 (cloud_zones[i])

    }
    // 4. 将【已滤波】的点云分配到最终的 Ring 和 Sector

    for (int i=0; i<num_zones_; ++i) {
        if (cloud_zones[i]->points.empty()) continue;

        for (auto const &pt : cloud_zones[i]->points) {
            int ring_idx, sector_idx;
            
            // 【再次计算 r 和 theta】
            // (这是在滤波后的、更少的点上计算，开销很小)
            double r = sqrt(pow(pt.x, 2) + pow(pt.y, 2));
            double theta = xy2theta(pt.x, pt.y);

            // 根据当前 Zone (i) 来决定如何计算 ring_idx 和 sector_idx
            // (这部分逻辑是从原 pc2czm 复制和修改而来的)
            switch(i) {
                case 0: // Zone 0 (r < min_range_z2_)
                    ring_idx   = std::min(static_cast<int>(((r - min_range_) / ring_sizes_[0])), num_rings_each_zone_[0] - 1);
                    sector_idx = std::min(static_cast<int>((theta / sector_sizes_[0])), num_sectors_each_zone_[0] - 1);
                    break;
                case 1: // Zone 1 (r < min_range_z3_)
                    ring_idx   = std::min(static_cast<int>(((r - min_range_z1_) / ring_sizes_[1])), num_rings_each_zone_[1] - 1);
                    sector_idx = std::min(static_cast<int>((theta / sector_sizes_[1])), num_sectors_each_zone_[1] - 1);
                    break;
                case 2: // Zone 2 (r < min_range_z4_)
                    ring_idx   = std::min(static_cast<int>(((r - min_range_z2_) / ring_sizes_[2])), num_rings_each_zone_[2] - 1);
                    sector_idx = std::min(static_cast<int>((theta / sector_sizes_[2])), num_sectors_each_zone_[2] - 1);
                    break;
                case 3: // Zone 3 (else)
                    ring_idx   = std::min(static_cast<int>(((r - min_range_z3_) / ring_sizes_[3])), num_rings_each_zone_[3] - 1);
                    sector_idx = std::min(static_cast<int>((theta / sector_sizes_[3])), num_sectors_each_zone_[3] - 1);
                    break;
            }
            
            // 确保索引有效
            if (ring_idx < 0 || sector_idx < 0) continue; 
            
            // 分配到最终的 bin
            czm[i][ring_idx][sector_idx].points.emplace_back(pt);
        }
    }
}

void PatchWork::extract_piecewiseground(
        const int zone_idx, const pcl::PointCloud<pcl::PointXYZI> &src,
        pcl::PointCloud<pcl::PointXYZI> &dst,
        pcl::PointCloud<pcl::PointXYZI> &non_ground_dst) {
    PatchGroundResult result;
    if (fit_patch_ground(zone_idx, src, result)) {
        dst = result.ground;
        non_ground_dst = result.nonground;
    } else {
        dst.clear();
        non_ground_dst = src;
    }
}

bool PatchWork::fit_patch_ground(const int zone_idx,
                                 const pcl::PointCloud<pcl::PointXYZI> &src,
                                 PatchGroundResult &result) {
    result = PatchGroundResult();
    if (src.points.size() < 3) {
        return false;
    }

    pcl::PointCloud<pcl::PointXYZI> sorted_src = src;
    std::sort(sorted_src.points.begin(), sorted_src.points.end(), point_z_cmp<pcl::PointXYZI>);

    // 0. Initialization
    if (!ground_pc_.empty()) ground_pc_.clear();
    // 1. set seeds!

    extract_initial_seeds_(zone_idx, sorted_src, ground_pc_);
    if (ground_pc_.points.size() < 3) {
        return false;
    }
    // 2. Extract ground
    for (int i = 0; i < num_iter_; i++) {
        estimate_plane_(ground_pc_);
        ground_pc_.clear();

        //pointcloud to matrix
        //Eigen::MatrixXf points(src.points.size(), 3);
        //int j = 0;
        //for (auto &p:src.points) {
        //    points.row(j++) << p.x, p.y, p.z;
        //}
        // ground plane model

        //Eigen::VectorXf result = points * normal_;
        // threshold filter
        for (int r = 0;r<src.points.size(); r++){
            const auto& p = src.points[r];
            // 手动计算点到平面的投影（等效于 result[r]）
            double distance = p.x * normal_(0) + p.y * normal_(1) + p.z * normal_(2);

            if (i < num_iter_ - 1) {
                if (distance < th_dist_d_) { // th_dist_d_ 在 estimate_plane_ 中已计算
                    ground_pc_.points.push_back(p);
                    }
            } 
            else { // Final stage
                if (distance < th_dist_d_) {
                    result.ground.points.push_back(p);
                    } 
                else {
                    result.nonground.points.push_back(p);
                    }
            }   
        }
        if (i < num_iter_ - 1 && ground_pc_.points.size() < 3) {
            return false;
        }
    }   

    result.normal = normal_;
    result.d = d_;
    result.mean = pc_mean_;
    result.singular_values = singular_values_.head<3>();
    result.valid = true;
    return true;
}

bool PatchWork::is_reliable_plane(const PatchGroundResult& result) const {
    if (!result.valid) return false;
    if (!result.normal.allFinite() || !result.mean.allFinite() || !result.singular_values.allFinite()) return false;
    if (!std::isfinite(result.d)) return false;
    if (result.ground.points.size() < 3) return false;
    if (std::abs(result.normal.z()) < uprightness_thr_) return false;
    const float denom = result.singular_values.x() + result.singular_values.y() + result.singular_values.z();
    return std::isfinite(denom) && denom > 1.0e-6f;
}

PatchWork::GroundPlaneState& PatchWork::get_plane_state(int zone_idx, int ring_idx, int sector_idx) {
    return plane_cache_[zone_idx][ring_idx][sector_idx];
}

const PatchWork::GroundPlaneState* PatchWork::get_plane_state(int zone_idx, int ring_idx, int sector_idx) const {
    if (zone_idx < 0 || zone_idx >= static_cast<int>(plane_cache_.size())) return nullptr;
    if (ring_idx < 0 || ring_idx >= static_cast<int>(plane_cache_[zone_idx].size())) return nullptr;
    if (sector_idx < 0 || sector_idx >= static_cast<int>(plane_cache_[zone_idx][ring_idx].size())) return nullptr;
    const auto& state = plane_cache_[zone_idx][ring_idx][sector_idx];
    return state.valid ? &state : nullptr;
}

void PatchWork::update_plane_state(GroundPlaneState& state, const PatchGroundResult& result) {
    Eigen::Vector3f normal = result.normal;
    float d = result.d;
    if (normal.z() < 0.0f) {
        normal = -normal;
        d = -d;
    }

    if (!use_ground_temporal_stabilizer_ || !state.valid) {
        state.normal = normal.normalized();
        state.d = d;
        state.mean = result.mean;
        state.singular_values = result.singular_values;
    } else {
        Eigen::Vector3f blended_normal =
            ((1.0 - ground_plane_ema_alpha_) * state.normal + ground_plane_ema_alpha_ * normal).normalized();
        float blended_d = static_cast<float>((1.0 - ground_plane_ema_alpha_) * state.d +
                                            ground_plane_ema_alpha_ * d);
        state.normal = blended_normal;
        state.d = blended_d;
        state.mean = ((1.0 - ground_plane_ema_alpha_) * state.mean + ground_plane_ema_alpha_ * result.mean).eval();
        state.singular_values =
            ((1.0 - ground_plane_ema_alpha_) * state.singular_values + ground_plane_ema_alpha_ * result.singular_values).eval();
    }

    state.valid = true;
    state.missed_frames = 0;
}

void PatchWork::classify_with_plane(const pcl::PointCloud<pcl::PointXYZI>& src,
                                    const Eigen::Vector3f& normal,
                                    float d,
                                    pcl::PointCloud<pcl::PointXYZI>& ground,
                                    pcl::PointCloud<pcl::PointXYZI>& nonground) const {
    for (const auto& p : src.points) {
        const double signed_distance = p.x * normal.x() + p.y * normal.y() + p.z * normal.z() + d;
        if (signed_distance <= ground_enter_dist_) {
            ground.points.push_back(p);
        } else if (signed_distance >= nonground_exit_dist_) {
            nonground.points.push_back(p);
        } else {
            ground.points.push_back(p);
        }
    }
}

void PatchWork::classify_small_patch(const pcl::PointCloud<pcl::PointXYZI>& src,
                                     const GroundPlaneState* state,
                                     pcl::PointCloud<pcl::PointXYZI>& ground,
                                     pcl::PointCloud<pcl::PointXYZI>& nonground) const {
    if (state && state->valid) {
        classify_with_plane(src, state->normal, state->d, ground, nonground);
        return;
    }

    if (src.points.empty()) return;
    double min_z = std::numeric_limits<double>::max();
    for (const auto& p : src.points) {
        min_z = std::min(min_z, static_cast<double>(p.z));
    }
    const double z_threshold = min_z + small_patch_fallback_z_;
    for (const auto& p : src.points) {
        if (p.z <= z_threshold) {
            ground.points.push_back(p);
        } else {
            nonground.points.push_back(p);
        }
    }
}

void PatchWork::estimate_ground(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_in,
        pcl::PointCloud<pcl::PointXYZI>::Ptr &ground_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr &non_ground_cloud)
        // double &time_taken) 
{
    // 1.Msg to pointcloud
    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
    laserCloudIn = *cloud_in;

    // t1 = ros::Time::now().toSec();
    // 4. pointcloud -> regionwise setting
    for (int k = 0; k < num_zones_; ++k) {
        flush_patches_in_zone(ConcentricZoneModel_[k], num_sectors_each_zone_[k], num_rings_each_zone_[k]);
    }
    pc2czm(laserCloudIn, ConcentricZoneModel_);
    pcl::PointCloud<pcl::PointXYZI> cloud_nonground;
    pcl::PointCloud<pcl::PointXYZI> cloud_out;

    cloud_out.clear();
    cloud_nonground.clear();
    revert_pc.clear();
    reject_pc.clear();

    ++frame_index_;
    int small_patch_count = 0;
    int fallback_patch_count = 0;
    int cache_hit_count = 0;

    int concentric_idx = 0;
    for (int k = 0; k < num_zones_; ++k) {
        const auto& zone = ConcentricZoneModel_[k];
        for (uint16_t ring_idx = 0; ring_idx < num_rings_each_zone_[k]; ++ring_idx) {
            for (uint16_t sector_idx = 0; sector_idx < num_sectors_each_zone_[k]; ++sector_idx) {
                GroundPlaneState& plane_state = get_plane_state(k, ring_idx, sector_idx);
                if (plane_state.valid) {
                    ++plane_state.missed_frames;
                    if (plane_state.missed_frames > ground_cache_max_miss_) {
                        plane_state.valid = false;
                    }
                }

                const auto* cached_plane = plane_state.valid ? &plane_state : nullptr;
                if (zone[ring_idx][sector_idx].points.size() > num_min_pts_) {
                    PatchGroundResult patch_result;
                    const bool fitted = fit_patch_ground(k, zone[ring_idx][sector_idx], patch_result);
                    const bool reliable = fitted && is_reliable_plane(patch_result);

                    // Status of each patch
                    // used in checking uprightness, elevation, and flatness, respectively
                    const double ground_z_vec = reliable ? std::abs(patch_result.normal.z()) : 0.0;
                    const double ground_z_elevation = reliable ? patch_result.mean(2, 0) : 0.0;
                    const double sv_sum = reliable ? patch_result.singular_values.sum() : 0.0;
                    const double surface_variable = sv_sum > 1.0e-6
                                                        ? patch_result.singular_values.minCoeff() / sv_sum
                                                        : std::numeric_limits<double>::infinity();
                    const bool has_near_threshold = concentric_idx < num_rings_of_interest_;
                    const bool passes_local_ground_check =
                        !has_near_threshold ||
                        (ground_z_elevation <= elevation_thr_[concentric_idx]) ||
                        (flatness_thr_[concentric_idx] > surface_variable);
                    const bool use_current_as_ground =
                        reliable &&
                        (ground_z_vec >= uprightness_thr_) &&
                        passes_local_ground_check &&
                        !(concentric_idx >= num_rings_of_interest_ &&
                          using_global_thr_ &&
                          (ground_z_elevation > global_elevation_thr_));

                    if (use_current_as_ground) {
                        update_plane_state(plane_state, patch_result);
                        classify_with_plane(zone[ring_idx][sector_idx], plane_state.normal, plane_state.d,
                                            cloud_out, cloud_nonground);
                    } else if (cached_plane) {
                        ++cache_hit_count;
                        classify_with_plane(zone[ring_idx][sector_idx], cached_plane->normal, cached_plane->d,
                                            cloud_out, cloud_nonground);
                    } else if (reliable) {
                        // All points are rejected
                        cloud_nonground += patch_result.ground;
                        cloud_nonground += patch_result.nonground;
                    } else {
                        ++fallback_patch_count;
                        classify_small_patch(zone[ring_idx][sector_idx], cached_plane, cloud_out, cloud_nonground);
                    }

                }
                else if (!zone[ring_idx][sector_idx].points.empty()) {
                    ++small_patch_count;
                    classify_small_patch(zone[ring_idx][sector_idx], cached_plane, cloud_out, cloud_nonground);
                }
            }
            ++concentric_idx;
        }
    }
    if (verbose_) {
        ROS_INFO_THROTTLE(1.0,
                          "[PatchWork] input=%zu ground=%zu nonground=%zu small_patch=%d fallback=%d cache_hit=%d",
                          laserCloudIn.points.size(), cloud_out.points.size(), cloud_nonground.points.size(),
                          small_patch_count, fallback_patch_count, cache_hit_count);
    }
    cloud_out.width = static_cast<uint32_t>(cloud_out.points.size());
    cloud_out.height = 1;
    cloud_out.is_dense = false;
    cloud_nonground.width = static_cast<uint32_t>(cloud_nonground.points.size());
    cloud_nonground.height = 1;
    cloud_nonground.is_dense = false;
    *ground_cloud = cloud_out;
    *non_ground_cloud = cloud_nonground;
}
