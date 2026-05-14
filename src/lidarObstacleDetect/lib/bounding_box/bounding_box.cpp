
#include "bounding_box.h"

void checkClusterMerge(std_msgs::Header header, size_t in_cluster_id, std::vector<BoundingBoxPtr> &in_clusters,
                       std::vector<bool> &in_out_visited_clusters, std::vector<size_t> &out_merge_indices,
                       float in_merge_threshold, float overlap_tolerance)
{
  // std::cout << "checkClusterMerge" << std::endl;
  pcl::PointXYZ point_a = in_clusters[in_cluster_id]->centroid_;

  // 获取当前簇 A 的边界信息
  pcl::PointXYZ min_a = in_clusters[in_cluster_id]->minPoint_;
  pcl::PointXYZ max_a = in_clusters[in_cluster_id]->maxPoint_;

  for (size_t i = 0; i < in_clusters.size(); i++)
  {
    if (i != in_cluster_id && !in_out_visited_clusters[i])
    {
      pcl::PointXYZ point_b = in_clusters[i]->centroid_;

      //1.质心距离判断
      double distance = sqrt(pow(point_b.x - point_a.x, 2) + pow(point_b.y - point_a.y, 2));
      bool is_centroid_close = (distance <= in_merge_threshold);

      //2.AABB 包围盒重叠/邻近判断
      pcl::PointXYZ min_b = in_clusters[i]->minPoint_;
      pcl::PointXYZ max_b = in_clusters[i]->maxPoint_;

      // 检查 X 轴重叠 (A的右边 > B的左边 且 A的左边 < B的右边)
      bool overlap_x = (max_a.x + overlap_tolerance > min_b.x) && (min_a.x - overlap_tolerance < max_b.x);
      
      // 检查 Y 轴重叠
      bool overlap_y = (max_a.y + overlap_tolerance > min_b.y) && (min_a.y - overlap_tolerance < max_b.y);
      
      // 检查 Z 轴重叠 (防止把高架桥和地面的车合并不必要的Z轴检查可以去掉，但建议保留以防万一)
      bool overlap_z = (max_a.z + overlap_tolerance > min_b.z) && (min_a.z - overlap_tolerance < max_b.z);

      bool is_aabb_close = overlap_x && overlap_y && overlap_z;

      if (is_centroid_close || is_aabb_close)
      {
        in_out_visited_clusters[i] = true;
        out_merge_indices.push_back(i);
        // 递归检查
        checkClusterMerge(header, i, in_clusters, in_out_visited_clusters, out_merge_indices, in_merge_threshold, overlap_tolerance);
      }
    }
  }
}

void mergeClusters(std_msgs::Header header, std::vector<BoundingBoxPtr> &in_clusters, std::vector<BoundingBoxPtr> &out_clusters,
                   std::vector<size_t> in_merge_indices, const size_t &current_index,
                   std::vector<bool> &in_out_merged_clusters)
{
  // std::cout << "mergeClusters:" << in_merge_indices.size() << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB> sum_cloud;
  pcl::PointCloud<pcl::PointXYZI> mono_cloud;
  BoundingBoxPtr merged_cluster(new BoundingBox());

  // 只要输入列表不为空，取第一个簇的参数即可（因为所有簇的配置都是一样的）
  if (!in_clusters.empty()) {
      merged_cluster->concave_hull_dist_ = in_clusters[0]->concave_hull_dist_;
      merged_cluster->concave_hull_alpha_ = in_clusters[0]->concave_hull_alpha_;
  }

  for (size_t i = 0; i < in_merge_indices.size(); i++)
  {
    sum_cloud += *(in_clusters[in_merge_indices[i]]->pointCloud_);
    in_out_merged_clusters[in_merge_indices[i]] = true;
  }
  std::vector<int> indices(sum_cloud.points.size(), 0);
  for (size_t i = 0; i < sum_cloud.points.size(); i++)
  {
    indices[i] = i;
  }

  if (sum_cloud.points.size() > 0)
  {
    pcl::copyPointCloud(sum_cloud, mono_cloud);
    merged_cluster->SetCloud(header, mono_cloud.makeShared());
    out_clusters.push_back(merged_cluster);
  }
}

void checkAllForMerge(std_msgs::Header header, std::vector<BoundingBoxPtr> &in_clusters, std::vector<BoundingBoxPtr> &out_clusters,
                      float in_merge_threshold, float overlap_tolerance)
{
   //std::cout << "checkAllForMerge" << std::endl;
  std::vector<bool> visited_clusters(in_clusters.size(), false);
  std::vector<bool> merged_clusters(in_clusters.size(), false);
  size_t current_index = 0;
  for (size_t i = 0; i < in_clusters.size(); i++)
  {
    if (!visited_clusters[i])
    {
      visited_clusters[i] = true;
      std::vector<size_t> merge_indices;
      merge_indices.push_back(i);
      checkClusterMerge(header, i, in_clusters, visited_clusters, merge_indices, in_merge_threshold, overlap_tolerance);
      mergeClusters(header, in_clusters, out_clusters, merge_indices, current_index++, merged_clusters);
    }
  }
  for (size_t i = 0; i < in_clusters.size(); i++)
  {
    // check for clusters not merged, add them to the output
    if (!merged_clusters[i])
    {
      out_clusters.push_back(in_clusters[i]);
    }
  }

  // ClusterPtr cluster(new Cluster());
}

BoundingBox::BoundingBox()
{
  clusterMergeThreshold_ = 0.0;
  boxMergeThreshold_ = 0.0;
  concave_hull_dist_ = 0.0;
  concave_hull_alpha_ = 1.0;
}

BoundingBox::BoundingBox(ros::NodeHandle pnh)
{
  pnh.param("clusterMergeThreshold", clusterMergeThreshold_, 0.7);//聚类合并阈值，质心距离
  pnh.param("boxMergeThreshold", boxMergeThreshold_, 0.5);
  
  pnh.param("concave_hull_dist", concave_hull_dist_, 10.0); // 多少米以内允许使用凹包 
  pnh.param("concave_hull_alpha", concave_hull_alpha_, 0.85); // 凹包 alpha 值 (值越小轮廓越紧致但可能破碎，建议 0.5~1.0)

  pnh.param("large_object_length_threshold", large_object_length_threshold_, 10.0f);
  pnh.param("large_object_width_threshold", large_object_width_threshold_, 10.0f);
}

BoundingBox::~BoundingBox() {}

// 生成多边形 (Polygon) - 结合凹包和凸包
void BoundingBox::ComputePolygon(std_msgs::Header header)
{
  if (!pointCloud_ || pointCloud_->empty()) return;
  // 重新计算AABB尺寸（确保使用最新的聚类结果）
  float current_length = maxPoint_.x - minPoint_.x;
  float current_width = maxPoint_.y - minPoint_.y;
  current_length = std::abs(current_length);
  current_width = std::abs(current_width);

  // 判断是否为大物体
  bool is_large_object = (current_length > large_object_length_threshold_) || (current_width > large_object_width_threshold_);

  double dist_to_sensor = std::sqrt(std::pow(centroid_.x - 2.0, 2) + std::pow(centroid_.y, 2));

  std::vector<cv::Point2f> points;
  for (unsigned int i = 0; i < pointCloud_->points.size(); i++)
  {
    cv::Point2f pt;
    pt.x = pointCloud_->points[i].x;
    pt.y = pointCloud_->points[i].y;
    points.push_back(pt);
  }

  std::vector<cv::Point2f> hull_cv;
  cv::convexHull(points, hull_cv);

  polygon_.header = header;
  polygon_.polygon.points.clear();

  if (is_large_object && (dist_to_sensor < concave_hull_dist_)) 
  {
      // 近距离：凹包
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
      cloud_2d->points.resize(pointCloud_->points.size());
      for (size_t i = 0; i < pointCloud_->points.size(); ++i) {
          cloud_2d->points[i].x = pointCloud_->points[i].x;
          cloud_2d->points[i].y = pointCloud_->points[i].y;
          cloud_2d->points[i].z = 0.0;
      }

      pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
      concave_hull.setInputCloud(cloud_2d);
      concave_hull.setAlpha(concave_hull_alpha_);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
      concave_hull.reconstruct(*cloud_hull);

      if (!cloud_hull->points.empty()) {
          for (const auto& pt : cloud_hull->points) {
              geometry_msgs::Point32 point;
              point.x = pt.x; point.y = pt.y; point.z = minPoint_.z;
              polygon_.polygon.points.push_back(point);
          }
      } else {
          // 回退到凸包
          for (size_t i = 0; i < hull_cv.size(); i++) {
              geometry_msgs::Point32 point;
              point.x = hull_cv[i].x; point.y = hull_cv[i].y; point.z = minPoint_.z;
              polygon_.polygon.points.push_back(point);
          }
      }
  } 
  else 
  {
      // 凸包
      for (size_t i = 0; i < hull_cv.size(); i++) {
        geometry_msgs::Point32 point;
        point.x = hull_cv[i].x; point.y = hull_cv[i].y; point.z = minPoint_.z;
        polygon_.polygon.points.push_back(point);
      }
  }
}


void BoundingBox::SetCloud(std_msgs::Header header, const pcl::PointCloud<pcl::PointXYZI>::Ptr in)
{

  // 1. 初始化和 AABB 计算 (Min/Max/Avg)
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr currentCluster(new pcl::PointCloud<pcl::PointXYZRGB>);
  float min_x = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_z = -std::numeric_limits<float>::max();
  float average_x = 0, average_y = 0, average_z = 0;
  centroid_.x = 0; centroid_.y = 0; centroid_.z = 0;

  for (int i = 0; i < in->points.size(); i++)
  {
    // fill new colored cluster point by point
    pcl::PointXYZRGB p;
    p.x = in->points[i].x;
    p.y = in->points[i].y;
    p.z = in->points[i].z;

    average_x += p.x;
    average_y += p.y;
    average_z += p.z;
    centroid_.x += p.x;
    centroid_.y += p.y;
    centroid_.z += p.z;
    currentCluster->points.push_back(p);

    if (p.x < min_x) min_x = p.x;
    if (p.y < min_y) min_y = p.y;
    if (p.z < min_z) min_z = p.z;
    if (p.x > max_x) max_x = p.x;
    if (p.y > max_y) max_y = p.y;
    if (p.z > max_z) max_z = p.z;
  }
  // 赋值 Min/Max Points
  minPoint_.x = min_x; minPoint_.y = min_y; minPoint_.z = min_z;
  maxPoint_.x = max_x; maxPoint_.y = max_y; maxPoint_.z = max_z;

  // // 计算平均值和质心
  if (in->points.size() > 0)
  {
    centroid_.x /= in->points.size();
    centroid_.y /= in->points.size();
    centroid_.z /= in->points.size();

    average_x /= in->points.size();
    average_y /= in->points.size();
    average_z /= in->points.size();
  }

  averagePoint_.x = average_x;
  averagePoint_.y = average_y;
  averagePoint_.z = average_z;

  // 2. 计算基本尺寸 (AABB Dimensions)
  float length_ = maxPoint_.x - minPoint_.x;
  float width_ = maxPoint_.y - minPoint_.y;
  float height_ = maxPoint_.z - minPoint_.z;

  boundingBox_.header = header;

  // 默认 Pose 设为 AABB 中心
  boundingBox_.pose.position.x = minPoint_.x + length_ / 2;
  boundingBox_.pose.position.y = minPoint_.y + width_ / 2;
  boundingBox_.pose.position.z = minPoint_.z + height_ / 2;

  // 默认方向为 Identity (无旋转)
  boundingBox_.pose.orientation.x = 0;
  boundingBox_.pose.orientation.y = 0;
  boundingBox_.pose.orientation.z = 0;
  boundingBox_.pose.orientation.w = 1;

  boundingBox_.dimensions.x = ((length_ < 0) ? -1 * length_ : length_);
  boundingBox_.dimensions.y = ((width_ < 0) ? -1 * width_ : width_);
  boundingBox_.dimensions.z = ((height_ < 0) ? -1 * height_ : height_);


  // PCA 特征值计算 
  if (currentCluster->points.size() > 3)
  {
    pcl::PCA<pcl::PointXYZ> currentClusterPca;
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster_mono(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*currentCluster, *current_cluster_mono);
    currentClusterPca.setInputCloud(current_cluster_mono);
    eigenVectors_ = currentClusterPca.getEigenVectors();
    eigenValues_ = currentClusterPca.getEigenValues();
  }

  validCluster_ = true;
  pointCloud_ = currentCluster;
}



void BoundingBox::getBoundingBox(std_msgs::Header header,
                                 std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &points_vector,
                                 autoware_msgs::CloudClusterArray &inOutClusters)
{
  std::vector<BoundingBoxPtr> Clusters;
  for (int i = 0; i < points_vector.size(); i++)
  {
    BoundingBoxPtr cluster(new BoundingBox());

    cluster->concave_hull_dist_ = this->concave_hull_dist_;
    cluster->concave_hull_alpha_ = this->concave_hull_alpha_;
    
    cluster->SetCloud(header, points_vector[i]);
    Clusters.push_back(cluster);
  }

  // Clusters can be merged or checked in here
  // check for mergable clusters
  std::vector<BoundingBoxPtr> midClusters;
  std::vector<BoundingBoxPtr> finalClusters;

  if (Clusters.size() > 0)
    checkAllForMerge(header, Clusters, midClusters, clusterMergeThreshold_, boxMergeThreshold_);
  else
    midClusters = Clusters;

  if (midClusters.size() > 0)
    checkAllForMerge(header, midClusters, finalClusters, clusterMergeThreshold_, boxMergeThreshold_);
  else
    finalClusters = midClusters;

  for (unsigned int i = 0; i < finalClusters.size(); i++)
  {
    if (finalClusters[i]->validCluster_)
    {
      finalClusters[i]->ComputePolygon(header);  // 最终输出前才计算多边形
      autoware_msgs::CloudCluster cloudCluster;
      finalClusters[i]->ToROSMessage(header, cloudCluster);
      inOutClusters.clusters.push_back(cloudCluster);
    }
  }
  inOutClusters.header = header;
}

void BoundingBox::ToROSMessage(std_msgs::Header header, autoware_msgs::CloudCluster &outClusterMessage)
{
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(*(this->pointCloud_), cloud_msg);
  cloud_msg.header = header;
  outClusterMessage.header = header;

  outClusterMessage.cloud = cloud_msg;
  outClusterMessage.min_point.header = header;
  outClusterMessage.min_point.point.x = this->minPoint_.x;
  outClusterMessage.min_point.point.y = this->minPoint_.y;
  outClusterMessage.min_point.point.z = this->minPoint_.z;

  outClusterMessage.max_point.header = header;
  outClusterMessage.max_point.point.x = this->maxPoint_.x;
  outClusterMessage.max_point.point.y = this->maxPoint_.y;
  outClusterMessage.max_point.point.z = this->maxPoint_.z;

  outClusterMessage.avg_point.header = header;
  outClusterMessage.avg_point.point.x = this->averagePoint_.x;
  outClusterMessage.avg_point.point.y = this->averagePoint_.y;
  outClusterMessage.avg_point.point.z = this->averagePoint_.z;

  outClusterMessage.centroid_point.header = header;
  outClusterMessage.centroid_point.point.x = this->centroid_.x;
  outClusterMessage.centroid_point.point.y = this->centroid_.y;
  outClusterMessage.centroid_point.point.z = this->centroid_.z;

  // outClusterMessage.estimated_angle = this->GetOrientationAngle();

  outClusterMessage.dimensions = this->boundingBox_.dimensions;

  outClusterMessage.bounding_box = this->boundingBox_;

  outClusterMessage.convex_hull = this->polygon_;

  Eigen::Vector3f eigen_values = this->eigenValues_;
  outClusterMessage.eigen_values.x = eigen_values.x();
  outClusterMessage.eigen_values.y = eigen_values.y();
  outClusterMessage.eigen_values.z = eigen_values.z();

  Eigen::Matrix3f eigen_vectors = this->eigenVectors_;
  for (unsigned int i = 0; i < 3; i++)
  {
    geometry_msgs::Vector3 eigen_vector;
    eigen_vector.x = eigen_vectors(i, 0);
    eigen_vector.y = eigen_vectors(i, 1);
    eigen_vector.z = eigen_vectors(i, 2);
    outClusterMessage.eigen_vectors.push_back(eigen_vector);
  }

  /*std::vector<float> fpfh_descriptor = GetFpfhDescriptor(8, 0.3, 0.3);
  out_cluster_message.fpfh_descriptor.data = fpfh_descriptor;*/
}
