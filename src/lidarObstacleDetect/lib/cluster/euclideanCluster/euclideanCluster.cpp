

#include "euclideanCluster.h"

EuclideanCluster::EuclideanCluster(ros::NodeHandle nh, ros::NodeHandle pnh)
{
    pnh.param("clusterTolerance", clusterTolerance_, 0.3);//距离容忍度,两个点要被视为同一聚类（簇）的最大距离
    pnh.param("minClusterSize", minClusterSize_, 10);//一个有效聚类所需要包含的最小点数
    pnh.param("maxClusterSize", maxClusterSize_, 5000);//一个有效聚类所能包含的最大点数
    pnh.param("use_multiple_thres", use_multiple_thres_, true);//是否使用多阈值聚类
    pnh.param("clustering_distances", clustering_distances_, {0.2, 0.3, 0.5, 0.7, 1.0});//多阈值聚类所使用的距离容忍度列表
    pnh.param("clustering_ranges", clustering_ranges_, {10, 20, 30, 45});
}
void EuclideanCluster::cluster_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &clusters)
{
    //设置查找方式－kdtree
    pcl::search::Search<pcl::PointXYZI>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZI>>(new pcl::search::KdTree<pcl::PointXYZI>);

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(clusterTolerance_); // 2cm
    ec.setMinClusterSize(minClusterSize_);     // 100
    ec.setMaxClusterSize(maxClusterSize_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(in);
    ec.extract(clusters);
}
/**
 * @brief 根据点到原点的距离对点云进行分段聚类，使用不同的距离阈值以适应不同区域的点云密度。
 * 
 * 该函数实现了一种基于欧几里得距离的聚类方法。如果启用多阈值模式（use_multiple_thres_ 为 true），
 * 则根据点到原点的距离将点云划分为多个子集，并为每个子集使用不同的聚类距离阈值，从而提高远距离点云的聚类效果。
 * 否则，使用统一的距离阈值对整个点云进行聚类。
 *
 * @param in 输入的点云数据指针，类型为 pcl::PointCloud<pcl::PointXYZI>::Ptr
 * @param outCloudPtr 输出的聚类结果合并后的点云指针，类型为 pcl::PointCloud<pcl::PointXYZI>::Ptr&
 * @param points_vector 存储每个聚类结果的点云指针向量，类型为 std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>&
 */
void EuclideanCluster::segmentByDistance(const pcl::PointCloud<pcl::PointXYZI>::Ptr in,
                                         pcl::PointCloud<pcl::PointXYZI>::Ptr &outCloudPtr, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &points_vector)
{
    // cluster the pointcloud according to the distance of the points using different thresholds (not only one for the entire pc)
    // in this way, the points farther in the pc will also be clustered
    std::vector<pcl::PointIndices> clusterIndices;
    if (!use_multiple_thres_)
    {

        cluster_vector(in, clusterIndices);

        for (auto it = clusterIndices.begin(); it != clusterIndices.end(); ++it)
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cluster(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::copyPointCloud(*in, it->indices, *temp_cluster);
            *outCloudPtr += *temp_cluster;
            points_vector.push_back(temp_cluster);
        }
    }
    else
    {
        size_t segment_count = clustering_ranges_.size() + 1;
        if (clustering_distances_.size() < segment_count)
        {
            segment_count = clustering_distances_.size();
        }
        if (segment_count == 0)
        {
            return;
        }

        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_segments_array(segment_count);
        for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            cloud_segments_array[i] = tmp_cloud;
        }

        for (unsigned int i = 0; i < in->points.size(); i++)
        {
            pcl::PointXYZI current_point;
            current_point.x = in->points[i].x;
            current_point.y = in->points[i].y;
            current_point.z = in->points[i].z;
            current_point.intensity = in->points[i].intensity;

            float origin_distance = sqrt(pow(current_point.x-5.0, 2) + pow(current_point.y, 2));

            size_t segment_idx = 0;
            while (segment_idx + 1 < segment_count && origin_distance >= clustering_ranges_[segment_idx])
            {
                ++segment_idx;
            }
            cloud_segments_array[segment_idx]->points.push_back(current_point);
        }

        std::vector<std::thread> thread_vec(cloud_segments_array.size());
        // 1. 创建一个 vector 来存储所有 future
        std::vector<std::shared_future<std::vector<pcl::PointIndices>>> futures;

        // 2. 循环一：仅用于启动所有线程

        for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
        {
            auto promiseObj = std::make_shared<std::promise<std::vector<pcl::PointIndices>>>();

            std::shared_future<std::vector<pcl::PointIndices>> futureObj = promiseObj->get_future();
            futures.push_back(futureObj); // 存入 vector
            thread_vec[i] = std::thread(&EuclideanCluster::clusterIndicesMultiThread, this, cloud_segments_array[i], std::ref(clustering_distances_[i]), promiseObj);
        }
        // 3. 循环二：等待所有线程完成并收集结果
        for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
        {
            // 现在 .get() 会等待对应的线程完成
            clusterIndices = futures[i].get(); 
            
            // (确保 cloud_segments_array[i] 在线程访问时是安全的，
            //  由于 pcl::PointCloud::Ptr 是智能指针，且线程只读，这里是安全的)

            for (int j = 0; j < clusterIndices.size(); j++)
            {
                pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cluster(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::copyPointCloud(*cloud_segments_array[i], clusterIndices[j], *temp_cluster);
                *outCloudPtr += *temp_cluster;
                points_vector.push_back(temp_cluster);
            }
        }
        for (int i = 0; i < thread_vec.size(); i++)
        {
            thread_vec[i].join();
        }
    }
}

void EuclideanCluster::clusterIndicesMultiThread(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, double in_max_cluster_distance,
                                                std::shared_ptr<std::promise<std::vector<pcl::PointIndices>>> promiseObj)
{
    // make it flat
    // for (size_t i = 0; i < cloud_2d->points.size(); i++)
    // {
    //     cloud_2d->points[i].z = 0;
    // }
    pcl::search::Search<pcl::PointXYZI>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZI>>(new pcl::search::KdTree<pcl::PointXYZI>);
    std::vector<pcl::PointIndices> indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(in_max_cluster_distance); 
    ec.setMinClusterSize(minClusterSize_);          
    ec.setMaxClusterSize(maxClusterSize_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(in_cloud_ptr);
    ec.extract(indices);

    promiseObj->set_value(indices);
}
