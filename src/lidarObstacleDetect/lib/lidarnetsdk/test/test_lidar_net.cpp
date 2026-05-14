
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <algorithm>

#include <iostream>
#include <memory>
#include <experimental/filesystem>

#include <set>
#include "lidar_net/lidar_net_det.h"

#include "common/npy.hpp"

#define TOML_EXCEPTIONS 0
#include "common/toml.hpp"
#include "spdlog/spdlog.h"

#include "user_log.h"
#include "common/deepways_time.h"

#include <opencv2/opencv.hpp>

namespace fs = std::experimental::filesystem;

using MatrixF = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;

inline bool file_exists(const std::string_view& file) {
    if (!fs::exists(file)) {
        std::cerr << "Could not find file in " << file << std::endl;
        std::cerr << "current directory is " << fs::current_path() << std::endl;
        return false;
    }
    return true;
}
// cloud: Nx5
void showResult(Eigen::MatrixXf cloud, const std::vector<lidar_net::base::BoundingBox>& objects_bbox) {
    const auto& points = cloud;
    const auto& objects = objects_bbox;
    float range_x_max = 100.0;
    float range_x_min = -0.0;
    float range_y_max = 50.0;
    float range_y_min = -50.0;
    float x_res = 0.2;
    float y_res = 0.2;
    int rows = (range_x_max - range_x_min) / x_res;
    int cols = (range_y_max - range_y_min) / y_res;
    // int rows = 1000;
    // int cols = 1000;

    cv::Mat show_image = cv::Mat::zeros(rows, cols, CV_8UC3);
    // for (const auto& pt : points->points_) {
    for (int i = 0; i < points.rows(); ++i) {
        int x = (points(i, 0) - range_x_min) / x_res;
        int y = (points(i, 1) - range_y_min) / y_res;
        if (x < 0 || y < 0 || x >= rows || y >= cols) {
            continue;
        }

        show_image.at<cv::Vec3b>(x, y) = cv::Vec3b(255, 255, 255);  // bgr
    }
    std::vector<std::vector<cv::Point>> contours;
    for (const auto& object : objects) {
        const auto& bbox = object;
        int cols = bbox.corners2d.cols();  // 4
        for (int c = 0; c < cols; ++c) {
            int cx = (bbox.corners2d(0, c) - range_x_min) / x_res;
            int cy = (bbox.corners2d(1, c) - range_y_min) / y_res;
            int cx1 = (bbox.corners2d(0, (c + 1) % cols) - range_x_min) / x_res;
            int cy1 = (bbox.corners2d(1, (c + 1) % cols) - range_y_min) / y_res;
            cv::line(show_image, cv::Point(cy, cx), cv::Point(cy1, cx1), cv::Scalar(0, 0, 255));
        }
    }

    cv::flip(show_image, show_image, 0);
    cv::flip(show_image, show_image, 1);
    cv::imshow("lidar_net", show_image);
    cv::waitKey();
}

void detect(const std::string config_path, const std::string config_type, const std::string data_path) {
    std::set<std::string> data_files;

    fs::path directory_path(data_path);
    if (fs::is_directory(directory_path)) {
        for (const auto& frame : fs::directory_iterator(directory_path)) {
            if (fs::is_regular_file(frame.path()) && frame.path().extension() == ".npy") {
                std::string file_name = fs::absolute(frame.path());
                data_files.insert(file_name);
            }
        }
    }

    // init logger
    spdlog::set_level(spdlog::level::info);
    std::shared_ptr<SpdlogUserLogger> user_logger = std::make_shared<SpdlogUserLogger>();

    // // toml config
    // const toml::table tbl = toml::parse_file(config_path).table();
    // const auto toml_params = toml::node_view(tbl["perception"]["detector"]);
    // std::stringstream toml_param_stream;
    // toml_param_stream << toml_params;

    // detector
    lidar_net::LidarNetDetectorUniPtr detector = std::make_unique<lidar_net::LidarNetDetector>();
    bool ret_new = detector->Init(config_path, user_logger, config_type);
    if (!ret_new) {
        std::cout << "detector init failed" << std::endl;
        return;
    }
    // // REQUIRE(ret_new);

    // for(const auto& file : data_files){
    // for(size_t i = 0; i < data_files.size(); i++){
    for (auto& file : data_files) {
        std::cout << file << std::endl;
        // load points
        std::vector<unsigned long> shape{};
        std::vector<float> data;
        bool fortran_order = false;
        npy::LoadArrayFromNumpy(file, shape, fortran_order, data);

        assert(shape.size() == 2);  // N x 6
        std::cout << "points num = " << shape[0] << ", " << shape[1] << std::endl;
        MatrixF points = decltype(points)(shape[0], shape[1]);
        std::memcpy(points.data(), data.data(), data.size() * sizeof(float));

        Eigen::MatrixXf point_4 = Eigen::MatrixXf::Identity(points.rows(), 4);
        point_4 = points.leftCols(4);
        std::vector<lidar_net::base::BoundingBox> objects;

        PERF_BLOCK_START();
        detector->ProcessingEigenCol(point_4, objects);
        PERF_BLOCK_END("lidar_net_detector process");

        std::cout << "objects : " << objects.size() << std::endl;
        // showResult(point_4, objects);
    }
}

int main(int argc, char** argv) {
    // TODO: 测试机器上gtest有问题，后面有时间改成 gtest
    // 获取自己路径
    // 获取当前可执行文件的绝对路径
    fs::path my_path = fs::canonical("/proc/self/exe");
    // 在绝对路径的父目录是.../bin, 在bin/的父目录里找配置文件
    std::string data_path = my_path.parent_path().string() + "/data/at128_test_data";

    // test dsvt
    std::string dsvt_config_path = my_path.parent_path().string() + "/config/dsvt_config_at128.toml";
    if (!file_exists(dsvt_config_path)) {
        std::cerr << "dsvt_config_path: " << dsvt_config_path << " is not exist, test failed." << std::endl;
        exit(-1);
    }
    detect(dsvt_config_path, "toml", data_path);

    // test flatformer
    std::string flatformer_config_path = my_path.parent_path().string() + "/config/flatformer_config_at128.toml";
    if (!file_exists(flatformer_config_path)) {
        std::cerr << "flatformer_config_path: " << flatformer_config_path << " is not exist, test failed." << std::endl;
        exit(-1);
    }
    detect(flatformer_config_path, "toml", data_path);
    return 0;
}