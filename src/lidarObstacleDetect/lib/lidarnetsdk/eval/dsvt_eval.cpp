/**
 * @file dsvt_eval.cpp
 * @author xxx (xxx@hotmail.com)
 * @brief 對已有測試集生成DSVT的測試結果
 * @version 0.1
 * @date 2024-08-21
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <algorithm>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <vector>

#include "lidar_net/lidar_net_det.h"
#include "common/json.hpp"

#include "pcd_load.h"
#include "indicators.hpp"
#include "json_operate.h"

namespace fs = std::experimental::filesystem;
using namespace lidar_net;
using namespace indicators;

struct Sample {
    lidar_net::RowMatrixXf cloud;
    std::vector<lidar_net::base::BoundingBox> bboxes;
    std::string file_name;
};

void LoadData(const std::string& data_dir, std::vector<Sample>& samples, int max_num = -1) {
    std::vector<fs::path> json_files;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".json") {
            json_files.push_back(entry.path());
        }
    }

    // sort
    std::sort(json_files.begin(), json_files.end());

    show_console_cursor(false);
    ProgressBar bar0{option::BarWidth(50),
                    option::Start{"["},
                    option::Fill{"="},
                    option::Lead{">"},
                    option::Remainder{" "},
                    option::End{"]"},
                    option::PostfixText{"Load data"},
                    option::ForegroundColor{Color::green},
                    option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    // load data
    int count = 0;
    for (const auto& file : json_files) {
        count++;
        Sample sample;

        std::string file_name = file.stem().string();
        sample.file_name = file_name;

        // std::cout << "Loading " << file.filename().string() << std::endl;
        bar0.set_option(option::PostfixText{"Load data: " + file_name});
        bar0.set_progress(count / (float)json_files.size() * 100);

        // load json
        std::ifstream ifs(file);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open " << file << std::endl;
            continue;
        }
        nlohmann::json json_data;
        try {
            ifs >> json_data;
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            return;
        }
        ifs.close();

        // load cloud
        std::string points_path = json_data["points_path"];
        // std::cout << "Loading points " << points_path << std::endl;
        pcd_read(points_path, sample.cloud);

        samples.push_back(sample);

        if (max_num > 0 && count >= max_num) {
            break;
        }

    }  // end for json_files
    bar0.mark_as_completed();
}

int main(int argc, char** argv) {
    // get input
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file> <data_path> <save_path>" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string data_path = argv[2];
    std::string save_path = argv[3];

    // init detector
    LidarNetDetector detector;
    bool ret = detector.Init(config_file, nullptr, "toml");
    if (!ret) {
        std::cerr << "Failed to init detector" << std::endl;
        return -1;
    }

    // load data
    std::vector<Sample> samples;
    LoadData(data_path, samples);
    int total = samples.size();

    // process
    show_console_cursor(false);
    ProgressBar bar1 {
        option::BarWidth(50),
        option::Start{"["}, 
        option::Fill{"="}, 
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Inference"},
        option::ForegroundColor{Color::green},
        option::FontStyles {
          std::vector<FontStyle> { FontStyle::bold }
        }
    };
    for (int i = 0; i < total; i++) {
        bar1.set_progress(i / (float)total * 100);
        detector.ProcessingEigenCol(samples[i].cloud, samples[i].bboxes);
    }
    bar1.mark_as_completed();

    // save result
    ProgressBar bar2{option::BarWidth(50),
                    option::Start{"["},
                    option::Fill{"="},
                    option::Lead{">"},
                    option::Remainder{" "},
                    option::End{"]"},
                    option::PostfixText{"Saving result"},
                    option::ForegroundColor{Color::green},
                    option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};
    for (int i = 0; i < total; i++) {
        bar2.set_progress(i / (float)total * 100);
        nlohmann::json json_object;
        json::ObjectsToJson(samples[i].bboxes, json_object);
        json::JsonToFile(json_object, save_path + "/" + samples[i].file_name + ".json");
    }
    bar2.mark_as_completed();

    return 0;
}
