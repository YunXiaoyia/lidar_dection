/**
 * @file pcd_load.h
 * @brief load pcd for test or demo
 * @author xxx
 * @date 2024-07-23
 * @version 1.0
 * @note Additional notes about the file
 */

#pragma once

#include <iostream>
#include <fstream>
#include "Eigen/Dense"

namespace lidar_net {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct PointField {
    std::string name;

    uint offset = 0;
    std::uint8_t datatype = 0;
    uint count = 0;

};  // struct PCLPointField

static void pcd_read_stream(std::istream& stream, RowMatrixXf& cloud) {
    std::size_t nr_points = 0;
    int point_step = 0;
    std::vector<PointField> fields;

    for (int jj = 0; jj < 11; ++jj) {
        std::string line;
        std::getline(stream, line);
        // Ignore empty lines
        if (line.empty()) {
            continue;
        }

        std::stringstream sstream(line);
        sstream.imbue(std::locale::classic());

        std::string line_type;
        sstream >> line_type;

        // Get the field indices (check for COLUMNS too for backwards compatibility)
        if ((line_type.substr(0, 6) == "FIELDS") || (line_type.substr(0, 7) == "COLUMNS")) {
            std::string tt;
            while (sstream >> tt) {
                fields.emplace_back();
                fields.back().name = tt;
            }
            continue;
        }

        // Get the field sizes
        if (line_type.substr(0, 4) == "SIZE") {
            int offset = 0;
            for (size_t i = 0; i < fields.size(); ++i) {
                int col_type;
                sstream >> col_type;
                fields[i].offset = offset;  // estimate and save the data offsets
                offset += col_type;
                // field_sizes[i] = col_type;                      // save a temporary
                // copy
            }
            point_step = offset;
            continue;
        }

        // Get the number of points
        if (line_type.substr(0, 6) == "POINTS") {
            sstream >> nr_points;
            // Need to allocate: N * point_step
            continue;
        }

        // Read the header + comments line by line until we get to <DATA>
        if (line_type.substr(0, 4) == "DATA") {
            break;
        }
    }

    const std::size_t data_size = nr_points * point_step;
    std::vector<char> buf(data_size);
    char* buf_start = buf.data();

    stream.read(buf_start, data_size);

    cloud.resize(nr_points, 4);
    for (std::size_t i = 0; i < nr_points; ++i) {
        for (size_t j = 0; j < fields.size(); ++j) {
            if (fields[j].name == "x") {
                cloud(i, 0) = *reinterpret_cast<float*>(buf_start + i * point_step + fields[j].offset);
            } else if (fields[j].name == "y") {
                cloud(i, 1) = *reinterpret_cast<float*>(buf_start + i * point_step + fields[j].offset);
            } else if (fields[j].name == "z") {
                cloud(i, 2) = *reinterpret_cast<float*>(buf_start + i * point_step + fields[j].offset);
            } else if (fields[j].name == "intensity") {
                cloud(i, 3) = *reinterpret_cast<float*>(buf_start + i * point_step + fields[j].offset);
            }
        }
    }
}

static void pcd_read(const std::string& file_name, RowMatrixXf& cloud) {
    std::ifstream fs;
    fs.open(file_name.c_str(), std::ios::binary);
    if (!fs.is_open() || fs.fail()) {
        std::cerr << "[PCD Loader] Couldn't open file ! " << file_name;
        fs.close();
        return;
    }
    pcd_read_stream(fs, cloud);
    fs.close();
}

}  // namespace lidar_net
