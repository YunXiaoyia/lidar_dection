#ifndef COMMON_BLOB_INFER_COMMON
#define COMMON_BLOB_INFER_COMMON

#include <Eigen/Dense>
#include <any>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

#include "common/toml.hpp"

namespace lidar_net {
namespace infer {

/**
 * @brief
 *
 */
enum struct InfStatus : int32_t {
    SUCCESS = 0,
    INVALID_ARGUMENT = 1,
    NONE = 2,
    NOT_READY = 3,
    INVALID_MEMCPY_DIRECTION = 4,
    OUT_OF_BOUNDS = 5,
    PROGRAM_STOPED = 6,
    TASK_TIMEOUT = 7,
    ERROR = 10,
    EVENT_ERROR = 11,
    MEMORY_ERROR = 12,
    STREAM_ERROR = 13,
};

/**
 * @brief
 *
 */
enum struct MempoolStatus : int32_t {
    MEMPOOL_FREE = 0,
    MEMPOOL_READY = 1,
    MEMPOOL_BUSY = 2,
};

/**
 * @brief
 *
 */
enum struct EventStatus : int32_t {
    EVENT_STATUS_COMPLETE = 0,   // 完成
    EVENT_STATUS_NOT_READY = 1,  // 未完成
    EVENT_STATUS_RESERVED = 2,   // 预留
};

/**
 * @enum DataType
 * @brief The dtype of weights and tensors.
 *
 */
enum struct DataType : int32_t {
    kFLOAT = 0,  ///< 32-bit floating point format.
    kHALF = 1,   ///< IEEE 16-bit floating-point format.
    kINT8 = 2,   ///< 8-bit integer representing a quantized floating-point value.
    kINT32 = 3,  ///< Signed 32-bit integer format.
    kBOOL = 4,   ///< 8-bit boolean. 0 = false, 1 = true, other values undefined.
    kINT64 = 5,  ///< 64-bit integer.
};

/// @brief Convert dtype of tensor from string.
const std::map<std::string, DataType> kDataTypeMap = {
    {"float32", DataType::kFLOAT}, {"float16", DataType::kHALF}, {"int32", DataType::kINT32}, {"int8", DataType::kINT8},
    {"bool", DataType::kBOOL},     {"int64", DataType::kINT64},  {"F4", DataType::kFLOAT},    {"F2", DataType::kHALF},
    {"I4", DataType::kINT32},      {"I1", DataType::kINT8},      {"B", DataType::kBOOL},      {"I8", DataType::kINT64},
};

/// @brief Get datasize from datatype.
const std::map<DataType, int> kDataTypeSizeMap = {
    {DataType::kFLOAT, 4}, {DataType::kHALF, 2}, {DataType::kINT32, 4},
    {DataType::kINT8, 1},  {DataType::kBOOL, 1}, {DataType::kINT64, 8},
};

/// @brief Get data dtype's name string.
const std::map<DataType, std::string> kDataTypeNameMap = {
    {DataType::kFLOAT, "float32"}, {DataType::kHALF, "float16"}, {DataType::kINT32, "int32"},
    {DataType::kINT8, "int8"},     {DataType::kBOOL, "bool"},    {DataType::kINT64, "int64"},
};

/**
 * @brief structure to define the dimensions of a tensor from TensorRT.
 * @note Currently the following formats are supported for layer inputs and outputs:
 * * zero or more index dimensions followed by one channel and two spatial dimensions (e.g. CHW)
 * * one time series dimension followed by one index dimension followed by one channel dimension (i.e. TNC)
 * TensorRT can also return an invalid dims structure. This structure is represented by nbDims == -1
 * and d[i] == 0 for all d.
 *
 * TensorRT can also return an "unknown rank" dims structure. This structure is represented by nbDims == -1
 * and d[i] == -1 for all d.
 */
struct Dims {
    static const int32_t MAX_DIMS = 8;  ///< The maximum number of dimensions supported for a tensor.
    size_t nbDims;                     ///< The number of dimensions.
    int32_t d[MAX_DIMS];                ///< The extent of each dimension.

    Dims() = default;

    Dims(std::initializer_list<int32_t> list) {
        if (list.size() > MAX_DIMS) {
            std::cerr << "ERROR: Dims initializer list size is larger than MAX_DIMS" << std::endl;
            return;
        }
        this->nbDims = list.size();
        size_t i = 0;
        for (auto elem : list) {
            d[i++] = elem;
        }
    }

    Dims(const Dims& other) {
        this->nbDims = other.nbDims;
        for (size_t i = 0; i < MAX_DIMS; i++) {
            this->d[i] = other.d[i];
        }
    }

    bool operator==(const Dims& other) const {
        if (this->nbDims != other.nbDims) return false;
        for (size_t i = 0; i < this->nbDims; ++i)
            if (this->d[i] != other.d[i]) return false;
        return true;
    }

    bool operator!=(const Dims& other) const { return !(*this == other); }

    int32_t operator[](size_t i) const {
        if (i >= nbDims) {
            std::cerr << "ERROR: Dims index out of range" << std::endl;
            return 0;
        }
        return d[i];
    }

    size_t numel() const {
        int32_t vol = 1;
        for (size_t i = 0; i < this->nbDims; ++i) vol *= this->d[i];
        if (vol < 0) {
            std::cerr << "ERROR: Dims numel is negative" << std::endl;
            return 0;
        }
        return static_cast<size_t>(vol);
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "nbDims: " << this->nbDims << " d: ";
        for (size_t i = 0; i < this->nbDims; ++i) ss << this->d[i] << " ";
        return ss.str();
    }
};

inline std::ostream& operator<<(std::ostream& os, const Dims& shape) {
    os << "[";
    for (size_t i = 0; i < shape.nbDims; ++i) os << shape.d[i] << ", ";
    os << "]";
    return os;
}

/**
 * @brief Device
 *
 */
enum class DeviceType : std::uint8_t {
    HOST = 0,     ///< CPU device
    DEVICE = 1,   ///< Inference device
    UNKNOWN = 2,  ///< Unknown device
};

/// @brief Get device's name string.
const std::map<DeviceType, std::string> kDeviceTypeNameMap = {
    {DeviceType::HOST, "host"},
    {DeviceType::DEVICE, "device"},
    {DeviceType::UNKNOWN, "unknown"},
};

/**
 * @brief
 *
 */
struct BlobInfo {
    Dims shape;      ///< The dimensions of the blob.
    DataType dtype;  ///< The dtype of the blob.

    BlobInfo(const Dims& shape, DataType dtype) : shape(shape), dtype(dtype) {}

    size_t numel() const { return shape.numel(); }

    size_t nbytes() const {
        int elem_size = kDataTypeSizeMap.at(dtype);
        return std::accumulate(shape.d, shape.d + shape.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
    }

    bool operator==(const BlobInfo& other) const {
        if (this->dtype == other.dtype && this->shape == other.shape) return true;
        return false;
    }

    bool operator!=(const BlobInfo& other) const { return !(*this == other); }
};

inline std::ostream& operator<<(std::ostream& os, const BlobInfo& blob_info) {
    os << "shape: " << blob_info.shape << "\n";
    os << "data dtype " << kDataTypeNameMap.at(blob_info.dtype) << "\n";
    return os;
}

}  // namespace infer
}  // namespace lidar_net

#endif /* COMMON_BLOB_INFER_COMMON */
