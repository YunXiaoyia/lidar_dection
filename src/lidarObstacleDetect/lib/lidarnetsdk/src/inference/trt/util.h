#pragma once

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include <memory>
#include <string>
#include <vector>

#include "NvInferRuntimeCommon.h"
#include "log/logging.h"
#include "log/logger.h"
#include "blob/infer_common.h"

namespace lidar_net {
namespace infer {
namespace tensorrt {

inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

inline void loadLibrary(const std::string& path) {
    if (!file_exists(path)) {
        PLOG_ERROR << "Plugin library file not found: " << path;
        exit(EXIT_FAILURE);
    }
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
        PLOG_ERROR << "Could not load plugin library: " << path << ", due to: " << dlerror();
        exit(EXIT_FAILURE);
    }
}

inline void CU_CHECK(cudaError_t status) {
    auto ret = (status);
    if (status != 0) {
        PLOG_ERROR << ret;
        abort();
    }
};

struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        obj->destroy();
    }
};

/**
 * @brief 自动构造和析构TRT相关指针的类
 * 如trt engine, context, plugin等
 */
template <typename T>
using unique_ptr = std::unique_ptr<T, InferDeleter>;

/**
 * @brief 生成trt engine的工厂函数
 *
 * @param engine_path trt文件路径
 * @return unique_ptr<nvinfer1::ICudaEngine> 自动析构trt engine的智能指针
 */
unique_ptr<nvinfer1::ICudaEngine> makeEngine(const std::string& engine_path,
                                             const unique_ptr<nvinfer1::IRuntime>& m_runtime);

/**
 * @brief 生成trt context的工厂函数
 *
 * @param m_engine 指向trt engine的指针
 * @return unique_ptr<nvinfer1::IExecutionContext> 自动析构trt context的智能指针
 */
unique_ptr<nvinfer1::IExecutionContext> makeContext(const unique_ptr<nvinfer1::ICudaEngine>& m_engine);

inline Dims makeDims(nvinfer1::Dims dim) {
    Dims ret;
    ret.nbDims = dim.nbDims;
    for (int i = 0; i < dim.nbDims; ++i) {
        ret.d[i] = dim.d[i];
    }
    return ret;
}

inline nvinfer1::Dims makeNvDims(Dims dim){
    nvinfer1::Dims ret;
    ret.nbDims = dim.nbDims;
    for(size_t i = 0; i < dim.nbDims; ++i){
        ret.d[i] = dim.d[i];
    }
    return ret;
}



inline DataType makeDataType(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return DataType::kFLOAT;
        case nvinfer1::DataType::kHALF:
            return DataType::kHALF;
        case nvinfer1::DataType::kINT8:
            return DataType::kINT8;
        case nvinfer1::DataType::kINT32:
            return DataType::kINT32;
        case nvinfer1::DataType::kBOOL:
            return DataType::kBOOL;
        default:
            return DataType::kFLOAT;
    }
}

}  // namespace tensorrt
}  // namespace infer
}  // namespace lidar_net
