#include <NvInferRuntime.h>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "util.h"

namespace lidar_net {
namespace infer {
namespace tensorrt {

unique_ptr<nvinfer1::ICudaEngine> makeEngine(const std::string& engine_path,
                                             const unique_ptr<nvinfer1::IRuntime>& m_runtime) {
    std::ifstream engine_stream(engine_path, std::ios::binary);

    /// - Check engine byte size
    engine_stream.seekg(0, std::ifstream::end);
    auto fsize = engine_stream.tellg();
    engine_stream.seekg(0, std::ifstream::beg);

    /// - Read engineFile as a vector<char>, where size(int8_t) == 1
    std::vector<char> engine_data(fsize);
    engine_stream.read(engine_data.data(), fsize);

    /// - reset the unique_ptr
    unique_ptr<nvinfer1::ICudaEngine> m_engine(m_runtime->deserializeCudaEngine(engine_data.data(), fsize, nullptr));

    if (m_engine.get() == nullptr) {
        PLOG_ERROR << "ERROR: creating CUDA Engine.\n";
        return nullptr;
    } else {
        return m_engine;
    }
};

unique_ptr<nvinfer1::IExecutionContext> makeContext(const unique_ptr<nvinfer1::ICudaEngine>& m_engine) {
    unique_ptr<nvinfer1::IExecutionContext> m_context(m_engine->createExecutionContext());
    if (m_context.get() == nullptr) {
        PLOG_ERROR << "ERROR: failed to create CUDA Context.\n";
        return nullptr;
    }
    return m_context;
}

}  // namespace tensorrt
}  // namespace infer
}  // namespace lidar_net