#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <optional>
#include <set>

#include "blob/infer_common.h"
#include "blob/synced_buffer.h"
#include "common/json.hpp"
#include "util.h"

namespace lidar_net {
namespace infer {

namespace trt = tensorrt;

using NameBlobInfo = std::map<std::string, BlobInfo>;

class TrtInfer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TrtInfer() = default;

    bool Init(const std::string_view param_node, const std::string& config_type);

    bool Infer();

    std::vector<void*> Infer(const std::vector<std::string>& input_names, const std::vector<void*>& input_values,
                             const std::vector<std::string>& output_names);

    void Infer(const std::vector<std::string>& input_names, const std::vector<void*>& input_values,
               const std::vector<std::string>& output_names, const std::vector<void*>& output_values);

    void InferDynamicInput(const std::vector<std::string>& input_names, const std::vector<std::vector<int>>& input_shapes);

    void* GetInput(const std::string& name) const;

    void* GetOutput(const std::string& name) const;

    size_t GetInputCount() const;

    size_t GetOutputCount() const;

    size_t GetInputIndex(const std::string& name) const;

    size_t GetOutputIndex(const std::string& name) const;

    BlobInfo GetInputBlobInfo(const std::string& name) const;

    BlobInfo GetOutputBlobInfo(const std::string& name) const;

    std::string GetInputName(size_t index) const;

    std::string GetOutputName(size_t index) const;

   private:
    bool CheckConfig(NameBlobInfo toml_config, NameBlobInfo engine_config);

    std::pair<NameBlobInfo, NameBlobInfo> LoadEngine(const std::string& engine_path, const NameBlobInfo& toml_config, const std::set<std::string>& dynamic_names);

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0);

    /// @brief Copy the contents of input host buffers to input device buffers synchronously.
    inline void copyInputToDevice() { this->memcpyBuffers(true, false, false); }

    /// @brief Copy the contents of output device buffers to output host buffers synchronously.
    inline void copyOutputToHost() { this->memcpyBuffers(false, true, false); }

    /// @brief Copy the contents of input host buffers to input device buffers asynchronously.
    inline void copyInputToDeviceAsync(const cudaStream_t& stream = 0) {
        this->memcpyBuffers(true, false, true, stream);
    }

    /// @brief Copy the contents of output device buffers to output host buffers asynchronously.
    inline void copyOutputToHostAsync(const cudaStream_t& stream = 0) {
        this->memcpyBuffers(false, true, true, stream);
    }

    void updateBindings();

    cudaStream_t net_stream_;
    cudaEvent_t net_event_;

    trt::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;         ///< 指向引擎
    trt::unique_ptr<nvinfer1::IRuntime> trt_runtime_;           ///< 指向运行时
    trt::unique_ptr<nvinfer1::IExecutionContext> trt_context_;  ///< 指向上下文管理器

    /// The index of the input and output blob
    std::map<std::string, int> inputs_name_trtindex_, outputs_name_trtindex_;
    std::map<int, std::string> inputs_trtindex_name_, outputs_trtindex_name_;

    std::set<std::string> dynamic_shapes_;

    /// The vector of pointers to managed buffers
    std::map<std::string, std::unique_ptr<SyncedTensor>> inputs_tensor_, outputs_tensor_;

    std::vector<void*> device_bindings_;  ///< The vector of device buffers needed for engine execution
};

}  // namespace infer
}  // namespace lidar_net
