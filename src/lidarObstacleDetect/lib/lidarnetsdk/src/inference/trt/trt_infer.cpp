#include <type_traits>
#include <utility>

#include "common/filesystem.h"
#include "blob/infer_common.h"
#include "log/logger.h"
#include "log/logging.h"
#include "trt_logger.h"
#include "trt_infer.h"
#include "util.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"

// extern "C" {
// bool initLibNvInferPlugins(void* logger, const char* libNamespace);
// }

namespace lidar_net {
namespace infer {

using NameBlobInfo = std::map<std::string, BlobInfo>;

inline void CheckCUDA(cudaError_t error) {
    if (error != cudaSuccess) {
        PLOG_FATAL << "[TRTINFER] [Inference] CUDA error: " << cudaGetErrorString(error);
    }
};

inline bool file_exists(const std::string_view& file) {
    if (!fs::exists(file)) {
        PLOG_FATAL << "[TRTINFER] Could not find file in " << file;
        PLOG_FATAL << "[TRTINFER] current directory is " << fs::current_path();
        return false;
    }
    return true;
}

BlobInfo GetBlobInfo(const nlohmann::json& shape, const std::string& dtype) {
    if (!shape.is_array()) {
        PLOG_FATAL << "[TRTINFER] ptr shape is null.";
        exit(-1);
    }

    Dims dims;
    dims.nbDims = shape.size();

    size_t i = 0;
    for (size_t j = 0; j < shape.size(); ++j) {
        dims.d[i++] = shape[j].get<int32_t>();
    }

    return BlobInfo(dims, kDataTypeMap.at(dtype));
};

NameBlobInfo GetBlobInfoFromConfig(const nlohmann::json& info) {
    NameBlobInfo blob_info_map;
    for (nlohmann::json::const_iterator it = info.begin(); it != info.end(); ++it) {
        const std::string name = it.key();
        auto values = it.value();
        // Get input name, shape, and dtype.
        if (!values.is_object()) PLOG_FATAL << "[TRTINFER] inputs (" << name << ") must have value.";
        const auto shape = values["shape"];
        if (!shape.is_array()) PLOG_FATAL << "[TRTINFER] inputs (" << name << ") must have shape.";
        const auto dtype = values["dtype"].get<std::string>();

        // Get BlobInfo
        blob_info_map.insert({std::string(name), GetBlobInfo(shape, dtype)});
    }
    return blob_info_map;
}

std::optional<std::pair<NameBlobInfo, NameBlobInfo>> JsonInit(const nlohmann::json& param_node) {
    // 获得json中IO描述
    NameBlobInfo inputs_config, outputs_config;
    if (param_node.contains("inputs")) {
        inputs_config = GetBlobInfoFromConfig(param_node["inputs"]);
    } else {
        PLOG_FATAL << "[TRTINFER] inputs must be a table.";
        return {};
    }

    if (param_node.contains("outputs")) {
        outputs_config = GetBlobInfoFromConfig(param_node["outputs"]);
    } else {
        PLOG_FATAL << "[TRTINFER] outputs must be a table.";
        return {};
    }
    return std::make_pair(inputs_config, outputs_config);
}

bool TrtInfer::Init(const std::string_view param_node, const std::string& config_type) {
    const std::string param_node_str(param_node);
    std::string config_type_cpy(config_type);
    std::transform(config_type_cpy.begin(), config_type_cpy.end(), config_type_cpy.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::string model_path;
    std::optional<std::pair<NameBlobInfo, NameBlobInfo>> io_configs;
    if (config_type_cpy == "json") {
        const nlohmann::json config = nlohmann::json::parse(param_node_str);
        io_configs = JsonInit(config);
        model_path = config["engine"].get<std::string>();
        dynamic_shapes_ = config["dynamic_shapes"].get<std::set<std::string>>();  // dynamic

    } else {
        PLOG_FATAL << "[Detector] config_type is not json";
        return false;
    }

    NameBlobInfo inputs_config, outputs_config;
    if (io_configs) {
        inputs_config = io_configs->first;
        outputs_config = io_configs->second;
    } else {
        return false;
    }

    // 初始化插件
    initLibNvInferPlugins(&trt::trtLogger, "");

    // 装载引擎，获得引擎中IO描述
    PLOG_INFO << "[TRTINFER] Loading engine from " << model_path;
    auto [inputs_config_trt, outputs_config_trt] = this->LoadEngine(model_path, inputs_config, dynamic_shapes_);

    // 检查toml与引擎中IO描述是否一致
    if (!CheckConfig(inputs_config, inputs_config_trt)) {
        PLOG_FATAL << "[TRTINFER] inputs config is not consistent with model.";
        return false;
    }
    if (!CheckConfig(outputs_config, outputs_config_trt)) {
        PLOG_FATAL << "[TRTINFER] outputs config is not consistent with model.";
        return false;
    }

    // 验证无误，开始初始化所有输入输出
    int num_bindings = trt_engine_->getNbBindings();
    for (int i = 0; i < num_bindings; ++i) {
        const auto name = std::string(trt_engine_->getBindingName(i));
        if (trt_engine_->bindingIsInput(i)) {
            auto iter = inputs_config_trt.find(name);
            auto tensor = std::make_unique<SyncedTensor>(iter->second);
            device_bindings_.emplace_back(tensor->buffer->get());
            this->inputs_tensor_.insert({iter->first, std::move(tensor)});
        } else {
            auto iter = outputs_config_trt.find(name);
            auto tensor = std::make_unique<SyncedTensor>(iter->second);
            device_bindings_.emplace_back(tensor->buffer->get());
            this->outputs_tensor_.insert({iter->first, std::move(tensor)});
        }
    }

    return true;
}

bool TrtInfer::Infer() {
    this->copyInputToDeviceAsync(net_stream_);
    bool status_infer = trt_context_->enqueueV2(device_bindings_.data(), net_stream_, nullptr);

    if (!status_infer) {
        PLOG_ERROR << "[TRTINFER]: TensorRT inference fail to enqueue.";
        return false;
    }
    this->copyOutputToHostAsync(net_stream_);

    CheckCUDA(cudaStreamSynchronize(net_stream_));

    return true;
}

std::vector<void*> TrtInfer::Infer(const std::vector<std::string>& input_names, const std::vector<void*>& input_values,
                                   const std::vector<std::string>& output_names) {
    // 检查输入输出名称是否正确
    if (input_names.size() != input_values.size()) {
        PLOG_FATAL << "[TRTINFER] input names and values size is not equal.";
        return {};
    }

    // 将各项输入拷贝到inputs_tensor_中
    for (size_t i = 0; i < input_names.size(); ++i) {
        auto iter = this->inputs_tensor_.find(input_names[i]);
        if (iter == this->inputs_tensor_.end()) {
            PLOG_FATAL << "[TRTINFER] input name " << input_names[i] << " is not found.";
            return {};
        }
        void* dst = iter->second->buffer->getHostBuffer();
        memcpy(dst, input_values[i], iter->second->nbytes());
    }

    // 开始推理
    this->Infer();

    // 将输出作为返回值
    std::vector<void*> output_values;
    for (const auto& output_name : output_names) {
        auto iter = this->outputs_tensor_.find(output_name);
        if (iter == this->outputs_tensor_.end()) {
            PLOG_FATAL << "[TRTINFER] output name " << output_name << " is not found.";
            return {};
        }
        output_values.emplace_back(iter->second->buffer->getHostBuffer());
    }

    return output_values;
}

void TrtInfer::Infer(const std::vector<std::string>& input_names, const std::vector<void*>& input_values,
                     const std::vector<std::string>& output_names, const std::vector<void*>& output_values) {
    // 检查输入输出名称是否正确
    if (input_names.size() != input_values.size()) {
        PLOG_FATAL << "[TRTINFER] input names and values size is not equal.";
        return;
    }
    if (output_names.size() != output_values.size()) {
        PLOG_FATAL << "[TRTINFER] output names and values size is not equal.";
        return;
    }

    // 将各项输入拷贝到inputs_tensor_中
    for (size_t i = 0; i < input_names.size(); ++i) {
        auto iter = this->inputs_tensor_.find(input_names[i]);
        if (iter == this->inputs_tensor_.end()) {
            PLOG_FATAL << "[TRTINFER] input name " << input_names[i] << " is not found.";
            return;
        }
        void* dst = iter->second->buffer->getHostBuffer();
        memcpy(dst, input_values[i], iter->second->nbytes());
    }

    // 开始推理
    this->Infer();

    // 将输出作为返回值
    for (size_t i = 0; i < output_names.size(); ++i) {
        const auto& output_name = output_names[i];
        auto iter = this->outputs_tensor_.find(output_name);
        if (iter == this->outputs_tensor_.end()) {
            PLOG_FATAL << "[TRTINFER] output name " << output_name << " is not found.";
            return;
        }
        void* dst = output_values[i];
        const void* src = iter->second->buffer->getHostBuffer();
        memcpy(dst, src, iter->second->nbytes());
    }
}

void TrtInfer::InferDynamicInput(const std::vector<std::string>& input_names, const std::vector<std::vector<int>>& input_shapes){
    if (input_names.size() != input_shapes.size()) {
        PLOG_FATAL << "[TRTINFER] input names and values size is not equal.";
        return;
    }

    size_t dyninput_num = input_names.size();
    for(size_t i = 0; i < dyninput_num; ++i){
        std::string name = input_names.at(i);
        std::vector<int> shape = input_shapes.at(i);
        const size_t input_index = GetInputIndex(name);
        nvinfer1::Dims nvDims;
        nvDims.nbDims = shape.size();
        for(int32_t i = 0; i < nvDims.nbDims; ++i){
            nvDims.d[i] = shape[i];
        }
        trt_context_->setBindingDimensions(input_index, nvDims);
    }

    // 开始推理
    this->Infer();
}

void* TrtInfer::GetInput(const std::string& name) const {
    auto iter = this->inputs_tensor_.find(name);
    if (iter == this->inputs_tensor_.end()) {
        PLOG_FATAL << "[TRTINFER] input name " << name << " is not found.";
        return nullptr;
    }
    return iter->second->buffer->getHostBuffer();
}

void* TrtInfer::GetOutput(const std::string& name) const {
    auto iter = this->outputs_tensor_.find(name);
    if (iter == this->outputs_tensor_.end()) {
        PLOG_FATAL << "[TRTINFER] output name " << name << " is not found.";
        return nullptr;
    }
    return iter->second->buffer->getHostBuffer();
}

size_t TrtInfer::GetInputCount() const { return this->inputs_tensor_.size(); }

size_t TrtInfer::GetOutputCount() const { return this->outputs_tensor_.size(); }

size_t TrtInfer::GetInputIndex(const std::string& name) const { return this->inputs_name_trtindex_.at(name); }

size_t TrtInfer::GetOutputIndex(const std::string& name) const { return this->outputs_name_trtindex_.at(name); }

BlobInfo TrtInfer::GetInputBlobInfo(const std::string& name) const { return this->inputs_tensor_.at(name)->info; }

BlobInfo TrtInfer::GetOutputBlobInfo(const std::string& name) const { return this->outputs_tensor_.at(name)->info; }

std::string TrtInfer::GetInputName(size_t index) const { return this->inputs_trtindex_name_.at(index); }

std::string TrtInfer::GetOutputName(size_t index) const { return this->outputs_trtindex_name_.at(index); }

bool TrtInfer::CheckConfig(NameBlobInfo toml_config, NameBlobInfo engine_config) {
    // 检查输入输出一致性
    for (auto& pair : engine_config) {
        auto iter = toml_config.find(pair.first);
        if (iter == toml_config.end()) {
            PLOG_FATAL << "[TRTINFER] input/output (" << pair.first << ") is not in toml config.";
            return false;
        } else {
            if (iter->second != pair.second) {
                PLOG_FATAL << "[TRTINFER] input/output (" << pair.first << ") " << iter->second << " / " << pair.second
                           << " is not the same as toml config.";
                return false;
            }
        }
    }
    return true;
}

std::pair<NameBlobInfo, NameBlobInfo> TrtInfer::LoadEngine(const std::string& engine_file, const NameBlobInfo& toml_config, const std::set<std::string>& dynamic_names ) {
    // 检查引擎文件路径是否存在
    if (!file_exists(engine_file)) {
        PLOG_FATAL << "[TRTINFER] engine file not found in " << engine_file;
        throw std::invalid_argument("engine file not found");
    }

    // 创建stream
    trt::CU_CHECK(cudaStreamCreate(&net_stream_));

    // 创建运行时
    trt_runtime_ = trt::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trt::trtLogger));

    // 创建引擎
    trt_engine_ = trt::makeEngine(engine_file, trt_runtime_);

    // 初始化上下文管理器
    trt_context_ = trt::makeContext(trt_engine_);

    int num_bindings = trt_engine_->getNbBindings();

    // Set Dynamic Shapes
    if(dynamic_names.size() > 0){
        for(int i = 0; i < num_bindings; i++){
            const auto name = std::string(trt_engine_->getBindingName(i));
            if(dynamic_names.find(name) != dynamic_names.end()){
                auto iter = toml_config.find(name);
                BlobInfo info = iter->second;

                auto nvDims = trt::makeNvDims(info.shape);
                trt_context_->setBindingDimensions(i, nvDims);
            }
        }
    }

    // 遍历检查引擎中的输入输出
    NameBlobInfo inputs_config_trt, outputs_config_trt;
    for (int i = 0; i < num_bindings; ++i) {
        const auto name = std::string(trt_engine_->getBindingName(i));
        const auto dtype = trt::makeDataType(trt_engine_->getBindingDataType(i));
        const auto shape = trt::makeDims(trt_context_->getBindingDimensions(i));

        if (trt_engine_->bindingIsInput(i)) {
            PLOG_INFO << "[TRTINFER] Input: " << name << " " << shape;
            inputs_config_trt.insert({name, BlobInfo(shape, dtype)});
            this->inputs_name_trtindex_.insert({name, i});
            this->inputs_trtindex_name_.insert({i, name});
        } else {
            PLOG_INFO << "[TRTINFER] Output: " << name << " " << shape;
            outputs_config_trt.insert({name, BlobInfo(shape, dtype)});
            this->outputs_name_trtindex_.insert({name, i});
            this->outputs_trtindex_name_.insert({i, name});
        }
    }

    return std::make_pair(inputs_config_trt, outputs_config_trt);
}

void TrtInfer::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async,
                             const cudaStream_t& stream) {
    if (copyInput) {
        for (auto& pair : this->inputs_tensor_) {
            pair.second->buffer->memcpyBuffers(deviceToHost, async, stream);
        }
    } else {
        for (auto& pair : this->outputs_tensor_) {
            pair.second->buffer->memcpyBuffers(deviceToHost, async, stream);
        }
    }
}

void TrtInfer::updateBindings() {
    int num_bindings = trt_engine_->getNbBindings();
    for (int i = 0; i < num_bindings; ++i) {
        const auto name = std::string(trt_engine_->getBindingName(i));
        if (trt_engine_->bindingIsInput(i)) {
            const auto& tensor = this->inputs_tensor_.at(name);
            auto index = this->inputs_name_trtindex_.at(name);
            device_bindings_[index] = tensor->buffer->get();
        } else {
            const auto& tensor = this->outputs_tensor_.at(name);
            auto index = this->outputs_name_trtindex_.at(name);
            device_bindings_[index] = tensor->buffer->get();
        }
    }
}

}  // namespace infer
}  // namespace lidar_net
