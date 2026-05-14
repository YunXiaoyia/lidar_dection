#include "scatterMax.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "kernel.h"
#include "scatterMaxKernel.cuh"

using namespace nvinfer1;
using nvinfer1::plugin::ScatterMax;
using nvinfer1::plugin::ScatterMaxPluginCreator;

namespace {
const char* SCATTERMAX_PLUGIN_VERSION{"1"};
const char* SCATTERMAX_PLUGIN_NAME{"ScatterMaxFunction"};
// const char* SCATTERMAX_PLUGIN_NAMESPACE{"MVF"};
}  // namespace

// Static class fields initialization
PluginFieldCollection ScatterMaxPluginCreator::mFC{};
std::vector<PluginField> ScatterMaxPluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer) {
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

ScatterMax::ScatterMax(size_t num) : num_segments_(num) {}

ScatterMax::ScatterMax(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    num_segments_ = readFromBuffer<size_t>(d);
}

nvinfer1::IPluginV2DynamicExt* ScatterMax::clone() const noexcept {
    auto* plugin = new ScatterMax(num_segments_);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

int ScatterMax::getNbOutputs() const noexcept { return 1; }

DimsExprs ScatterMax::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs,
                                          IExprBuilder& exprBuilder) noexcept {
    PLUGIN_ASSERT(outputIndex == 0);
    DimsExprs output;
    output.nbDims = 3;
    auto batch_size = inputs[0].d[0];
    auto channel = inputs[0].d[2];
    output.d[0] = batch_size;
    output.d[1] = exprBuilder.constant(num_segments_);
    output.d[2] = channel;
    return output;
}

int ScatterMax::initialize() noexcept { return 0; }

void ScatterMax::terminate() noexcept {}

/*********************************************
 *  @brief TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
 * *******************************************/
bool ScatterMax::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
                                           int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(pos < 3);
    const PluginTensorDesc& desc = inOut[pos];
    bool ret = false;
    switch (pos) {
        case 0:
            ret = (desc.type == nvinfer1::DataType::kFLOAT && desc.format == TensorFormat::kLINEAR);
            break;
        case 1:
            ret = (desc.type == nvinfer1::DataType::kINT32 && desc.format == TensorFormat::kLINEAR);
            break;
        case 2:
            ret = (desc.type == nvinfer1::DataType::kFLOAT && desc.format == TensorFormat::kLINEAR);
            break;
    }
    return ret;
}
void ScatterMax::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
                                 const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {}
// TODO, 是否需要而外的空间？
size_t ScatterMax::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0;
}

int32_t ScatterMax::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                            const void* const* inputs, void* const* outputs, void* workspace,
                            cudaStream_t stream) noexcept {
    int batchsize = inputDesc[0].dims.d[0];
    int max_num_points = inputDesc[0].dims.d[1];
    int numFeats = inputDesc[0].dims.d[2];
    nvinfer1::DataType type0 = inputDesc[0].type;
    int batch2 = inputDesc[1].dims.d[0];
    int max_num2 = inputDesc[1].dims.d[1];
    nvinfer1::DataType type1 = inputDesc[1].type;

    int out_num = outputDesc[0].dims.d[1];
    PLUGIN_ASSERT(out_num == num_segments_);

    auto src_feats = static_cast<const float*>(inputs[0]);
    auto src_index = static_cast<const int32_t*>(inputs[1]);

    auto out_data = static_cast<float*>(outputs[0]);

    // int* src_index_cpu;
    // src_index_cpu = (int*)malloc(max_num_points * sizeof(int));
    // cudaMemcpy(src_index_cpu, src_index, max_num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // float* src_feats_cpu;
    // src_feats_cpu = (float*)malloc(max_num_points * numFeats * sizeof(float));
    // cudaMemcpy(src_feats_cpu, src_feats, max_num_points * numFeats * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemsetAsync(out_data, 0, batchsize * out_num * numFeats * sizeof(float), stream);
    scatterMaxKernelLaunch(stream, batchsize, max_num_points, numFeats, num_segments_, src_feats, src_index, out_data);

    return 0;
}

size_t ScatterMax::getSerializationSize() const noexcept { return sizeof(size_t); }

void ScatterMax::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<size_t>(d, num_segments_);
    return;
}
void ScatterMax::destroy() noexcept { delete this; }

void ScatterMax::setPluginNamespace(const char* libNamespace) noexcept { mPluginNamespace = libNamespace; }

const char* ScatterMax::getPluginNamespace() const noexcept { return mPluginNamespace.c_str(); }

// Return the DataType of the plugin output at the requested index
DataType ScatterMax::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    PLUGIN_ASSERT(index == 0);
    return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ScatterMax::attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* gpuAllocator) noexcept {
    return;
}
// Detach the plugin object from its execution context.
void ScatterMax::detachFromContext() noexcept {}
const char* ScatterMax::getPluginType() const noexcept { return SCATTERMAX_PLUGIN_NAME; }

const char* ScatterMax::getPluginVersion() const noexcept { return SCATTERMAX_PLUGIN_VERSION; }

/****************************************************************/
ScatterMaxPluginCreator::ScatterMaxPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_segments", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
const char* ScatterMaxPluginCreator::getPluginName() const noexcept { return SCATTERMAX_PLUGIN_NAME; }

const char* ScatterMaxPluginCreator::getPluginVersion() const noexcept { return SCATTERMAX_PLUGIN_VERSION; }

const PluginFieldCollection* ScatterMaxPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* ScatterMaxPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int num_segments = 0;
    for (int i = 0; i < nbFields; ++i) {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "num_segments")) {
            const int* ts = static_cast<const int*>(fields[i].data);
            num_segments = ts[0];
        }
    }
    IPluginV2* plugin = new ScatterMax(num_segments);
    return plugin;
}

IPluginV2* ScatterMaxPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                      size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed,
    IPluginV2* plugin = new ScatterMax(serialData, serialLength);
    return plugin;
}

void ScatterMaxPluginCreator::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }

const char* ScatterMaxPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
