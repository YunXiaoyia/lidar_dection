// Copyright (c) OpenMMLab. All rights reserved.
#include "trt_grid_sampler.hpp"

#include <assert.h>

#include <chrono>
#include <thread>

#include "trt_grid_sampler_kernel.hpp"
#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GridSample"};
}  // namespace

TRTGridSampler::TRTGridSampler(const std::string &name, int mode, int paddingMode, bool alignCorners)
    : TRTPluginBase(name), mMode(mode), mPaddingMode(paddingMode), mAlignCorners(alignCorners) {}

TRTGridSampler::TRTGridSampler(const std::string name, const void *data, size_t length) : TRTPluginBase(name) {
    deserialize_value(&data, &length, &mMode);
    deserialize_value(&data, &length, &mPaddingMode);
    deserialize_value(&data, &length, &mAlignCorners);
}

nvinfer1::IPluginV2DynamicExt *TRTGridSampler::clone() const TRT_NOEXCEPT {
    TRTGridSampler *plugin = new TRTGridSampler(mLayerName, mMode, mPaddingMode, mAlignCorners);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs TRTGridSampler::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                        int nbInputs,
                                                        nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
    nvinfer1::DimsExprs ret;
    ret.nbDims = inputs[0].nbDims;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    for (int i = 2; i < ret.nbDims; ++i) {
        ret.d[i] = inputs[1].d[i - 1];
    }
    return ret;
}

bool TRTGridSampler::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
                                               int nbOutputs) TRT_NOEXCEPT {
    if (pos == 0) {
        return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
                ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    } else {
        return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
    }
}

void TRTGridSampler::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                     const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) TRT_NOEXCEPT {
    // Validate input arguments
}

size_t TRTGridSampler::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                        const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const TRT_NOEXCEPT {
    return 0;
}

int TRTGridSampler::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                            const void *const *inputs, void *const *outputs, void *workSpace,
                            cudaStream_t stream) TRT_NOEXCEPT {
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    nvinfer1::Dims grid_dims = inputDesc[1].dims;
    nvinfer1::Dims output_dims = outputDesc[0].dims;

    GridSamplerInterpolation interp_mode = GridSamplerInterpolation::Bilinear;
    switch (mMode) {
        case 0:
            interp_mode = GridSamplerInterpolation::Bilinear;
            break;
        case 1:
            interp_mode = GridSamplerInterpolation::Nearest;
            break;
        default:
            break;
    }

    GridSamplerPadding padding_mode = GridSamplerPadding::Zeros;
    switch (mPaddingMode) {
        case 0:
            padding_mode = GridSamplerPadding::Zeros;
            break;

        case 1:
            padding_mode = GridSamplerPadding::Border;
            break;

        case 2:
            padding_mode = GridSamplerPadding::Reflection;
            break;
        default:
            break;
    }

    auto data_type = inputDesc[0].type;

    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            grid_sample<float>((float *)outputs[0], (float *)inputs[0], (float *)inputs[1], &(output_dims.d[0]),
                               &(input_dims.d[0]), &(grid_dims.d[0]), input_dims.nbDims, interp_mode, padding_mode,
                               mAlignCorners, stream);
            break;
        default:
            return 1;
            break;
    }

    return 0;
}

nvinfer1::DataType TRTGridSampler::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                     int nbInputs) const TRT_NOEXCEPT {
    return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTGridSampler::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTGridSampler::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int TRTGridSampler::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTGridSampler::getSerializationSize() const TRT_NOEXCEPT {
    return serialized_size(mMode) + serialized_size(mPaddingMode) + serialized_size(mAlignCorners);
}

void TRTGridSampler::serialize(void *buffer) const TRT_NOEXCEPT {
    serialize_value(&buffer, mMode);
    serialize_value(&buffer, mPaddingMode);
    serialize_value(&buffer, mAlignCorners);
}

////////////////////// creator /////////////////////////////

TRTGridSamplerCreator::TRTGridSamplerCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("mode", nullptr, nvinfer1::PluginFieldType::kCHAR, 8));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("padding_mode", nullptr, nvinfer1::PluginFieldType::kCHAR, 10));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *TRTGridSamplerCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTGridSamplerCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *TRTGridSamplerCreator::createPlugin(const char *name,
                                                         const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
    int mode = 0;
    int paddingMode = 0;
    bool alignCorners = false;

    for (int i = 0; i < fc->nbFields; i++) {
        if (fc->fields[i].data == nullptr) {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("mode") == 0) {
            const std::string modeStr = static_cast<const char *>(fc->fields[i].data);
            if (modeStr.substr(0, 8) == "bilinear") {
                mode = 0;
            } else if (modeStr.substr(0, 7) == "nearest") {
                mode = 1;
            } else {
                throw std::runtime_error("[GridSamplerPlugin] Unknown mode " + std::string(modeStr));
            }
        }

        if (field_name.compare("padding_mode") == 0) {
            const std::string paddingModeStr = static_cast<const char *>(fc->fields[i].data);
            if (paddingModeStr.substr(0, 5) == "zeros") {
                paddingMode = 0;
            } else if (paddingModeStr.substr(0, 6) == "border") {
                paddingMode = 1;
            } else if (paddingModeStr.substr(0, 10) == "reflection") {
                paddingMode = 2;
            } else {
                throw std::runtime_error("[GridSamplerPlugin] Unknown padding_mode " + std::string(paddingModeStr));
            }
        }

        if (field_name.compare("align_corners") == 0) {
            const int *d = static_cast<const int *>(fc->fields[i].data);
            alignCorners = static_cast<bool>(d[0]);
        }
    }

    TRTGridSampler *plugin = new TRTGridSampler(name, mode, paddingMode, alignCorners);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

nvinfer1::IPluginV2 *TRTGridSamplerCreator::deserializePlugin(const char *name, const void *serialData,
                                                              size_t serialLength) TRT_NOEXCEPT {
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new TRTGridSampler(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

}  // namespace mmdeploy