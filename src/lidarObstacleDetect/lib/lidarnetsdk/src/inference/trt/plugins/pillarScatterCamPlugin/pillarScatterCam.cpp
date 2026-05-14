/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cstring>
#include "kernel.h"
#include "pillarScatterCamKernel.cuh"
#include "pillarScatterCam.h"

using namespace nvinfer1;
using nvinfer1::plugin::PillarScatterCamPlugin;
using nvinfer1::plugin::PillarScatterCamPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"PillarScatterCamPlugin"};

// Static class fields initialization
PluginFieldCollection PillarScatterCamPluginCreator::mFC{};
std::vector<PluginField> PillarScatterCamPluginCreator::mPluginAttributes;

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

PillarScatterCamPlugin::PillarScatterCamPlugin(size_t h, size_t w) : feature_y_size_(h), feature_x_size_(w) {}

PillarScatterCamPlugin::PillarScatterCamPlugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    feature_y_size_ = readFromBuffer<size_t>(d);
    feature_x_size_ = readFromBuffer<size_t>(d);
}

nvinfer1::IPluginV2DynamicExt* PillarScatterCamPlugin::clone() const noexcept {
    auto* plugin = new PillarScatterCamPlugin(feature_y_size_, feature_x_size_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs PillarScatterCamPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                int nbInputs,
                                                                nvinfer1::IExprBuilder& exprBuilder) noexcept {
    assert(outputIndex == 0);
    nvinfer1::DimsExprs output;
    auto batch_size = inputs[0].d[0];
    output.nbDims = 4;
    output.d[0] = batch_size;
    output.d[1] = inputs[0].d[2];
    output.d[2] = exprBuilder.constant(feature_y_size_);
    output.d[3] = exprBuilder.constant(feature_x_size_);
    return output;
}

bool PillarScatterCamPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                                       int nbOutputs) noexcept {
    assert(nbInputs == 3);
    assert(nbOutputs == 1);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0) {
#ifdef __aarch64__
        return in.type == nvinfer1::DataType::kFLOAT && in.format == TensorFormat::kLINEAR;
#else
        return (in.type == nvinfer1::DataType::kFLOAT || in.type == nvinfer1::DataType::kHALF) &&
               (in.format == TensorFormat::kLINEAR);
#endif
    }
    if (pos == 1) {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2) {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3) {
        return (in.type == inOut[0].type) && (in.format == TensorFormat::kLINEAR || in.format == TensorFormat::kHWC8);
    }
    return false;
}

void PillarScatterCamPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                             const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    return;
}

size_t PillarScatterCamPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                                const nvinfer1::PluginTensorDesc* outputs,
                                                int nbOutputs) const noexcept {
    return 0;
}

int PillarScatterCamPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        int batchSize = inputDesc[0].dims.d[0];
        int maxPillarNum = inputDesc[0].dims.d[1];
        int numFeatures = inputDesc[0].dims.d[2];

        nvinfer1::DataType inputType = inputDesc[0].type;

        auto coords_data = static_cast<const unsigned int*>(inputs[1]);
        auto params_data = static_cast<const unsigned int*>(inputs[2]);

        unsigned int featureY = feature_y_size_;
        unsigned int featureX = feature_x_size_;

        int status = -1;

        if (inputType == nvinfer1::DataType::kHALF) {
            auto pillar_features_data = static_cast<const half*>(inputs[0]);
            auto spatial_feature_data = static_cast<half*>(outputs[0]);
            cudaMemsetAsync(spatial_feature_data, 0, batchSize * numFeatures * featureY * featureX * sizeof(half),
                            stream);
            status = pillarScatterCamKernelLaunch<half>(batchSize, maxPillarNum, numFeatures, pillar_features_data,
                                                        coords_data, params_data, featureX, featureY,
                                                        spatial_feature_data, stream);
            PLUGIN_ASSERT(status == STATUS_SUCCESS);
            return status;
        } else if (inputType == nvinfer1::DataType::kFLOAT) {
            auto pillar_features_data = static_cast<const float*>(inputs[0]);
            auto spatial_feature_data = static_cast<float*>(outputs[0]);
            cudaMemsetAsync(spatial_feature_data, 0, batchSize * numFeatures * featureY * featureX * sizeof(float),
                            stream);
            status = pillarScatterCamKernelLaunch<float>(batchSize, maxPillarNum, numFeatures, pillar_features_data,
                                                         coords_data, params_data, featureX, featureY,
                                                         spatial_feature_data, stream);
            PLUGIN_ASSERT(status == STATUS_SUCCESS);
            return status;
        } else {
            PLUGIN_ASSERT(status == STATUS_SUCCESS);
            return status;
        }
    } catch (const std::exception& e) {
        caughtError(e);
    }
    return -1;
}

nvinfer1::DataType PillarScatterCamPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                             int nbInputs) const noexcept {
    return inputTypes[0];
}

const char* PillarScatterCamPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }

const char* PillarScatterCamPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

int PillarScatterCamPlugin::getNbOutputs() const noexcept { return 1; }

int PillarScatterCamPlugin::initialize() noexcept { return 0; }

void PillarScatterCamPlugin::terminate() noexcept {}

size_t PillarScatterCamPlugin::getSerializationSize() const noexcept { return 3 * sizeof(size_t); }

void PillarScatterCamPlugin::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<size_t>(d, feature_y_size_);
    writeToBuffer<size_t>(d, feature_x_size_);
}

void PillarScatterCamPlugin::destroy() noexcept { delete this; }

void PillarScatterCamPlugin::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }

const char* PillarScatterCamPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

PillarScatterCamPluginCreator::PillarScatterCamPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dense_shape", nullptr, PluginFieldType::kINT32, 2));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* PillarScatterCamPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }

const char* PillarScatterCamPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

const PluginFieldCollection* PillarScatterCamPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* PillarScatterCamPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int target_h = 0;
    int target_w = 0;
    for (int i = 0; i < nbFields; ++i) {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "dense_shape")) {
            const int* ts = static_cast<const int*>(fields[i].data);
            target_h = ts[0];
            target_w = ts[1];
        }
    }
    IPluginV2* plugin = new PillarScatterCamPlugin(target_h, target_w);
    return plugin;
}

IPluginV2* PillarScatterCamPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                            size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed,
    IPluginV2* plugin = new PillarScatterCamPlugin(serialData, serialLength);
    return plugin;
}

void PillarScatterCamPluginCreator::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }

const char* PillarScatterCamPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
