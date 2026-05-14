#include <cassert>
#include <iostream>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "NvInferRuntimeCommon.h"
#include "multiViewVoxelGenerator.h"
#include "multiViewVoxelGeneratorKernel.cuh"

#define checkCudaErrors(status)                                                                                   \
    {                                                                                                             \
        if (status != 0) {                                                                                        \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " at line " << __LINE__ << " in file " \
                      << __FILE__ << " error status: " << status << std::endl;                                    \
            abort();                                                                                              \
        }                                                                                                         \
    }

using namespace nvinfer1;
using nvinfer1::plugin::MultiViewVoxelGeneratorPlugin;
using nvinfer1::plugin::MultiViewVoxelGeneratorPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"MultiViewVoxelGeneratorFunction"};

// Static class fields initialization
PluginFieldCollection MultiViewVoxelGeneratorPluginCreator::mFC{};
std::vector<PluginField> MultiViewVoxelGeneratorPluginCreator::mPluginAttributes;

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

// OK
MultiViewVoxelGeneratorPlugin::MultiViewVoxelGeneratorPlugin(
    std::vector<size_t> grid_size, std::vector<float> voxel_size, std::vector<float> points_range,
    std::vector<size_t> rv_grid_size, std::vector<float> rv_voxel_size, std::vector<float> rv_points_range)
    : grid_size_(grid_size),
      voxel_size_(voxel_size),
      points_range_(points_range),
      rv_grid_size_(rv_grid_size),
      rv_voxel_size_(rv_voxel_size),
      rv_points_range_(rv_points_range) {}

MultiViewVoxelGeneratorPlugin::MultiViewVoxelGeneratorPlugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    int grid_size_x = readFromBuffer<size_t>(d);
    size_t grid_size_y = readFromBuffer<size_t>(d);
    size_t grid_size_z = readFromBuffer<size_t>(d);
    grid_size_.clear();
    grid_size_.push_back(grid_size_x);
    grid_size_.push_back(grid_size_y);
    grid_size_.push_back(grid_size_z);

    float voxel_size_x = readFromBuffer<float>(d);
    float voxel_size_y = readFromBuffer<float>(d);
    float voxel_size_z = readFromBuffer<float>(d);
    voxel_size_.clear();
    voxel_size_.push_back(voxel_size_x);
    voxel_size_.push_back(voxel_size_y);
    voxel_size_.push_back(voxel_size_z);

    float points_range_min_x = readFromBuffer<float>(d);
    float points_range_min_y = readFromBuffer<float>(d);
    float points_range_min_z = readFromBuffer<float>(d);
    float points_range_max_x = readFromBuffer<float>(d);
    float points_range_max_y = readFromBuffer<float>(d);
    float points_range_max_z = readFromBuffer<float>(d);
    points_range_.clear();
    points_range_.push_back(points_range_min_x);
    points_range_.push_back(points_range_min_y);
    points_range_.push_back(points_range_min_z);
    points_range_.push_back(points_range_max_x);
    points_range_.push_back(points_range_max_y);
    points_range_.push_back(points_range_max_z);

    size_t rv_grid_size_x = readFromBuffer<size_t>(d);
    size_t rv_grid_size_y = readFromBuffer<size_t>(d);
    size_t rv_grid_size_z = readFromBuffer<size_t>(d);
    rv_grid_size_.clear();
    rv_grid_size_.push_back(rv_grid_size_x);
    rv_grid_size_.push_back(rv_grid_size_y);
    rv_grid_size_.push_back(rv_grid_size_z);

    float rv_voxel_size_x = readFromBuffer<float>(d);
    float rv_voxel_size_y = readFromBuffer<float>(d);
    float rv_voxel_size_z = readFromBuffer<float>(d);
    rv_voxel_size_.clear();
    rv_voxel_size_.push_back(rv_voxel_size_x);
    rv_voxel_size_.push_back(rv_voxel_size_y);
    rv_voxel_size_.push_back(rv_voxel_size_z);

    float rv_points_range_min_x = readFromBuffer<float>(d);
    float rv_points_range_min_y = readFromBuffer<float>(d);
    float rv_points_range_min_z = readFromBuffer<float>(d);
    float rv_points_range_max_x = readFromBuffer<float>(d);
    float rv_points_range_max_y = readFromBuffer<float>(d);
    float rv_points_range_max_z = readFromBuffer<float>(d);
    rv_points_range_.clear();
    rv_points_range_.push_back(rv_points_range_min_x);
    rv_points_range_.push_back(rv_points_range_min_y);
    rv_points_range_.push_back(rv_points_range_min_z);
    rv_points_range_.push_back(rv_points_range_max_x);
    rv_points_range_.push_back(rv_points_range_max_y);
    rv_points_range_.push_back(rv_points_range_max_z);
}

nvinfer1::IPluginV2DynamicExt* MultiViewVoxelGeneratorPlugin::clone() const noexcept {
    auto* plugin = new MultiViewVoxelGeneratorPlugin(grid_size_, voxel_size_, points_range_, rv_grid_size_,
                                                     rv_voxel_size_, rv_points_range_);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

int MultiViewVoxelGeneratorPlugin::getNbOutputs() const noexcept { return 4; }

nvinfer1::DimsExprs MultiViewVoxelGeneratorPlugin::getOutputDimensions(int outputIndex,
                                                                       const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                                       nvinfer1::IExprBuilder& exprBuilder) noexcept {
    assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());

    auto batch_size = inputs[0].d[0];
    auto max_num_points = inputs[0].d[1];

    if (outputIndex == 0) {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = exprBuilder.constant(1);
        dim0.d[1] = max_num_points;
        dim0.d[2] = exprBuilder.constant(9);
        return dim0;
    }
    if (outputIndex == 1) {
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 3;
        dim1.d[0] = exprBuilder.constant(1);
        dim1.d[1] = max_num_points;
        dim1.d[2] = exprBuilder.constant(9);
        return dim1;
    }
    if (outputIndex == 2) {
        nvinfer1::DimsExprs dim2{};
        dim2.nbDims = 3;
        dim2.d[0] = exprBuilder.constant(1);
        dim2.d[1] = max_num_points;
        dim2.d[2] = exprBuilder.constant(3);
        return dim2;
    }
    if (outputIndex == 3) {
        nvinfer1::DimsExprs dim3{};
        dim3.nbDims = 3;
        dim3.d[0] = exprBuilder.constant(1);
        dim3.d[1] = max_num_points;
        dim3.d[2] = exprBuilder.constant(3);
        return dim3;
    }
}

int MultiViewVoxelGeneratorPlugin::initialize() noexcept { return 0; }

void MultiViewVoxelGeneratorPlugin::terminate() noexcept {}

bool MultiViewVoxelGeneratorPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                              int nbInputs, int nbOutputs) noexcept {
    assert(nbInputs == 2);
    assert(nbOutputs == 4);
    const PluginTensorDesc& in = inOut[pos];

    if (pos == 0)  // points_padded
    {
        bool flag = (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        return flag;
    }
    if (pos == 1)  // points_mask
    {
        bool flag = (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        return flag;
    }
    if (pos == 2 || pos == 3 || pos == 4 || pos == 5)  // output
    {
        bool flag = (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        return flag;
    }
    return false;
}

void MultiViewVoxelGeneratorPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
                                                    const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {}

size_t MultiViewVoxelGeneratorPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                                       const nvinfer1::PluginTensorDesc* outputs,
                                                       int nbOutputs) const noexcept {
    int batchsize = inputs[0].dims.d[0];
    int max_num_points = inputs[0].dims.d[1];
    int points_feat_num = inputs[0].dims.d[2];

    size_t size_points_rv = batchsize * max_num_points * points_feat_num * sizeof(float);

    size_t size_bev_voxel_sum = batchsize * grid_size_[0] * grid_size_[1] * grid_size_[2] * 3 * sizeof(float);
    size_t size_bev_points_num_per_voxel = batchsize * grid_size_[0] * grid_size_[1] * grid_size_[2] * sizeof(int);

    size_t size_rv_voxel_sum = batchsize * rv_grid_size_[0] * rv_grid_size_[1] * rv_grid_size_[2] * 3 * sizeof(float);
    size_t size_rv_points_num_per_voxel =
        batchsize * rv_grid_size_[0] * rv_grid_size_[1] * rv_grid_size_[2] * sizeof(int);

    size_t workspaces[5];
    workspaces[0] = size_points_rv;
    workspaces[1] = size_bev_voxel_sum;
    workspaces[2] = size_bev_points_num_per_voxel;
    workspaces[3] = size_rv_voxel_sum;
    workspaces[4] = size_rv_points_num_per_voxel;

    return calculateTotalWorkspaceSize(workspaces, 5);
}

int32_t MultiViewVoxelGeneratorPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                                               const void* const* inputs, void* const* outputs, void* workspace,
                                               cudaStream_t stream) noexcept {
    int batchsize = inputDesc[0].dims.d[0];
    int max_num_points = inputDesc[0].dims.d[1];
    int points_feat_num = inputDesc[0].dims.d[2];

    int points_status_num = outputDesc[0].dims.d[2];

    // TRT input
    const float* pointcloud = const_cast<float*>((const float*)inputs[0]);
    const float* points_mask = const_cast<float*>((const float*)inputs[1]);

    // TRT output
    float* bev_features = (float*)(outputs[0]);
    float* rv_features = (float*)(outputs[1]);
    float* bev_voxel_coors = (float*)(outputs[2]);
    float* rv_voxel_coors = (float*)(outputs[3]);

    // workspace
    size_t size_points_rv = batchsize * max_num_points * points_feat_num * sizeof(float);
    size_t size_bev_voxel_sum = batchsize * grid_size_[0] * grid_size_[1] * grid_size_[2] * 3 * sizeof(float);
    size_t size_bev_points_num_per_voxel = batchsize * grid_size_[0] * grid_size_[1] * grid_size_[2] * sizeof(int);
    size_t size_rv_voxel_sum = batchsize * rv_grid_size_[0] * rv_grid_size_[1] * rv_grid_size_[2] * 3 * sizeof(float);
    size_t size_rv_points_num_per_voxel =
        batchsize * rv_grid_size_[0] * rv_grid_size_[1] * rv_grid_size_[2] * sizeof(int);

    size_t workspaces[5];
    workspaces[0] = size_points_rv;
    workspaces[1] = size_bev_voxel_sum;
    workspaces[2] = size_bev_points_num_per_voxel;
    workspaces[3] = size_rv_voxel_sum;
    workspaces[4] = size_rv_points_num_per_voxel;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 5);

    float* points_rv = static_cast<float*>(workspace);
    float* bev_voxel_sum =
        reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(points_rv), size_points_rv));
    int* bev_points_num_per_voxel =
        reinterpret_cast<int*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(bev_voxel_sum), size_bev_voxel_sum));
    float* rv_voxel_sum = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(bev_points_num_per_voxel), size_bev_points_num_per_voxel));
    int* rv_points_num_per_voxel =
        reinterpret_cast<int*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(rv_voxel_sum), size_rv_voxel_sum));

    // Initialize workspace memory
    checkCudaErrors(cudaMemsetAsync(points_rv, 0, total_workspace, stream));

    // Initialize output
    size_t size_bev_features = batchsize * max_num_points * points_status_num * sizeof(float);
    size_t size_rv_features = batchsize * max_num_points * points_status_num * sizeof(float);
    size_t size_bev_voxel_coors = batchsize * max_num_points * points_feat_num * sizeof(float);
    size_t size_rv_voxel_coors = batchsize * max_num_points * points_feat_num * sizeof(float);

    checkCudaErrors(cudaMemsetAsync(bev_features, 0, size_bev_features, stream));
    checkCudaErrors(cudaMemsetAsync(rv_features, 0, size_rv_features, stream));
    checkCudaErrors(cudaMemsetAsync(bev_voxel_coors, 0, size_bev_voxel_coors, stream));
    checkCudaErrors(cudaMemsetAsync(rv_voxel_coors, 0, size_rv_voxel_coors, stream));

    generatorPointsCylinder_launch(batchsize, max_num_points, pointcloud, points_mask, points_rv, stream);
    // bev voxel
    pointsToVoxels_launch(batchsize, max_num_points, points_feat_num, grid_size_[0], grid_size_[1], grid_size_[2],
                          voxel_size_[0], voxel_size_[1], voxel_size_[2], points_range_[0], points_range_[1],
                          points_range_[2], points_range_[3], points_range_[4], points_range_[5], pointcloud,
                          points_mask, bev_voxel_coors, stream);
    // rv voxel
    pointsToVoxels_launch(batchsize, max_num_points, points_feat_num, rv_grid_size_[0], rv_grid_size_[1],
                          rv_grid_size_[2], rv_voxel_size_[0], rv_voxel_size_[1], rv_voxel_size_[2],
                          rv_points_range_[0], rv_points_range_[1], rv_points_range_[2], rv_points_range_[3],
                          rv_points_range_[4], rv_points_range_[5], points_rv, points_mask, rv_voxel_coors, stream);
    // bev status
    pointsStatus_launch(batchsize, max_num_points, points_feat_num, points_status_num, grid_size_[0], grid_size_[1],
                        grid_size_[2], voxel_size_[0], voxel_size_[1], voxel_size_[2], points_range_[0],
                        points_range_[1], points_range_[2], pointcloud, points_mask, bev_voxel_coors, bev_voxel_sum,
                        bev_points_num_per_voxel, bev_features, stream);
    // rv status
    pointsStatus_launch(batchsize, max_num_points, points_feat_num, points_status_num, rv_grid_size_[0],
                        rv_grid_size_[1], rv_grid_size_[2], rv_voxel_size_[0], rv_voxel_size_[1], rv_voxel_size_[2],
                        rv_points_range_[0], rv_points_range_[1], rv_points_range_[2], points_rv, points_mask,
                        rv_voxel_coors, rv_voxel_sum, rv_points_num_per_voxel, rv_features, stream);
    return 0;
}

size_t MultiViewVoxelGeneratorPlugin::getSerializationSize() const noexcept {
    return 3 * sizeof(size_t) * 2 + 3 * sizeof(float) * 2 + 6 * sizeof(float) * 2;
}

void MultiViewVoxelGeneratorPlugin::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    // writeToBuffer<size_t>(d, num_segments_);
    writeToBuffer<size_t>(d, grid_size_[0]);
    writeToBuffer<size_t>(d, grid_size_[1]);
    writeToBuffer<size_t>(d, grid_size_[2]);

    writeToBuffer<float>(d, voxel_size_[0]);
    writeToBuffer<float>(d, voxel_size_[1]);
    writeToBuffer<float>(d, voxel_size_[2]);

    writeToBuffer<float>(d, points_range_[0]);
    writeToBuffer<float>(d, points_range_[1]);
    writeToBuffer<float>(d, points_range_[2]);
    writeToBuffer<float>(d, points_range_[3]);
    writeToBuffer<float>(d, points_range_[4]);
    writeToBuffer<float>(d, points_range_[5]);

    writeToBuffer<size_t>(d, rv_grid_size_[0]);
    writeToBuffer<size_t>(d, rv_grid_size_[1]);
    writeToBuffer<size_t>(d, rv_grid_size_[2]);

    writeToBuffer<float>(d, rv_voxel_size_[0]);
    writeToBuffer<float>(d, rv_voxel_size_[1]);
    writeToBuffer<float>(d, rv_voxel_size_[2]);

    writeToBuffer<float>(d, rv_points_range_[0]);
    writeToBuffer<float>(d, rv_points_range_[1]);
    writeToBuffer<float>(d, rv_points_range_[2]);
    writeToBuffer<float>(d, rv_points_range_[3]);
    writeToBuffer<float>(d, rv_points_range_[4]);
    writeToBuffer<float>(d, rv_points_range_[5]);
    return;
}

void MultiViewVoxelGeneratorPlugin::destroy() noexcept { delete this; }

void MultiViewVoxelGeneratorPlugin::setPluginNamespace(const char* libNamespace) noexcept {
    mPluginNamespace = libNamespace;
}

const char* MultiViewVoxelGeneratorPlugin::getPluginNamespace() const noexcept { return mPluginNamespace.c_str(); }

// Return the DataType of the plugin output at the requested index
DataType MultiViewVoxelGeneratorPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                          int nbInputs) const noexcept {
    return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void MultiViewVoxelGeneratorPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                                    IGpuAllocator* gpuAllocator) noexcept {
    return;
}

// Detach the plugin object from its execution context.
void MultiViewVoxelGeneratorPlugin::detachFromContext() noexcept {}

const char* MultiViewVoxelGeneratorPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }

const char* MultiViewVoxelGeneratorPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

/****************************************************************/

MultiViewVoxelGeneratorPluginCreator::MultiViewVoxelGeneratorPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("grid_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("points_range", nullptr, PluginFieldType::kFLOAT32, 1));

    mPluginAttributes.emplace_back(PluginField("rv_grid_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rv_voxel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rv_points_range", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MultiViewVoxelGeneratorPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }

const char* MultiViewVoxelGeneratorPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

const PluginFieldCollection* MultiViewVoxelGeneratorPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* MultiViewVoxelGeneratorPluginCreator::createPlugin(const char* name,
                                                              const PluginFieldCollection* fc) noexcept {
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    std::vector<size_t> grid_size, rv_grid_size;
    std::vector<float> voxel_size, points_range, rv_voxel_size, rv_points_range;

    for (int i = 0; i < nbFields; ++i) {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "grid_size")) {
            const int* d = static_cast<const int*>(fields[i].data);
            grid_size.clear();
            grid_size.emplace_back(d[0]);
            grid_size.emplace_back(d[1]);
            grid_size.emplace_back(d[2]);
        } else if (!strcmp(attr_name, "voxel_size")) {
            const float* d = static_cast<const float*>(fields[i].data);
            voxel_size.clear();
            voxel_size.emplace_back(d[0]);
            voxel_size.emplace_back(d[1]);
            voxel_size.emplace_back(d[2]);
        } else if (!strcmp(attr_name, "points_range")) {
            const float* d = static_cast<const float*>(fields[i].data);
            points_range.clear();
            points_range.emplace_back(d[0]);
            points_range.emplace_back(d[1]);
            points_range.emplace_back(d[2]);
            points_range.emplace_back(d[3]);
            points_range.emplace_back(d[4]);
            points_range.emplace_back(d[5]);
        } else if (!strcmp(attr_name, "rv_grid_size")) {
            const int* d = static_cast<const int*>(fields[i].data);
            rv_grid_size.clear();
            rv_grid_size.emplace_back(d[0]);
            rv_grid_size.emplace_back(d[1]);
            rv_grid_size.emplace_back(d[2]);

        } else if (!strcmp(attr_name, "rv_voxel_size")) {
            const float* d = static_cast<const float*>(fields[i].data);
            rv_voxel_size.clear();
            rv_voxel_size.emplace_back(d[0]);
            rv_voxel_size.emplace_back(d[1]);
            rv_voxel_size.emplace_back(d[2]);

        } else if (!strcmp(attr_name, "rv_points_range")) {
            const float* d = static_cast<const float*>(fields[i].data);
            rv_points_range.clear();
            rv_points_range.emplace_back(d[0]);
            rv_points_range.emplace_back(d[1]);
            rv_points_range.emplace_back(d[2]);
            rv_points_range.emplace_back(d[3]);
            rv_points_range.emplace_back(d[4]);
            rv_points_range.emplace_back(d[5]);
        }
    }
    IPluginV2* plugin = new MultiViewVoxelGeneratorPlugin(grid_size, voxel_size, points_range, rv_grid_size,
                                                          rv_voxel_size, rv_points_range);
    return plugin;
}

IPluginV2* MultiViewVoxelGeneratorPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                                   size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed,
    IPluginV2* plugin = new MultiViewVoxelGeneratorPlugin(serialData, serialLength);
    return plugin;
}

void MultiViewVoxelGeneratorPluginCreator::setPluginNamespace(const char* libNamespace) noexcept {
    mNamespace = libNamespace;
}

const char* MultiViewVoxelGeneratorPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
