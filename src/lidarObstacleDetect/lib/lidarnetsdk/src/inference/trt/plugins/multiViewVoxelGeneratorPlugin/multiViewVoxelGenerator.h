#ifndef LIB_INFERENCE_TRT_PLUGINS_MULTI_VIEW_VXOEL_GENERATOR_PLUGIN
#define LIB_INFERENCE_TRT_PLUGINS_MULTI_VIEW_VXOEL_GENERATOR_PLUGIN

#include <cublas_v2.h>
#include <string>
#include <vector>
#include "NvInferPlugin.h"
#include "plugin.h"
#include "kernel.h"

namespace nvinfer1 {
namespace plugin {

class MultiViewVoxelGeneratorPlugin : public IPluginV2DynamicExt {
   public:
    MultiViewVoxelGeneratorPlugin() = delete;
    MultiViewVoxelGeneratorPlugin(const void* data, size_t length);
    MultiViewVoxelGeneratorPlugin(std::vector<size_t> grid_size, std::vector<float> voxel_size,
                                  std::vector<float> points_range, std::vector<size_t> rv_grid_size,
                                  std::vector<float> rv_voxel_size, std::vector<float> rv_points_range);
    ~MultiViewVoxelGeneratorPlugin() override = default;

    /*********************************************
     *  @brief 插件op返回多少个Tensor，如果操作只输出一个Tensor，直接return 1
     * *******************************************/
    int getNbOutputs() const noexcept override;

    /*********************************************
     *  @brief TensorRT支持Dynamic-shape的时候，batch这一维度必须是explicit的，
     *         也就是说，TensorRT处理的维度从以往的三维[3,-1,-1]变成了[1,3,-1,-1]
     * *******************************************/
    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override;

    /*********************************************
     *  @brief TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
     * *******************************************/
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;

    /*********************************************
     *  @brief 配置这个插件op，判断输入和输出类型数量是否正确。
               官方还提到通过这个配置信息可以告知TensorRT去选择合适的算法(algorithm)去调优这个模型。
     * *******************************************/
    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out,
                         int32_t nbOutputs) noexcept override;

    /*********************************************
     *  @brief 这个函数需要返回这个插件op需要中间显存变量的实际数据大小(bytesize)，
               这个是通过TensorRT的接口去获取，是比较规范的方式
     * *******************************************/
    virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,
                                    int32_t nbOutputs) const noexcept override;
    /*********************************************
     *  @brief 初始化函数，在这个插件准备开始run之前执行。
     *         主要初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数，假如我们的算子需要这些参数，则在这里需要提前开辟显存。
     * *******************************************/
    int initialize() noexcept override;

    void terminate() noexcept override;

    /*********************************************
     *  @brief 实际插件op的执行函数，
     * *******************************************/
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    /*********************************************
     *  @brief 返回序列化时需要写多少字节到buffer中。
     * *******************************************/
    size_t getSerializationSize() const noexcept override;


    /*********************************************
     *  @brief 把需要用的数据按照顺序序列化到buffer里头。
     * *******************************************/
    void serialize(void* buffer) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2DynamicExt* clone() const noexcept override;

    /*********************************************
     *  @brief 为这个插件设置namespace名字，如果不设置则默认是""，
     *         需要注意的是同一个namespace下的plugin如果名字相同会冲突。
     * *******************************************/
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    /*********************************************
     *  @brief 返回结果的类型，一般来说我们插件op返回结果类型与输入类型一致
     * *******************************************/
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    /*********************************************
     *  @brief 如果这个op使用到了一些其他东西，例如cublas handle，可以直接借助TensorRT内部提供的cublas handle:
     * *******************************************/
    void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
                         IGpuAllocator* gpuAllocator) noexcept override;

    void detachFromContext() noexcept override;

   private:
    std::string mPluginNamespace;

    std::vector<size_t> grid_size_;
    std::vector<float> voxel_size_;
    std::vector<float> points_range_;

    std::vector<size_t> rv_grid_size_;
    std::vector<float> rv_voxel_size_;
    std::vector<float> rv_points_range_;
};


class MultiViewVoxelGeneratorPluginCreator : public nvinfer1::pluginInternal::BaseCreator {
   public:
    MultiViewVoxelGeneratorPluginCreator();

    ~MultiViewVoxelGeneratorPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    /*********************************************
     *  @brief 这个成员函数作用是通过PluginFieldCollection去创建plugin，
     *         将op需要的权重和参数一个一个取出来，然后调用上面的构造函数
     * *******************************************/
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    /*********************************************
     *  @brief 这个函数会被onnx-tensorrt的一个叫做TRT_PluginV2的转换op调用，
     *         这个op会读取onnx模型的data数据将其反序列化到network中。
     * *******************************************/
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

   private:
    static PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};
}  // namespace plugin
}  // namespace nvinfer1


#endif