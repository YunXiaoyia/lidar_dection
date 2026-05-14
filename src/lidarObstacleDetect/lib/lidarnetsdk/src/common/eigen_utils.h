#pragma once

#include <functional>
#include <map>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace lidar_net {
namespace base {

// using TensorMap = Eigen::TensorMap<Eigen::Tensor<float, 5, Eigen::RowMajor>>;  // waymo, nx5
// using MatrixMap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
// using MatrixIMap = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;

/**
 * @brief inplace eigen softmax
 *
 * @param input
 */
inline void eigen_softmax_(Eigen::Ref<Eigen::VectorXf> input) {
    input = input.array().exp();
    input /= input.sum();
}

/**
 * @brief eigen softmax
 *
 * @param input
 */
inline Eigen::VectorXf eigen_softmax(const Eigen::Ref<const Eigen::VectorXf>& input) {
    Eigen::VectorXf output = input.array().exp();
    output /= output.sum();
    return output;
}

/**
 * @brief return the argmax index of input eigen vector
 *
 * @param input
 * @return size_t
 */
inline size_t eigen_argmax(const Eigen::Ref<const Eigen::VectorXf>& input) {
    size_t argmax = 0;
    input.maxCoeff(&argmax);
    return argmax;
}

/**
 * @brief write eigen matrix or matrix to binary file
 *
 * @tparam T
 * @param filename
 * @param matrix
 */
template <class T>
void write_binary(const char* filename, const T& matrix) {
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    out.write((char*)matrix.data(), matrix.size() * sizeof(typename T::Scalar));
    out.close();
}

/**
 * @brief return if eigen vector has been sorted ascendingly
 *
 * @param p
 * @return true
 * @return false
 */
inline bool if_sorted_ascend(const Eigen::VectorXf& p) {
    for (int i = 0; i < p.size() - 1; ++i) {
        if (p[i] > p[i + 1]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief return if eigen vector has been sorted descendingly
 *
 * @param p
 * @return true
 * @return false
 */
inline bool if_sorted_descend(const Eigen::VectorXf& p) {
    for (int i = 0; i < p.size() - 1; ++i) {
        if (p[i] < p[i + 1]) {
            return false;
        }
    }
    return true;
}



template<class T1, class T2>
inline void cumsum(T1* array, const int& num, T2* csum){
    std::partial_sum(array, array+num, csum);
}


template<class T>
void argsort(T* array, const int& num, int* index){
	for (int i = 0; i < num; ++i){
        index[i] = i;
    }
    const auto function = [array](int pos1, int pos2) {return (array[pos1] < array[pos2]);};
	std::stable_sort(index, index + num, function);
}

}  // namespace base
}  // namespace lidar_net
