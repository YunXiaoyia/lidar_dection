#include "dsvt_input_layer.h"
#include <math.h>
#include <cmath>
#include <cstring>
#include <set>
#include "common/json_params_get.hpp"
#include "common/deepways_time.h"

// #include "npy.hpp"

namespace lidar_net {

using namespace base;

bool DSVTInputLayer::Init(const nlohmann::json& param_node){

    if (!this->getParams(param_node)) {
        PLOG_FATAL << "[Detector] init DSVTInputLayer Params Unsuccessfully";
        return false;
    }

    increasing_index_.resize(max_voxel_input);
    for(int i = 0; i < max_voxel_input; ++i){
        increasing_index_(i) = i;
    }


    return true;
}

bool DSVTInputLayer::getParams(const nlohmann::json& param_node){

    try{
        //
        GET_VALUE(param_node, stage_num);
        GET_VALUE(param_node, num_shifts);
        GET_VALUE(param_node, max_voxel_input);
        GET_VALUE(param_node, max_win_num);
        GET_VALUE(param_node, flag_multi_thread);

        // sparse_shape_
        {
            std::vector<int> sparse_shape;
            GET_VALUE(param_node, sparse_shape);
            sparse_shape_ << sparse_shape[0], sparse_shape[1], sparse_shape[2];
        }
        // windows_shape_
        {
            std::vector<std::vector<int>> windows_shape;
            GET_VALUE(param_node, windows_shape);
            Eigen::Vector3i win;
            for(const auto& w : windows_shape){
                win << w[0], w[1], w[2];
                windows_shape_.push_back(win);            }
        }
        //set info
        {
            std::vector<int> set_info;
            GET_VALUE(param_node, set_info);
            set_info_ << set_info[0], set_info[1];

        }
        // hybrid_factor_
        {
            std::vector<int> hybrid_factor;
            GET_VALUE(param_node, hybrid_factor);
            hybrid_factor_ << hybrid_factor[0], hybrid_factor[1], hybrid_factor[2];
        }
        // shift_list
        {
            std::vector<std::vector<int>> shift_list;
            GET_VALUE(param_node, shift_list);
            Eigen::Vector3i shift;
            for(const auto& s : shift_list){
                shift << s[0], s[1], s[2];
                shift_list_.push_back(shift);
            }
        }
        return true;
    }

    catch (const std::exception& e) {
            PLOG_FATAL << "[Detector] DSVTInputLayer Param JSON config failed with: " << e.what();
    }

    return false;
}


void DSVTInputLayer::doDSVTInput(const Eigen::MatrixXi& voxel_coors,
                                int* set_voxel_inds_tensor_shift_0,
                                int* set_voxel_inds_tensor_shift_1,
                                bool* set_voxel_masks_tensor_shift_0,
                                bool* set_voxel_masks_tensor_shift_1,
                                int* coors_in_win,
                                int& valid_set_num_0,
                                int& valid_set_num_1){
    int voxel_num = voxel_coors.rows();

    std::vector<Eigen::VectorXi> batch_win_inds;   // voxel 在所属的win的index

    batch_win_inds.resize(2);
    for(auto& b : batch_win_inds){
        b.resize(voxel_num);
        b.setZero();
    }
    PERF_BLOCK_START();
    this->windowPartition(voxel_coors, batch_win_inds, coors_in_win); // coors_in_win : [2, N, 3]
    // PERF_BLOCK_END("doDSVTInput - windowPartition");

    this->getSet(batch_win_inds, coors_in_win, voxel_num,
                set_voxel_inds_tensor_shift_0,
                set_voxel_inds_tensor_shift_1,
                set_voxel_masks_tensor_shift_0,
                set_voxel_masks_tensor_shift_1,
                valid_set_num_0,
                valid_set_num_1
                );
    // PERF_BLOCK_END("doDSVTInput - getSet");
}


void DSVTInputLayer::windowPartition(const Eigen::MatrixXi& voxel_coors, std::vector<Eigen::VectorXi>& batch_win_inds, int* coors_in_win){
    int voxel_num = voxel_coors.rows();
    for(int shift_id = 0; shift_id < 2; ++shift_id){
        // MatrixIMap coor_in_map_mat = decltype(coor_in_map_mat)(&coors_in_win(shift_id, 0, 0), voxel_num, coors_in_win_dims[2]);
        MatrixIMap coor_in_map_mat = decltype(coor_in_map_mat)(&coors_in_win[shift_id * voxel_num * 3], voxel_num, 3);
        getWindowCoors(voxel_coors, this->sparse_shape_, this->windows_shape_[shift_id],
        this->shift_list_[shift_id], shift_id==1, batch_win_inds[shift_id], coor_in_map_mat);
    }
}


/********


*********/
void DSVTInputLayer::getWindowCoors(const Eigen::MatrixXi& voxel_coors,
                        const Eigen::Vector3i& sparse_shape,
                        const Eigen::Vector3i& window_shape,  // wx, wy, wz
                        const Eigen::Vector3i& shift,
                        const bool& do_shift,
                        Eigen::VectorXi& batch_win_inds,
                        MatrixIMap& coors_in_win){
    // int max_num_win_x = int(std::ceil((sparse_shape[0] * 1.0 / window_shape[0])) + 1);
    int max_num_win_y = int(std::ceil((sparse_shape[1] * 1.0 / window_shape[1])) + 1);
    int max_num_win_z = int(std::ceil((sparse_shape[2] * 1.0 / window_shape[2])) + 1);
    // int max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z;

    const auto& shifted_coors_x = voxel_coors.col(3).array() + shift[0];
    const auto& shifted_coors_y = voxel_coors.col(2).array() + shift[1];
    const auto& shifted_coors_z = voxel_coors.col(1).array() + shift[2];

    Eigen::VectorXi win_coors_x = shifted_coors_x / window_shape[0];
    Eigen::VectorXi win_coors_y = shifted_coors_y / window_shape[1];
    Eigen::VectorXi win_coors_z = shifted_coors_z / window_shape[2];

    batch_win_inds =  // voxel_coors.col(0).array() * max_num_win_per_sample +
                     win_coors_x.array() * max_num_win_y * max_num_win_z +
                     win_coors_y.array() * max_num_win_z +
                     win_coors_z.array();

    coors_in_win.col(2) = shifted_coors_x.array() - win_coors_x.array() * window_shape[0];
    coors_in_win.col(1) = shifted_coors_y.array() - win_coors_y.array() * window_shape[1];
    coors_in_win.col(0) = shifted_coors_z.array() - win_coors_z.array() * window_shape[2];
}


/****************************************************************************
This is one of the core operation of DSVT.
Given voxels' window ids and relative-coords inner window, we partition them into window-bounded and size-equivalent local sets.
To make it clear and easy to follow, we do not use loop to process two shifts.
*****************************************************************************/
void DSVTInputLayer::getSet(
                const std::vector<Eigen::VectorXi>& batch_win_inds,
                int* const coors_in_win,
                const int& valid_voxel_num,
                int* set_voxel_inds_tensor_shift_0,
                int* set_voxel_inds_tensor_shift_1,
                bool* set_voxel_masks_tensor_shift_0,
                bool* set_voxel_masks_tensor_shift_1,
                int& valid_set_num_0,
                int& valid_set_num_1){

    const MatrixIMap  coors_in_win_shift_0 = decltype(coors_in_win_shift_0)(&coors_in_win[0*valid_voxel_num*3], valid_voxel_num, 3);
    const MatrixIMap  coors_in_win_shift_1 = decltype(coors_in_win_shift_1)(&coors_in_win[1*valid_voxel_num*3], valid_voxel_num, 3);

    if (flag_multi_thread){
        std::vector<std::future<void>> results;
        results.emplace_back(
            thread_pools_.enqueue([this, &coors_in_win_shift_0, &batch_win_inds, &set_voxel_inds_tensor_shift_0, &set_voxel_masks_tensor_shift_0, &valid_set_num_0]()
            {this->getSetSingleShift(batch_win_inds[0], coors_in_win_shift_0, 0, 0,
                set_voxel_inds_tensor_shift_0, set_voxel_masks_tensor_shift_0, valid_set_num_0);}));

        results.emplace_back(
            thread_pools_.enqueue([this, &coors_in_win_shift_1, &batch_win_inds, &set_voxel_inds_tensor_shift_1, &set_voxel_masks_tensor_shift_1, &valid_set_num_1]()
            {this->getSetSingleShift(batch_win_inds[0], coors_in_win_shift_1, 0, 1,
                set_voxel_inds_tensor_shift_1, set_voxel_masks_tensor_shift_1, valid_set_num_1);}));

        for (auto& result : results) {
            result.get();
        }
    }
    else{
        getSetSingleShift(batch_win_inds[0], coors_in_win_shift_0, 0, 0,
            set_voxel_inds_tensor_shift_0,  set_voxel_masks_tensor_shift_0, valid_set_num_0);
        getSetSingleShift(batch_win_inds[1], coors_in_win_shift_1, 0, 1,
        set_voxel_inds_tensor_shift_1,  set_voxel_masks_tensor_shift_1, valid_set_num_1);
    }

}


/**
 * @brief  torch.unique
 *
 * TODO, 耗时
 */
inline void unique(const Eigen::VectorXi& vec, Eigen::VectorXi& uniqueVec, Eigen::VectorXi& indices, Eigen::VectorXi& voxelnum_per_win) {
    int index_num = vec.size();

    // sorted_vec and sorted_idx
    Eigen::VectorXi sorted_idx = Eigen::VectorXi::Zero(index_num);
    Eigen::VectorXi sorted_vec = Eigen::VectorXi::Zero(index_num);
    argsort(vec.data(), index_num, sorted_idx.data());
    for(int i = 0; i < index_num; ++i){
        sorted_vec(i) = vec(sorted_idx[i]);
    }
    // unique_vec and unique_vec_count
    uniqueVec = sorted_vec;
    auto itr = std::unique(uniqueVec.data(), uniqueVec.data()+index_num);
    uniqueVec.conservativeResize(itr - uniqueVec.data());

    // count uniqueVec
    voxelnum_per_win.resize(uniqueVec.size());
    voxelnum_per_win.setZero();
    int idx_vpw = 0;
    int count = 1;
    for(int i = 1; i < sorted_vec.rows(); ++i){
        if(sorted_vec[i] == sorted_vec[i-1]){
            count++;
        }
        else{
            voxelnum_per_win[idx_vpw] = count;
            ++idx_vpw;
            count = 1;
        }
    }
    voxelnum_per_win[idx_vpw] = count;

    // sorted_sorted_idx
    Eigen::VectorXi sorted_sorted_idx = Eigen::VectorXi::Zero(index_num);
    argsort(sorted_idx.data(), index_num, sorted_sorted_idx.data());

    Eigen::VectorXi sorted_sorted_vec = Eigen::VectorXi::Zero(index_num);
    for(int i = 0; i < index_num; ++i){
        sorted_sorted_vec(i) = sorted_idx(sorted_sorted_idx[i]);
    }

    // contiguous_ids
    Eigen::VectorXi contiguous_id = Eigen::VectorXi::Zero(index_num);
    int conti_idx = 0;
    for(int i = 0; i < voxelnum_per_win.rows(); ++i){
        for(int j = 0; j < voxelnum_per_win(i); ++j){
            contiguous_id[conti_idx++] = i;
        }
    }

    indices.resize(index_num);
    indices.setZero();
    for(int i = 0; i < index_num; ++i){
        indices[i] = contiguous_id[sorted_sorted_idx[i]];
    }
}



inline void get_continue_inds(const Eigen::VectorXi& setnum_per_win, Eigen::VectorXi& set_win_inds, Eigen::VectorXi& set_inds_in_win){
    int win_num = setnum_per_win.rows();
    int set_num = setnum_per_win.sum();
    Eigen::VectorXi setnum_per_win_cumsum = Eigen::VectorXi::Zero(win_num);
    cumsum(setnum_per_win.data(), win_num, setnum_per_win_cumsum.data());

    set_win_inds.resize(set_num);
    set_win_inds.setZero();
    for(int i = 0; i < win_num-1; ++i){
        set_win_inds(setnum_per_win_cumsum(i)) = 1;
    }
    cumsum(set_win_inds.data(), set_num, set_win_inds.data());

    set_inds_in_win.resize(set_num);
    set_inds_in_win.setZero();
    int idx = 0;
    for(int i = 0; i < win_num; ++i){
        for(int j = 0; j < setnum_per_win[i]; ++j){
            set_inds_in_win(idx++) = j;
        }
    }
}

/****************
return unordered inner window indexs of each voxel
voxel在window内的
*************/
inline void ingroup_inds(const Eigen::VectorXi& contiguous_win_inds, Eigen::VectorXi& inner_voxel_inds){
    int max_win_inds = contiguous_win_inds.maxCoeff();
    Eigen::VectorXi counter = Eigen::VectorXi::Zero(max_win_inds+1);

    for(int i = 0; i < contiguous_win_inds.size(); ++i){
        inner_voxel_inds[i] = counter(contiguous_win_inds(i));
        counter(contiguous_win_inds(i)) += 1;
    }
}


/**********
对于win的划分，12x12，或 24x24
batch_win_inds : voxel 所属的 win的index
coors_in_win   : voxel 在所处win内的coors

get_continue_inds， 范围set与win相关的标号
set_win_inds: set说属的win编号
set_inds_in_win: set在win内的编号


************/
void DSVTInputLayer::getSetSingleShift(const Eigen::VectorXi& batch_win_inds,
                                       const MatrixIMap& coors_in_win,
                                       const int& stage_id,
                                       const int shift_id,
                                       int* set_voxel_inds_tensor,
                                       bool* set_voxel_masks,
                                       int& valid_set_num){


    // the number of voxels assigned to a set, 36
    int voxel_num = batch_win_inds.size();         // voxel_num  20938
    int voxel_num_set = set_info_[0];              // 36, 一个set中的voxel个数
    int max_voxel = windows_shape_[shift_id][0] * windows_shape_[shift_id][1] * windows_shape_[shift_id][2];  // 12x12;  24x24

    Eigen::VectorXi unique_inds;
    Eigen::VectorXi contiguous_win_inds;   // 连续的win的索引
    Eigen::VectorXi voxelnum_per_win;
    unique(batch_win_inds, unique_inds, contiguous_win_inds, voxelnum_per_win);

    Eigen::VectorXi set_win_inds;         // .size() = 936
    Eigen::VectorXi set_inds_in_win;      // .size() = 936
    Eigen::VectorXi setnum_per_win = (voxelnum_per_win.cast<float>().array() / static_cast<float>(voxel_num_set)).ceil().cast<int>();
    get_continue_inds(setnum_per_win, set_win_inds, set_inds_in_win);

    valid_set_num = set_win_inds.size();

    // return unordered inner window indexs of each voxel
    Eigen::VectorXi inner_voxel_inds = Eigen::VectorXi::Zero(voxel_num);
    ingroup_inds(contiguous_win_inds, inner_voxel_inds);  //

    Eigen::VectorXi global_voxel_inds = contiguous_win_inds.array() * max_voxel + inner_voxel_inds.array();
    Eigen::VectorXi order_global = Eigen::VectorXi::Zero(voxel_num);
    argsort(global_voxel_inds.data(),voxel_num, order_global.data());

    Eigen::VectorXi global_sorted_inner_voxel_inds = Eigen::VectorXi::Zero(voxel_num);
    for(int i = 0; i < voxel_num; ++i){
        global_sorted_inner_voxel_inds(i) = inner_voxel_inds(order_global(i));
    }

    // obtain unique indexs in whole space
    Eigen::MatrixXi base_idx = Eigen::MatrixXi::Zero(valid_set_num, voxel_num_set);  // N x 36
    for(int i = 0; i < voxel_num_set; ++i){
        // base_idx(i) = i;
        base_idx.col(i).array() = i;
    }

    Eigen::VectorXi voxel_num_scale = Eigen::VectorXi::Zero(valid_set_num);
    Eigen::VectorXf set_num_scale = Eigen::VectorXf::Zero(valid_set_num);

    for(int i = 0; i < valid_set_num; i++){
        voxel_num_scale(i) = voxelnum_per_win(set_win_inds(i));
        set_num_scale(i) = setnum_per_win(set_win_inds(i)) * 36.0;
    }
    Eigen::MatrixXi select_idx = (base_idx.colwise() + set_inds_in_win * voxel_num_set);   // 936 X 36
    select_idx = select_idx.array().colwise() * voxel_num_scale.array();
    select_idx = (select_idx.array().cast<float>().colwise() / set_num_scale.array()).floor().cast<int>();
    select_idx = select_idx.colwise() + set_win_inds * max_voxel;  // select_idx ok

    // y
    Eigen::VectorXi global_voxel_inds_sort_y = contiguous_win_inds.array() * max_voxel +
                                               coors_in_win.col(1).array() * windows_shape_[shift_id][0] * windows_shape_[shift_id][2] +
                                               coors_in_win.col(2).array() * windows_shape_[shift_id][2] +
                                               coors_in_win.col(0).array();


    Eigen::VectorXi order_sort_y = Eigen::VectorXi::Zero(global_voxel_inds_sort_y.rows());
    argsort(global_voxel_inds_sort_y.data(), global_voxel_inds_sort_y.rows(), order_sort_y.data());

    Eigen::VectorXi inner_voxel_inds_sorty = Eigen::VectorXi::Ones(voxel_num) * -1;
    // // inner_voxel_inds_sorty.scatter_(dim=0, index=order2, src=inner_voxel_inds[order1])
    for(int i = 0; i < voxel_num; i++){
        inner_voxel_inds_sorty(order_sort_y(i)) = global_sorted_inner_voxel_inds(i);
    }

    Eigen::VectorXi voxel_inds_padding_max_ = Eigen::VectorXi::Ones(max_win_num * 36) * -1;
    Eigen::VectorXi voxel_inds_in_batch_sorty = inner_voxel_inds_sorty + max_voxel * contiguous_win_inds;
    for(int i = 0; i < voxel_num; ++i){
        voxel_inds_padding_max_(voxel_inds_in_batch_sorty(i)) = i;
    }

    // output，先y
    MatrixIMap set_voxel_inds_sorty = decltype(set_voxel_inds_sorty)(&set_voxel_inds_tensor[0*valid_set_num * voxel_num_set], valid_set_num, voxel_num_set);
    for(int r = 0; r < select_idx.rows(); ++r){
        for(int c = 0; c < select_idx.cols(); ++c){
            set_voxel_inds_sorty(r, c) = voxel_inds_padding_max_(select_idx(r, c));
        }
    }

    // x
    Eigen::VectorXi global_voxel_inds_sort_x = contiguous_win_inds.array() * max_voxel +
                                               coors_in_win.col(2).array() * windows_shape_[shift_id][1] * windows_shape_[shift_id][2] +
                                               coors_in_win.col(1).array() * windows_shape_[shift_id][2] +
                                               coors_in_win.col(0).array();
    Eigen::VectorXi order_sort_x = Eigen::VectorXi::Zero(global_voxel_inds_sort_x.rows());
    argsort(global_voxel_inds_sort_x.data(), global_voxel_inds_sort_x.rows(), order_sort_x.data());
    Eigen::VectorXi inner_voxel_inds_sortx = Eigen::VectorXi::Ones(voxel_num) * -1;
    // // inner_voxel_inds_sorty.scatter_(dim=0, index=order2, src=inner_voxel_inds[order1])
    for(int i = 0; i < voxel_num; i++){
        inner_voxel_inds_sortx(order_sort_x(i)) = global_sorted_inner_voxel_inds(i);
    }
    Eigen::VectorXi voxel_inds_in_batch_sortx = inner_voxel_inds_sortx + max_voxel * contiguous_win_inds;
    for(int i = 0; i < voxel_num; ++i){
        voxel_inds_padding_max_(voxel_inds_in_batch_sortx(i)) = i;
    }

    // output，后x
    MatrixIMap set_voxel_inds_sortx = decltype(set_voxel_inds_sortx)(&set_voxel_inds_tensor[1*valid_set_num * voxel_num_set], valid_set_num, voxel_num_set);
    for(int r = 0; r < select_idx.rows(); ++r){
        for(int c = 0; c < select_idx.cols(); ++c){
            set_voxel_inds_sortx(r, c) = voxel_inds_padding_max_(select_idx(r, c));
        }
    }

    MatrixBMap mask_y = decltype(mask_y)(&set_voxel_masks[0*valid_set_num*36], valid_set_num, 36);
    MatrixBMap mask_x = decltype(mask_x)(&set_voxel_masks[1*valid_set_num*36], valid_set_num, 36);

    getKeyMask(set_voxel_inds_sorty, mask_y);
    getKeyMask(set_voxel_inds_sortx, mask_x);

}

void DSVTInputLayer::getKeyMask(const MatrixIMap& set_voxel_inds, MatrixBMap& mask){
    int cols = set_voxel_inds.cols();
    Eigen::MatrixXi prefix_inds = Eigen::MatrixXi(set_voxel_inds.rows(), set_voxel_inds.cols());
    prefix_inds.rightCols(cols-1) = set_voxel_inds.leftCols(cols-1);
    prefix_inds.col(0).array() = -1;
    mask = (prefix_inds.array() == set_voxel_inds.array());

}

}