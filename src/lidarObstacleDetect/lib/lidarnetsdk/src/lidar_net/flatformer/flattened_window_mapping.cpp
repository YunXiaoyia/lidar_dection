#include "flattened_window_mapping.h"
#include <math.h>
#include <cmath>
#include <cstring>
#include <set>
#include "common/json_params_get.hpp"
#include "common/deepways_time.h"


namespace lidar_net {
using namespace base;

bool FlattenedWindowMapping::Init(const nlohmann::json &param_node){
    if(!this->getParams(param_node)){
        PLOG_FATAL << "[Detector] init FlattenedWindowMapping Params Unsuccessfully";
        return false;
    }

    return true;
}

bool FlattenedWindowMapping::getParams(const nlohmann::json &param_node){
    try{
        GET_VALUE(param_node, num_shifts);
        GET_VALUE(param_node, max_voxel_input);
        GET_VALUE(param_node, max_group_num);
        GET_VALUE(param_node, group_size);
        GET_VALUE(param_node, flag_multi_thread);

        // sparse_shape_
        {
            std::vector<int> sparse_shape;
            GET_VALUE(param_node, sparse_shape);
            sparse_shape_ << sparse_shape[0], sparse_shape[1], sparse_shape[2];
        }
        // windows_shape_
        {
            std::vector<int> windows_shape;
            GET_VALUE(param_node, windows_shape);
            windows_shape_ << windows_shape[0], windows_shape[1], windows_shape[2];
        }

        windows_num_ = ceil(sparse_shape_.cast<float>().array() /  windows_shape_.cast<float>().array()).cast<int>() + 1;
        coor_shift_ = floor(windows_shape_.cast<float>().array() / 2).cast<int>();
        windows_shape_2_ = windows_shape_.array() * 2;
        windows_num_2_ = windows_num_.array() * 2;
    }

   catch (const std::exception& e) {
            PLOG_FATAL << "[Detector] FlattenedWindowMapping Param JSON config failed with: " << e.what();
    }
    return true;
}


void FlattenedWindowMapping::doMapping(const Eigen::MatrixXi& voxel_coors,
                     int* flat2win,
                     int* win2flat,
                     int* mapping_x,
                     int* mapping_y,
                     int* mapping_x_shift,
                     int* mapping_y_shift,
                     int& valid_group_num
                    ){

    int voxel_num = voxel_coors.rows();
    int padded_voxel_num = ((voxel_num + group_size - 1) / group_size) * group_size;
    valid_group_num = padded_voxel_num / group_size;

    MatrixIMap flat2win_map = decltype(flat2win_map)(&flat2win[0], valid_group_num, group_size);
    VectorIMap win2flat_map = decltype(win2flat_map)(&win2flat[0], voxel_num);
    VectorIMap mapping_x_map = decltype(mapping_x_map)(&mapping_x[0], voxel_num);
    VectorIMap mapping_y_map = decltype(mapping_y_map)(&mapping_y[0], voxel_num);
    VectorIMap mapping_x_shift_map = decltype(mapping_x_shift_map)(&mapping_x_shift[0], voxel_num);
    VectorIMap mapping_y_shift_map = decltype(mapping_y_shift_map)(&mapping_y_shift[0], voxel_num);

    if(flag_multi_thread){
        std::vector<std::future<void>> results;
        results.emplace_back(
            thread_pools_.enqueue([this, &voxel_num, &padded_voxel_num, &flat2win_map, &win2flat_map]()
            {this->getWindFlatMapping(voxel_num, padded_voxel_num, flat2win_map, win2flat_map);}));

        results.emplace_back(
            thread_pools_.enqueue([this, &voxel_coors, &mapping_x_map, &mapping_y_map]()
            {this->getWindowCoorsShift(voxel_coors, false, mapping_x_map, mapping_y_map);}));
        results.emplace_back(
            thread_pools_.enqueue([this, &voxel_coors, &mapping_x_shift_map, &mapping_y_shift_map]()
            {this->getWindowCoorsShift(voxel_coors, true, mapping_x_shift_map, mapping_y_shift_map);}));

        for (auto& result : results) {
            result.get();
        }
    }
    else{
        // flat2win, win2flat
        getWindFlatMapping(voxel_num, padded_voxel_num, flat2win_map, win2flat_map);
        // mapping_x, mapping_y, mapping_x_shift, mapping_y_shift
        getWindowCoorsShift(voxel_coors, false, mapping_x_map, mapping_y_map);
        getWindowCoorsShift(voxel_coors, true, mapping_x_shift_map, mapping_y_shift_map);
    }
}


void FlattenedWindowMapping::getWindFlatMapping(const int& voxel_num, const int& padded_voxel_num, MatrixIMap& flat2win_map, VectorIMap& win2flat_map){
 // falt2win & win2flat
    win2flat_map = Eigen::VectorXi::LinSpaced(voxel_num, 0, voxel_num);
    Eigen::VectorXi f2w = Eigen::VectorXi::LinSpaced(padded_voxel_num, 0, padded_voxel_num);
    if(voxel_num != padded_voxel_num){
        int pad_num = padded_voxel_num - voxel_num;
        f2w.tail(pad_num) = f2w.segment(voxel_num-group_size, pad_num);
    }

    for(int i = 0; i < padded_voxel_num; ++i){
        int idx0 = i / group_size;
        int idx1 = i % group_size;
        flat2win_map(idx0, idx1) = f2w[i];
    }
}

// 优化，不用for，只是用eigen会快吗
inline Eigen::VectorXi arrayParity(Eigen::VectorXi& array){
    int array_num = array.size();
    std::vector<int> array_std(array.data(), array.data() + array_num);

    Eigen::VectorXi result(array_num);
    std::transform(array_std.begin(), array_std.end(), result.data(), [](int num) {
        return num % 2 == 0 ? 1 : -1;
    });
    return result;
}

void FlattenedWindowMapping::getWindowCoorsShift(const Eigen::MatrixXi& coords,
                        const bool& shifted, VectorIMap& mapping_x, VectorIMap& mapping_y){

    Eigen::VectorXi coord_x = coords.col(3);
    Eigen::VectorXi coord_y = coords.col(2);

    if(shifted){
        coord_x.array() += coor_shift_[0];
        coord_y.array() += coor_shift_[1];
    }

    // 1
    Eigen::VectorXi x_win_idx = (coord_x.array() / windows_shape_[0]);
    Eigen::VectorXi y_win_idx = (coord_y.array() / windows_shape_[1]);
    // 2
    Eigen::VectorXi x_win_shift = coord_x - (x_win_idx * windows_shape_[0]);
    Eigen::VectorXi y_win_shift = coord_y - (y_win_idx * windows_shape_[1]);

    // Eigen::VectorXi ff = x_win_shift.array();
    Eigen::VectorXi parity_win_idx_x = arrayParity(x_win_idx);
    Eigen::VectorXi parity_win_idx_y = arrayParity(y_win_idx);
    Eigen::VectorXi parity_win_shift_x = arrayParity(x_win_shift);
    Eigen::VectorXi parity_win_shift_y = arrayParity(y_win_shift);

    mapping_x = (windows_num_2_[0] * y_win_idx.array() + parity_win_idx_y.array() * x_win_idx.array())
                 * windows_shape_2_[0] * windows_shape_2_[1] + parity_win_idx_y.array() *
                 (windows_shape_2_[1] * x_win_shift.array() + parity_win_shift_x.array() * y_win_shift.array());

    mapping_y = (windows_num_2_[1] * x_win_idx.array() + parity_win_idx_x.array() * y_win_idx.array())
                 * windows_shape_2_[0] * windows_shape_2_[1] + parity_win_idx_x.array() *
                 (windows_shape_2_[0] * y_win_shift.array() + parity_win_shift_y.array() * x_win_shift.array());

    argsort(mapping_x.data(), mapping_x.size(), mapping_x.data());
    argsort(mapping_y.data(), mapping_y.size(), mapping_y.data());
}

}

/*************************
python code
*************************

def get_window_coors_shift(coords, sparse_shape, window_shape, shifted):
    n, m, _ = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    if shifted:
        # shift_x, shift_y = (n2 // 2, m2 // 2)
        shift_x, shift_y = (torch.div(n2, 2, rounding_mode='floor'), torch.div(m2, 2, rounding_mode='floor'))
        x = coords[:, 3] + shift_x
        y = coords[:, 2] + shift_y
    else:
        x = coords[:, 3]
        y = coords[:, 2]

    # x1 = x // n2
    # y1 = y // m2
    x1 = torch.div(x, n2, rounding_mode='floor')
    y1 = torch.div(y, m2, rounding_mode='floor')
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2

class FlattenedWindowMapping(nn.Module):
    def __init__(
        self,
        window_shape,
        sparse_shape,
        group_size,
    ) -> None:
        super().__init__()
        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.group_size = group_size

    def forward(self, coords: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        coords = coords.long()

        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True) # 4792
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
            torch.div(
                batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1, # 4935
                self.group_size,
                rounding_mode="trunc",
            )
            * self.group_size
        )
        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))            # [0, 4896]
        flat2win = torch.arange(batch_start_indices_p[-1]).to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1]).to(coords.device)
        for i in range(batch_size):
            win2flat[batch_start_indices[i] : batch_start_indices[i + 1]] += (
                batch_start_indices_p[i] - batch_start_indices[i]
            )
            if num_per_batch[i] != num_per_batch_p[i]:
                flat2win[
                    batch_start_indices_p[i + 1]
                    - self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                ] = flat2win[
                    batch_start_indices_p[i + 1]
                    - 2 * self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                    - self.group_size
                ]
            flat2win[batch_start_indices_p[i] : batch_start_indices_p[i + 1]] -= (
                batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}
        for shifted in [False, True]:
            (
                n2,
                m2,
                n1,
                m1,
                x1,
                y1,
                x2,
                y2,
            ) = get_window_coors_shift(coords, self.sparse_shape, self.window_shape, shifted=shifted)
            vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
            vx += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
            vy += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
            _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        return mappings
********/
