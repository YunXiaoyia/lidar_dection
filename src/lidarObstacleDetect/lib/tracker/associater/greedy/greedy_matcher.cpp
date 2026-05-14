
#include "greedy_matcher.h"
#include <algorithm>
#include <vector>


#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


void GreedyMatcher::Match(const float& cost_thresh, const float& bound_value,
                            SecureMat<float>& cost_mat,
                            std::vector<std::pair<size_t, size_t>>* assignments,
                            std::vector<size_t>* unassigned_rows,
                            std::vector<size_t>* unassigned_cols) {

    
    cost_elements_.clear();
    for (size_t i = 0; i < cost_mat.height(); ++i) {
        for (size_t j = 0; j < cost_mat.width(); ++j) {
            if(cost_mat(i, j) > cost_thresh) {
                continue;
            }

            cost_elements_.emplace_back(cost_mat(i, j), i, j);
        }
    }

    std::sort(cost_elements_.begin(), cost_elements_.end(), 
                [](CostElement& a, CostElement& b) {
                    return a.cost < b.cost;
                });
    
    std::vector<bool> row_checked(cost_mat.height(), false);
    std::vector<bool> col_checked(cost_mat.width(), false);

    for (const auto& ele : cost_elements_) {
        if (row_checked[ele.row_idx]) {
            continue;
        }
        
        if (col_checked[ele.col_idx]) {
            continue;
        }

        assignments->emplace_back(ele.row_idx, ele.col_idx);
        row_checked[ele.row_idx] = true;
        col_checked[ele.col_idx] = true;
    }

    for (size_t i = 0; i < row_checked.size(); ++i) {
        if (row_checked[i]) {
            continue;
        }
        unassigned_rows->emplace_back(i);
    }

    for (size_t i = 0; i < col_checked.size(); ++i) {
        if (col_checked[i]) {
            continue;
        }
        unassigned_cols->emplace_back(i);
    }
}


}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
