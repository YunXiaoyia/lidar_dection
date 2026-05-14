//
// Created by jie.gong on 22-07-14.
//

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include "../secure_matrix.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


class GreedyMatcher {

public:
    GreedyMatcher() = default;
    ~GreedyMatcher() = default;

    void Match(const float& cost_thresh, const float& bound_value,
               SecureMat<float>& cost_mat,
                std::vector<std::pair<size_t, size_t>>* assignments,
                std::vector<size_t>* unassigned_rows,
                std::vector<size_t>* unassigned_cols);

private:
    struct CostElement {
        float cost;
        size_t row_idx;
        size_t col_idx;

        CostElement(float a, size_t b, size_t c) :
            cost(a), row_idx(b), col_idx(c) {}
    };
    std::vector<CostElement> cost_elements_;

};  // class GreedyMatcher



}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

