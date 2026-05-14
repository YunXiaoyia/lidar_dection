//
// Created by jie.gong on 22-03-17.
//
#pragma once

#include <deque>
#include <memory>
#include <iostream>

#include "toml.hpp"

// 临时定义 TLOG_WARN 宏
#ifndef TLOG_WARN
#define TLOG_WARN std::cout << "[WARN] "
#endif

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

class IDManager {
private:
    size_t MAX_ID_NUMBER = 128;
    size_t START_NUM = 1;
public:
    // Init class, cached id number
    IDManager() {
        id_pool_.clear();
        for (size_t i = START_NUM; i < MAX_ID_NUMBER; ++i) {
            id_pool_.push_back(i);
        }
    }

    bool Init(const toml::node_view<const toml::node>& param_node) {

        auto st_num = param_node.at_path("start_id").value<int>();
        if (!st_num) {
            TLOG_WARN << "[IDManager] Don't find param: start_id, use default value:" << START_NUM;
        } else {
            START_NUM = st_num.value();
        }

        auto max_id = param_node.at_path("max_id").value<int>();
        if (!max_id) {
            TLOG_WARN << "[IDManager] Don't find param: max_id, use default value:" << MAX_ID_NUMBER;
        } else {
            MAX_ID_NUMBER = max_id.value();
        }

        id_pool_.clear();
        for (size_t i = START_NUM; i < MAX_ID_NUMBER; ++i) {
            id_pool_.push_back(i);
        }

        return true;
    }

    // Return usable id
    size_t get_id() {

        if (id_pool_.empty()) { return MAX_ID_NUMBER;}

        size_t id = id_pool_.front();
        id_pool_.pop_front();

        return id;
    }

    // Recycling id
    void id_recycled(const size_t& id) {

        if (id == 0) { return; }

        id_pool_.push_back(id);
    }

private:
    std::deque<size_t> id_pool_;

};

using IDManagerPtr = std::shared_ptr<IDManager>;

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
