#pragma once
#include "../../common/base.h"
// 使用 SDK 中的 toml 解析器
#include "toml.hpp" 

HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

class BaseTargetTrack {
public:
    virtual ~BaseTargetTrack() = default;
    virtual bool Init(const toml::node_view<const toml::node>& param_node) = 0;
    virtual bool Track(common::LidarFramePtr& frame_data) = 0;
};

using BaseTargetTrackPtr = std::shared_ptr<BaseTargetTrack>;

}
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
