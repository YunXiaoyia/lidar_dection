

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "associater/associater.h"
#include "associater/two_stage_associater.h"
#include "associater/jpda/jpda_associater.h"
#include "common/lidar_perception_log.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


enum class AssociateType : std::uint8_t {
    TWO_STAGE,
    JPDA,
    UNKNOWN
};

static std::unordered_map<std::string, AssociateType> associate_map{
    {"two_stage", AssociateType::TWO_STAGE},
    {"jpda", AssociateType::JPDA}
};

AssociaterPtr AssociaterFactory::MakeAssociater(const toml::node_view<const toml::node>& param_node) {

    std::string type = "two_stage";
    if (!param_node["associate_type"].is_value()) {
        TLOG_WARN << "[Associate] Associate handle init error! please check param: associate_type"
                     << " use default: two_stage.";
    } else {
        type = param_node["associate_type"].value<std::string>().value();
    }
    auto ass_type = associate_map[type];
    if (ass_type == AssociateType::TWO_STAGE) {
        auto ass_obj = std::make_shared<TwoStageAssociater>();
        ass_obj->Init(param_node);
        return ass_obj;
    } else if (ass_type == AssociateType::JPDA) {
        auto ass_obj = std::make_shared<JPDAAssociater>();
        ass_obj->Init(param_node);
        return ass_obj;
    } else {
        auto ass_obj = std::make_shared<TwoStageAssociater>();
        ass_obj->Init(param_node);
        return ass_obj;
    }
}


}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
