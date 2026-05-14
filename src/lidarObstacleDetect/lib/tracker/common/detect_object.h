
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <Eigen/src/Core/Matrix.h>
#include "common/base.h"
#include "common/point_cloud.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {



class DetectObject {
public:
    DetectObject(common::ObjectPtr& object);
    DetectObject& operator=(const DetectObject& rhs) = default;

private:
    // Object copy
    common::ObjectPtr object_ptr;

    // Attribution
    std::shared_ptr<Eigen::Vector3f> bary_point_ = nullptr;
    std::shared_ptr<Eigen::Vector3f> cloest_point_ = nullptr;

    void CalBaryPoint(Eigen::Vector3f& point_out);
    void CalClosetPoint(Eigen::Vector3f& point_out);

public:
    const common::PointCloudPtr& points() const { return object_ptr->points; };
    const int detect_id() const { return object_ptr->detect_id; };

    const Eigen::Vector3f& bary_point();
    const Eigen::Vector3f& cloest_point();
    
};

/// ////////////////////////////
/// get functions
/// ////////////////////////////


inline const Eigen::Vector3f& DetectObject::bary_point()
{
    if (bary_point_ == nullptr) 
    {
        bary_point_ = std::make_shared<Eigen::Vector3f>();

        CalBaryPoint(*bary_point_);
    }

    return *bary_point_;
}

inline const Eigen::Vector3f& DetectObject::cloest_point()
{
    if (cloest_point_ == nullptr) 
    {
        cloest_point_ = std::make_shared<Eigen::Vector3f>();

        CalClosetPoint(*cloest_point_);
    }

    return *cloest_point_;
}

typedef std::shared_ptr<DetectObject> DetectObjectPtr;
typedef std::shared_ptr<const DetectObject> DetectObjectConstPtr;

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

