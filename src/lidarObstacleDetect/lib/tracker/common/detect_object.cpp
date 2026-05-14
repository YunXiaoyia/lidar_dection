
#include "common/detect_object.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {



void DetectObject::CalBaryPoint(Eigen::Vector3f& point_out) {
    auto points_eigen = object_ptr->points->getMatrixXfMap(3, 8, 0);
    point_out = points_eigen.rowwise().mean();
}

void DetectObject::CalClosetPoint(Eigen::Vector3f& res) {

}



}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

