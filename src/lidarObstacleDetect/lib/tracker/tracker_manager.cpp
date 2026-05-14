
#include "tracker_manager.h"

#include "simple_track/simple_track.h"
#include "l_shape_tracker/l_shape_track.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

BaseTargetTrackPtr TargetTrackManager::create(const TrackMethod& method) {
    BaseTargetTrackPtr target_track_ptr = nullptr;
    if (method == TrackMethod::SIMPLE_TRACK) {
        target_track_ptr = std::make_shared<SimpleTrack>();
    } else if (method == TrackMethod::L_SHAPE) {
        target_track_ptr = std::make_shared<LShapeTrack>();
    } else {
        target_track_ptr = nullptr;
    }
    return target_track_ptr;
}

} // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
