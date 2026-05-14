
#include <cmath>

#include "motion_filter/l_shape/lshape_motion_filter.h"
#include "motion_filter/kalman_model/extended_kalman_cv_model.hpp"
#include "motion_filter/kalman_model/extended_kalman_ca_model.hpp"
#include "motion_filter/kalman_model/normal_kalman_cv_model.hpp"
#include "lib/utils/kalman/SimpleKalmanFilter.hpp"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

static const double switchCornerAngle = 1.0;

// Normalizes the angle to be 0 to 2*M_PI.
static double normalize_angle_positive(double angle) {
    return std::fmod(std::fmod(angle, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
}

// Normalizes the angle to be -M_PI circle to +M_PI circle
static double normalize_angle(double angle) {
    double a = normalize_angle_positive(angle);
    if (a > M_PI) a -= 2.0 * M_PI;
    return a;
}

// Given 2 angles, this returns the shortest angular difference.
// The result would always be -pi <= result <= pi.
static double shortest_angular_distance(double from, double to) {
    return normalize_angle(to - from);
}

///////////////////////
/// L-shape Function
///////////////////////

bool LShapeMotionFilter::InitStaticParams(const toml::node_view<const toml::node>& param_node) {

    return true;
}

double LShapeMotionFilter::findTurn(const double& new_angle, const double& old_angle) {
    // https://math.stackexchange.com/questions/1366869/calculating-rotation-direction-between-two-angles
    double theta_pro = new_angle - old_angle;
    double turn = 0;
    if (-M_PI <= theta_pro && theta_pro <= M_PI) {
        turn = theta_pro;
    } else if (theta_pro > M_PI) {
        turn = theta_pro - 2 * M_PI;
    } else if (theta_pro < -M_PI) {
        turn = theta_pro + 2 * M_PI;
    }
    return turn;
}

int LShapeMotionFilter::detectCornerPointSwitch(const double& from, const double& to) {
    // Corner Point Switch Detection
    double turn = findTurn(from, to);
    if (turn < -1.0 * switchCornerAngle) {
        this->CounterClockwisePointSwitch();
        return -1;
    } else if (turn > switchCornerAngle) {
        this->ClockwisePointSwitch();
        return 1;
    }

    return 0;
}

/// Switch different model
void LShapeMotionFilter::ClockwisePointSwitch() {
    Kalman::shape::State shape_state = shape_filter_->getState();

    double L1 = shape_state(0);  // l1
    double L2 = shape_state(1);  // l2
    double theta = shape_state(2);  // theta

    // L1 和 L2 交换
    shape_state(0) = L2;
    shape_state(1) = L1;
    shape_state(2) = theta - M_PI_2;
    shape_filter_->init(shape_state);
}

/// Switch different model
void LShapeMotionFilter::CounterClockwisePointSwitch() {
    Kalman::shape::State shape_state = shape_filter_->getState();

    double L1 = shape_state(0);  // l1
    double L2 = shape_state(1);  // l2
    double theta = shape_state(2);  // theta

    // L1 和 L2 交换
    shape_state(0) = L2;
    shape_state(1) = L1;
    shape_state(2) = theta + M_PI_2;
    shape_filter_->init(shape_state);
}

///////////////////////
/// Class Function
///////////////////////

bool LShapeMotionFilter::Init(const TrackFilterInitOptions& options)
{  
    printf("[DEBUG] LShapeMotionFilter::Init started\n");
    auto& lshape_info = options.tracked_object->object_ptr->lshape_box;

    printf("[DEBUG] Creating motion filter\n");
    motion_filter_.reset(new NormalKalmanCV);
    printf("[DEBUG] Motion filter created at %p\n", motion_filter_.get());
    
    printf("[DEBUG] Creating shape filter\n");
    shape_filter_.reset(new Kalman::SimpleShapeKalmanFilter);
    printf("[DEBUG] Shape filter created at %p\n", shape_filter_.get());
    
    printf("[DEBUG] Testing virtual function call on motion_filter_\n");
    // Test if the virtual function table is properly set up
    try {
        motion_filter_->getState();
        printf("[DEBUG] getState() call succeeded\n");
    } catch (...) {
        printf("[DEBUG] getState() call failed\n");
    }
    printf("[DEBUG] Filters created successfully\n");

    printf("[DEBUG] About to initialize motion filter\n");
    // Init motion filter first
    Eigen::Vector3f measured_position((float)lshape_info.reference_point.x(),
                                      (float)lshape_info.reference_point.y(), 0.0f);
    Eigen::Vector3f measured_velocity = Eigen::Vector3f::Zero();
    motion_filter_->Init(measured_position, measured_velocity);
    printf("[DEBUG] Motion filter initialized successfully\n");

    printf("[DEBUG] About to initialize shape filter\n");
    // Init shape filter system covariance
    auto system_covariance = shape_sys_model_.getCovariance();
    system_covariance.setZero();
    system_covariance(0, 0) = 1e-3;
    system_covariance(1, 1) = 1e-3;
    system_covariance(2, 2) = 0.001;
    system_covariance(3, 3) = 0.01;
    shape_sys_model_.setCovariance(system_covariance);

    // Init shape filter measurement covariance
    auto meas_covariance = shape_meas_model_.getCovariance();
    meas_covariance.setIdentity();
    shape_meas_model_.setCovariance(meas_covariance);

    // Init shape filter state
    Kalman::shape::State init_state;
    init_state.setZero();
    init_state(0) = lshape_info.l_shape(0);  // l1
    init_state(1) = lshape_info.l_shape(1);  // l2
    init_state(2) = lshape_info.l_shape(2);  // theta
    // init_state.theta_velocity() = 0.0f;  // 如果有第4个元素，可以设置
    shape_filter_->init(init_state);
    printf("[DEBUG] Shape filter initialized successfully\n");

    // Update time stamp
    last_predict_timestamp_motion_ = options.tracked_object->timestamp;
    last_predict_timestamp_shape_ = options.tracked_object->timestamp;

    // Update state status
    last_theta_ = lshape_info.l_shape.z();
    
    return true;
}

void LShapeMotionFilter::TransState(const Eigen::Isometry3f& tf) {

    if (motion_filter_) {
        motion_filter_->StateChange(tf);
    }

    return;
}

void LShapeMotionFilter::UpdateWithObject(const TrackedObjectConstPtr& track_data,
                               TrackedObjectPtr& new_object) {
    auto& lshape_info = new_object->object_ptr->lshape_box;

    // 判断角点是否发生转换
    int is_switch = detectCornerPointSwitch(last_theta_, lshape_info.l_shape[2]);

    // 若角点发生转换则转换模型中角点的位置
    if (is_switch != 0) {
        const auto& cur_state = motion_filter_->getState();
        float offset_x = lshape_info.reference_point[0] - cur_state.x();
        float offset_y = lshape_info.reference_point[1] - cur_state.y();
        motion_filter_->StateChange(offset_x, offset_y);
    }

    const auto& shape_state = shape_filter_->getState();
    double norm = normalize_angle(shape_state(2));  // theta
    double distance = shortest_angular_distance(norm, lshape_info.l_shape[2]);
    double theta = distance + shape_state(2);
    last_theta_ = lshape_info.l_shape[2];

    /* Update motion filter */
    if (is_switch == 0) {

        double time_diff = new_object->timestamp - last_predict_timestamp_motion_;

        // Predict process
        motion_filter_->predict(time_diff);

        // Update predict time
        last_predict_timestamp_motion_ = new_object->timestamp;

        // MOtion filter update process 
        Eigen::Vector3f measured_position;
        measured_position << lshape_info.reference_point[0], lshape_info.reference_point[1], 0.0f;
        motion_filter_->update(measured_position, Eigen::Vector3f::Zero());
    }

    /* Update shape filter */
    // Predict process
    double time_diff = new_object->timestamp - last_predict_timestamp_shape_;
    Kalman::shape::Control u;
    u.dt() = time_diff;
    shape_filter_->predict(shape_sys_model_, u);

    // Update predict time
    last_predict_timestamp_shape_ = new_object->timestamp;

    // Update shape filter measurement variance
    auto meas_covariance = shape_meas_model_.getCovariance();
    meas_covariance.setIdentity();
    meas_covariance(0, 0) = std::pow(lshape_info.l_shape[0], -1.0);
    meas_covariance(1, 1) = std::pow(lshape_info.l_shape[1], -1.0);
    shape_meas_model_.setCovariance(meas_covariance);

    // Shape filter update process
    Kalman::shape::ShapeMeasurement shape_meas;
    shape_meas.l1() = lshape_info.l_shape[0];  // l1
    shape_meas.l2() = lshape_info.l_shape[1];  // l2
    shape_meas.theta() = theta;  // theta
    shape_filter_->update(shape_meas_model_, shape_meas);

    // Update final state for result out
    ExportStateOut(new_object);
    
    return;
}

void LShapeMotionFilter::ExportStateOut(TrackedObjectPtr& new_object)  {
    // Eigen::Vector3d shape_filter_states = history_shape_meas_;      // shape: L1, L2, theta
    // Eigen::Vector2d dynamic_filter_states = history_dynamic_meas_;  // 角点 x, y

    // float L1 = shape_filter_->getState().l1();
    // float L2 = shape_filter_->getState().l2();
    // float theta = shape_filter_->getState().theta();

    // // Output center
    // // Equations 30 of "L-Shape Model Switching-Based precise motion tracking of
    // // moving vehicles"
    // double ex = (L1 * cos(theta) + L2 * sin(theta)) / 2;
    // double ey = (L1 * sin(theta) - L2 * cos(theta)) / 2;
    // new_object->output_center.x() = motion_filter_->getState()[0] + ex;
    // new_object->output_center.y() = motion_filter_->getState()[1] + ey;
    // new_object->output_center.z() = 0.0f;

    // Output velocity
    // Equations 31 of "L-Shape Model Switching-Based precise motion tracking of
    // moving vehicles"
    new_object->output_velocity.x() = motion_filter_->getState()[2];
    new_object->output_velocity.y() = motion_filter_->getState()[3];
    new_object->output_velocity.z() = 0.0f;

    // // Output corner points
    // Eigen::Vector3f corner_point;
    // new_object->output_corners[0].x() = motion_filter_->getState()[0];
    // new_object->output_corners[0].y() = motion_filter_->getState()[1];
    // new_object->output_corners[0].z() = 0.0f;

    // new_object->output_corners[1].x() += L1 * cos(theta);
    // new_object->output_corners[1].y() += L1 * sin(theta);
    // new_object->output_corners[1].z()  = 0.0f;

    // new_object->output_corners[2].x() += L2 * sin(theta);
    // new_object->output_corners[2].y() -= L2 * cos(theta);
    // new_object->output_corners[2].z()  = 0.0f;

    // new_object->output_corners[3].x() -= L1 * cos(theta);
    // new_object->output_corners[3].y() -= L1 * sin(theta);
    // new_object->output_corners[3].z()  = 0.0f;

    // // Output direction and size
    // float heading, length, width;
    // findOrientation(heading, length, width);
    // new_object->output_size.x() = length;
    // new_object->output_size.y() = width;
    // new_object->output_size.z() = 0;

    // new_object->output_theta = heading;

    // new_object->output_direction.x() = std::cos(heading);
    // new_object->output_direction.y() = std::sin(heading);
    // new_object->output_direction.z() = 0.0f;

}

void LShapeMotionFilter::findOrientation(float& direction, float& length, float& width){
    
    const auto& shape_info = shape_filter_->getState();
    const auto& motion_info = motion_filter_->getState();
    //This function finds the orientation of a moving object, when given an L-shape orientation
    std::vector<float> angles;
    double angle_norm = normalize_angle(shape_info(2));  // theta
    angles.push_back(angle_norm);
    angles.push_back(angle_norm + M_PI);
    angles.push_back(angle_norm + M_PI / 2);
    angles.push_back(angle_norm + 3 * M_PI / 2);

    double vsp = atan2(motion_info(3), motion_info(2));
    double min = 1.56;
    double distance, orientation = 0;
    int pos = 0;
    for(int i = 0; i < 4; ++i){
        distance = std::abs(shortest_angular_distance(vsp, angles[i]));
        if(distance < min){
            min = distance;
            orientation = normalize_angle(angles[i]);
            pos = i;
        }
    }
    if(pos == 0 || pos == 1){
        length = shape_info(0);  // l1
        width = shape_info(1);   // l2
    }
    else{
        length = shape_info(1);  // l2
        width = shape_info(0);   // l1
    }
    direction = normalize_angle(orientation);
}

void LShapeMotionFilter::UpdateWithoutObject(const TrackedObjectConstPtr& latest_object, 
                             TrackedObjectPtr& track_data) 
{
    return;
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
