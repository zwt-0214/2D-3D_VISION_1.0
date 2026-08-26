#include "lite_motion_planner/robot_model.hpp"

#include <lite_motion_msgs/srv/add_scene_object.hpp>
#include <lite_motion_msgs/srv/attach_scene_object.hpp>
#include <lite_motion_msgs/srv/plan_arm_motion.hpp>
#include <lite_motion_msgs/srv/remove_scene_object.hpp>
#include <lite_motion_msgs/srv/set_scene_collision_allowance.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Geometry>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

namespace lite_motion_planner {
namespace {

constexpr const char* kFilePrefix = "file://";
constexpr const char* kPackagePrefix = "package://";
constexpr double kPi = 3.14159265358979323846;

std::string normalizeAngleUnit(std::string unit) {
  std::transform(unit.begin(), unit.end(), unit.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return unit;
}

bool isDegreeUnit(const std::string& unit) {
  return unit == "deg" || unit == "degree" || unit == "degrees";
}

bool isRadianUnit(const std::string& unit) {
  return unit == "rad" || unit == "radian" || unit == "radians";
}

void degreesToRadiansInPlace(std::vector<double>& values) {
  for (double& value : values) {
    value *= kPi / 180.0;
  }
}

geometry_msgs::msg::Quaternion quaternionMsgFromRpy(double roll, double pitch, double yaw) {
  const Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
  const Eigen::Quaterniond q = rz * ry * rx;

  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

Eigen::Quaterniond quaternionEigenFromMsg(const geometry_msgs::msg::Quaternion& q_msg) {
  Eigen::Quaterniond q(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
  if (q.norm() < 1e-12) {
    q = Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q;
}

geometry_msgs::msg::Pose poseMsgFromSE3(const pinocchio::SE3& tf) {
  geometry_msgs::msg::Pose out;
  out.position.x = tf.translation().x();
  out.position.y = tf.translation().y();
  out.position.z = tf.translation().z();
  const Eigen::Quaterniond q(tf.rotation());
  out.orientation.x = q.x();
  out.orientation.y = q.y();
  out.orientation.z = q.z();
  out.orientation.w = q.w();
  return out;
}

pinocchio::SE3 poseMsgToSE3(const geometry_msgs::msg::Pose& pose) {
  return pinocchio::SE3(
    quaternionEigenFromMsg(pose.orientation).toRotationMatrix(),
    Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z));
}

double trajectoryDurationSec(const trajectory_msgs::msg::JointTrajectory& traj) {
  if (traj.points.empty()) {
    return 0.0;
  }
  const auto& last = traj.points.back().time_from_start;
  return static_cast<double>(last.sec) + static_cast<double>(last.nanosec) * 1e-9;
}

builtin_interfaces::msg::Duration durationFromSec(double seconds) {
  builtin_interfaces::msg::Duration out;
  seconds = std::max(0.0, seconds);
  out.sec = static_cast<int32_t>(std::floor(seconds));
  out.nanosec = static_cast<uint32_t>(
    std::llround((seconds - static_cast<double>(out.sec)) * 1e9));
  if (out.nanosec >= 1000000000u) {
    out.sec += 1;
    out.nanosec -= 1000000000u;
  }
  return out;
}

std::string markerMeshResource(const std::string& mesh_resource) {
  if (mesh_resource.rfind(kFilePrefix, 0) == 0 ||
      mesh_resource.rfind(kPackagePrefix, 0) == 0) {
    return mesh_resource;
  }
  return std::string(kFilePrefix) + mesh_resource;
}

}  // namespace

class TzbCatchNode : public rclcpp::Node {
public:
  TzbCatchNode()
  : Node("tzb_catch") {
    const auto urdf_path = this->declare_parameter<std::string>("urdf_path", "");
    const auto srdf_path = this->declare_parameter<std::string>("srdf_path", "");
    const auto mesh_package_dirs = this->declare_parameter<std::vector<std::string>>(
      "mesh_package_dirs", std::vector<std::string>());

    planning_frame_ = this->declare_parameter<std::string>("planning_frame", "base_link");
    arm_group_ = this->declare_parameter<std::string>("tzb.arm_group", "tzb_group");
    tcp_frame_ = this->declare_parameter<std::string>("tzb.tcp_frame", "tcp_link");
    e3_object_id_ = this->declare_parameter<std::string>("tzb.e3_object_id", "E3");
    e3_mesh_resource_ = this->declare_parameter<std::string>(
      "tzb.e3_mesh_resource",
      "/home/zwt/tzb_ws/src/lite_motion_planner/config/E3.STL");
    mesh_scale_ = this->declare_parameter<std::vector<double>>(
      "tzb.mesh_scale", std::vector<double>{0.001, 0.001, 0.001});
    if (!mesh_scale_.empty() && mesh_scale_.size() != 3) {
      throw std::runtime_error("tzb.mesh_scale must contain exactly 3 values");
    }

    e3_target_frame_ = this->declare_parameter<std::string>("tzb.e3_target_frame", "base_link");
    e3_target_xyz_ = declareVec3Param("tzb.e3_target_xyz", {0.35, 0.20, 0.20});
    e3_target_rpy_ = declareVec3Param("tzb.e3_target_rpy", {0.0, 0.0, 0.0});
    use_yaml_e3_target_ = this->declare_parameter<bool>("tzb.use_yaml_e3_target", true);
    pose_estimator_target_topic_ = this->declare_parameter<std::string>(
      "tzb.pose_estimator_target_topic", "/pose_estimator/object_pose");
    pose_estimator_wait_timeout_sec_ = std::max(
      0.0,
      this->declare_parameter<double>("tzb.pose_estimator_wait_timeout_sec", 3.0));
    pose_estimator_continuous_required_sec_ = std::max(
      0.0,
      this->declare_parameter<double>("tzb.pose_estimator_continuous_required_sec", 2.0));
    pose_estimator_continuous_max_gap_sec_ = std::max(
      1e-3,
      this->declare_parameter<double>("tzb.pose_estimator_continuous_max_gap_sec", 0.75));
    pose_estimator_reference_frame_ = this->declare_parameter<std::string>(
      "tzb.pose_estimator_reference_frame", "");
    pose_estimator_reference_transform_ = declareMat4Param(
      "tzb.pose_estimator_reference_transform_matrix",
      {1.0, 0.0, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0,
       0.0, 0.0, 1.0, 0.0,
       0.0, 0.0, 0.0, 1.0});
    freeze_camera_pointcloud_enabled_ =
      this->declare_parameter<bool>("tzb.freeze_camera_pointcloud_enabled", true);
    camera_pointcloud_topic_ = this->declare_parameter<std::string>(
      "tzb.camera_pointcloud_topic", "/camera/depth/points");
    frozen_camera_pointcloud_topic_ = this->declare_parameter<std::string>(
      "tzb.frozen_camera_pointcloud_topic", "/tzb_catch/frozen_observe_camera_points");
    frozen_camera_frame_id_ = this->declare_parameter<std::string>(
      "tzb.frozen_camera_frame_id", "tzb_observe_camera_link");
    apply_reference_transform_to_frozen_frame_ =
      this->declare_parameter<bool>("tzb.apply_reference_transform_to_frozen_frame", true);
    transform_frozen_pointcloud_to_reference_frame_ =
      this->declare_parameter<bool>("tzb.transform_frozen_pointcloud_to_reference_frame", true);
    frozen_camera_pointcloud_log_throttle_sec_ = std::max(
      0.1,
      this->declare_parameter<double>("tzb.frozen_camera_pointcloud_log_throttle_sec", 2.0));
    pose_estimator_axis_conversion_ = declareMat3Param(
      "tzb.pose_estimator_axis_conversion_matrix",
      {1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0});
    pre_grasp_distance_m_ = std::max(
      1e-4,
      this->declare_parameter<double>("tzb.pre_grasp_distance_m", 0.10));

    gripper_joint_names_ = this->declare_parameter<std::vector<std::string>>(
      "tzb.gripper_joint_names",
      std::vector<std::string>{"claw1_joint", "claw2_joint"});
    gripper_home_positions_ = this->declare_parameter<std::vector<double>>(
      "tzb.gripper_home_positions", std::vector<double>{0.0, 0.0});
    gripper_open_positions_ = this->declare_parameter<std::vector<double>>(
      "tzb.gripper_open_positions", std::vector<double>{-0.06, 0.06});
    gripper_close_positions_ = this->declare_parameter<std::vector<double>>(
      "tzb.gripper_close_positions", std::vector<double>{0.0, 0.0});

    grasp_collision_links_ = this->declare_parameter<std::vector<std::string>>(
      "tzb.grasp_collision_links",
      std::vector<std::string>{"tcp_link", "J6_link", "claw1_link", "claw2_link"});
    grasp_collision_allowed_penetration_m_ = this->declare_parameter<double>(
      "tzb.grasp_collision_allowed_penetration", -1.0);

    joint_planning_timeout_ = std::max(
      0.05,
      this->declare_parameter<double>("tzb.joint_planning_timeout", 2.0));
    cartesian_planning_timeout_ = std::max(
      0.05,
      this->declare_parameter<double>("tzb.cartesian_planning_timeout", 2.0));
    path_check_step_m_ = std::max(
      0.001,
      this->declare_parameter<double>("tzb.path_check_step", 0.05));
    cartesian_max_step_m_ = std::max(
      1e-4,
      this->declare_parameter<double>("tzb.cartesian_max_step_m", 0.02));
    cartesian_min_interpolation_steps_ = std::max<int>(
      1,
      this->declare_parameter<int>("tzb.cartesian_min_interpolation_steps", 2));
    cartesian_refinement_max_steps_ = std::max<int>(
      cartesian_min_interpolation_steps_,
      this->declare_parameter<int>("tzb.cartesian_refinement_max_steps", 16));
    cartesian_use_request_solver_settings_ =
      this->declare_parameter<bool>("tzb.cartesian_use_request_solver_settings", true);
    cartesian_use_sdls_ = this->declare_parameter<bool>("tzb.cartesian_use_sdls", true);
    cartesian_ik_max_iterations_ = std::max<int>(
      50,
      this->declare_parameter<int>("tzb.cartesian_ik_max_iterations", 800));
    cartesian_ik_pos_tolerance_ = std::max(
      1e-5,
      this->declare_parameter<double>("tzb.cartesian_ik_pos_tolerance", 0.002));
    cartesian_ik_rot_tolerance_ = std::max(
      1e-5,
      this->declare_parameter<double>("tzb.cartesian_ik_rot_tolerance", 0.05));
    cartesian_ik_damping_ = std::max(
      1e-12,
      this->declare_parameter<double>("tzb.cartesian_ik_damping", 1.0e-3));
    cartesian_ik_alpha_ = std::max(
      1e-3,
      this->declare_parameter<double>("tzb.cartesian_ik_alpha", 0.55));
    cartesian_ik_max_step_norm_ = std::max(
      1e-3,
      this->declare_parameter<double>("tzb.cartesian_ik_max_step_norm", 0.25));
    cartesian_ik_max_seed_count_ = std::max<int>(
      1,
      this->declare_parameter<int>("tzb.cartesian_ik_max_seed_count", 12));
    cartesian_nullspace_gain_ = std::max(
      0.0,
      this->declare_parameter<double>("tzb.cartesian_nullspace_gain", 0.0));
    cartesian_singular_threshold_ = std::max(
      1e-6,
      this->declare_parameter<double>("tzb.cartesian_singular_threshold", 0.03));
    cartesian_branch_jump_max_rad_ = std::max(
      1e-6,
      this->declare_parameter<double>("tzb.cartesian_branch_jump_max_rad", 0.75));
    cartesian_branch_jump_norm_max_ = std::max(
      1e-6,
      this->declare_parameter<double>("tzb.cartesian_branch_jump_norm_max", 1.20));
    cartesian_waypoint_pos_tolerance_ = std::max(
      1e-6,
      this->declare_parameter<double>("tzb.cartesian_waypoint_pos_tolerance", 0.002));
    cartesian_waypoint_rot_tolerance_ = std::max(
      1e-6,
      this->declare_parameter<double>("tzb.cartesian_waypoint_rot_tolerance", 0.05));

    post_action_wait_sec_ = this->declare_parameter<double>("tzb.post_action_wait_sec", 0.05);
    service_wait_sec_ = this->declare_parameter<double>("tzb.service_wait_sec", 30.0);
    response_wait_sec_ = this->declare_parameter<double>("tzb.response_wait_sec", 30.0);
    e3_marker_topic_ = this->declare_parameter<std::string>("tzb.e3_marker_topic", "/tzb_catch/e3_marker");
    reference_marker_republish_hz_ = std::max(
      0.1,
      this->declare_parameter<double>("tzb.reference_marker_republish_hz", 5.0));
    sequence_loop_ = this->declare_parameter<bool>("tzb.sequence_loop", false);
    loop_pause_sec_ = std::max(0.0, this->declare_parameter<double>("tzb.loop_pause_sec", 0.5));
    replay_after_complete_ = this->declare_parameter<bool>("tzb.replay_after_complete", true);
    replay_loop_ = this->declare_parameter<bool>("tzb.replay_loop", true);
    replay_speed_scale_ = std::max(
      1e-3,
      this->declare_parameter<double>("tzb.replay_speed_scale", 0.35));
    replay_segment_pause_sec_ = std::max(
      0.0,
      this->declare_parameter<double>("tzb.replay_segment_pause_sec", 0.15));
    joint_goal_tolerance_rad_ = std::max(
      1e-6,
      this->declare_parameter<double>("tzb.joint_goal_tolerance_rad", 0.01));
    arm_command_enabled_ = this->declare_parameter<bool>("tzb.arm_command_enabled", true);
    arm_command_publish_hz_ = std::max(
      1.0,
      this->declare_parameter<double>("tzb.arm_command_publish_hz", 20.0));
    arm_command_speed_scale_ = std::max(
      1e-3,
      this->declare_parameter<double>("tzb.arm_command_speed_scale", 1.0));
    arm_command_goal_tolerance_rad_ = std::max(
      1e-6,
      this->declare_parameter<double>("tzb.arm_command_goal_tolerance_rad", 0.005));
    arm_command_publish_during_replay_ =
      this->declare_parameter<bool>("tzb.arm_command_publish_during_replay", false);

    enable_visualization_ = this->declare_parameter<bool>("enable_visualization", true);
    visualization_playback_enabled_ =
      enable_visualization_ && this->declare_parameter<bool>("enable_joint_state_playback", true);
    visualization_playback_speed_scale_ = std::max(
      1e-3,
      this->declare_parameter<double>("joint_state_playback_speed_scale", 1.0));

    robot_model_ = std::make_unique<RobotModel>(urdf_path, srdf_path, mesh_package_dirs);
    const auto& arm_group = robot_model_->getPlanningGroup(arm_group_);

    joint_positions_unit_ =
      normalizeAngleUnit(this->declare_parameter<std::string>("tzb.joint_positions_unit", "rad"));
    if (!isRadianUnit(joint_positions_unit_) && !isDegreeUnit(joint_positions_unit_)) {
      throw std::runtime_error("tzb.joint_positions_unit must be 'rad' or 'deg'");
    }
    initial_arm_positions_ = this->declare_parameter<std::vector<double>>(
      "tzb.ini_joint_positions", std::vector<double>(arm_group.joint_names.size(), 0.0));
    observe_joint_positions_ = this->declare_parameter<std::vector<double>>(
      "tzb.observe_joint_positions",
      std::vector<double>{0.0, 0.0, 1.7, 0.0, 0.0, 0.0});
    max_arm_command_velocity_rad_s_ = this->declare_parameter<std::vector<double>>(
      "tzb.max_arm_command_velocity_rad_s",
      std::vector<double>{0.5, 0.5, 0.5, 0.8, 0.8, 1.0});

    if (initial_arm_positions_.size() != arm_group.joint_names.size()) {
      throw std::runtime_error("tzb.ini_joint_positions size does not match tzb arm group dof");
    }
    if (observe_joint_positions_.size() != arm_group.joint_names.size()) {
      throw std::runtime_error("tzb.observe_joint_positions size does not match tzb arm group dof");
    }
    if (isDegreeUnit(joint_positions_unit_)) {
      degreesToRadiansInPlace(initial_arm_positions_);
      degreesToRadiansInPlace(observe_joint_positions_);
    }
    if (max_arm_command_velocity_rad_s_.size() != arm_group.joint_names.size()) {
      throw std::runtime_error("tzb.max_arm_command_velocity_rad_s size does not match tzb arm group dof");
    }
    if (gripper_home_positions_.size() != gripper_joint_names_.size() ||
        gripper_open_positions_.size() != gripper_joint_names_.size() ||
        gripper_close_positions_.size() != gripper_joint_names_.size()) {
      throw std::runtime_error("tzb gripper position arrays must match tzb.gripper_joint_names size");
    }

    arm_state_ = initial_arm_positions_;
    gripper_state_ = gripper_home_positions_;

    plan_client_ = this->create_client<lite_motion_msgs::srv::PlanArmMotion>("plan_arm_motion");
    add_object_client_ = this->create_client<lite_motion_msgs::srv::AddSceneObject>("add_scene_object");
    attach_object_client_ = this->create_client<lite_motion_msgs::srv::AttachSceneObject>("attach_scene_object");
    remove_object_client_ = this->create_client<lite_motion_msgs::srv::RemoveSceneObject>("remove_scene_object");
    set_scene_collision_allowance_client_ =
      this->create_client<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
        "set_scene_collision_allowance");
    trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      this->declare_parameter<std::string>(
        "joint_trajectory_topic", "/joint_trajectory_controller/joint_trajectory"),
      10);
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      this->declare_parameter<std::string>("visualization_joint_states_topic", "/joint_states"),
      rclcpp::QoS(10));
    if (arm_command_enabled_) {
      arm_command_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        this->declare_parameter<std::string>(
          "tzb.arm_command_topic", "/tzb_catch/arm_joint_command"),
        rclcpp::SensorDataQoS());
    }
    if (!use_yaml_e3_target_) {
      pose_estimator_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        pose_estimator_target_topic_,
        rclcpp::QoS(10),
        [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          onPoseEstimatorTarget(*msg);
        });
    }
    if (freeze_camera_pointcloud_enabled_) {
      frozen_camera_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
      rclcpp::QoS frozen_cloud_qos(2);
      frozen_cloud_qos.reliable().durability_volatile();
      frozen_camera_pointcloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
          frozen_camera_pointcloud_topic_, frozen_cloud_qos);
      camera_pointcloud_sub_ =
        this->create_subscription<sensor_msgs::msg::PointCloud2>(
          camera_pointcloud_topic_,
          rclcpp::SensorDataQoS(),
          [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            onCameraPointCloud(*msg);
          });
      frozen_camera_tf_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        [this]() {
          publishFrozenCameraFrame();
        });
    }

    if (enable_visualization_) {
      rclcpp::QoS latched_qos(1);
      latched_qos.transient_local().reliable();
      ref_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "tzb_catch/reference_frames", latched_qos);
      e3_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        e3_marker_topic_, latched_qos);
      reference_marker_timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / reference_marker_republish_hz_)),
        [this]() {
          publishReferenceMarkers();
          publishE3Marker();
        });
    }

    RCLCPP_INFO(
      this->get_logger(),
      "tzb_catch configured: arm_group=%s tcp_frame=%s joint_positions_unit=%s target_frame=%s target_source=%s estimator_reference_frame=%s frozen_cloud=%s continuous_required=%.2fs pre_dis=%.3f e3_mesh=%s",
      arm_group_.c_str(),
      tcp_frame_.c_str(),
      joint_positions_unit_.c_str(),
      e3_target_frame_.c_str(),
      use_yaml_e3_target_ ? "yaml" : pose_estimator_target_topic_.c_str(),
      pose_estimator_reference_frame_.empty() ? "<message_header>" : pose_estimator_reference_frame_.c_str(),
      freeze_camera_pointcloud_enabled_ ? frozen_camera_pointcloud_topic_.c_str() : "disabled",
      pose_estimator_continuous_required_sec_,
      pre_grasp_distance_m_,
      e3_mesh_resource_.c_str());

    start_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        start_timer_->cancel();
        publishCurrentTheoreticalJointState("initial_home");
        publishReferenceMarkers("initial");
        publishE3Marker();
        worker_ = std::thread([this]() { runSequence(); });
      });
  }

  ~TzbCatchNode() override {
    stop_requested_.store(true);
    pose_estimator_cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  std::array<double, 3> declareVec3Param(
    const std::string& name,
    const std::array<double, 3>& default_value) {
    auto values = this->declare_parameter<std::vector<double>>(
      name, std::vector<double>{default_value[0], default_value[1], default_value[2]});
    if (values.size() != 3) {
      throw std::runtime_error(name + " must contain exactly 3 values");
    }
    return {values[0], values[1], values[2]};
  }

  Eigen::Matrix3d declareMat3Param(
    const std::string& name,
    const std::array<double, 9>& default_value) {
    auto values = this->declare_parameter<std::vector<double>>(
      name,
      std::vector<double>(default_value.begin(), default_value.end()));
    if (values.size() != 9) {
      throw std::runtime_error(name + " must contain exactly 9 values");
    }

    Eigen::Matrix3d out;
    out << values[0], values[1], values[2],
           values[3], values[4], values[5],
           values[6], values[7], values[8];
    const double orthogonality_error =
      (out.transpose() * out - Eigen::Matrix3d::Identity()).norm();
    const double det = out.determinant();
    if (orthogonality_error > 1e-3 || std::abs(det - 1.0) > 1e-3) {
      RCLCPP_WARN(
        this->get_logger(),
        "%s is not a proper rotation matrix: det=%.6f orthogonality_error=%.6f",
        name.c_str(),
        det,
        orthogonality_error);
    }
    return out;
  }

  Eigen::Matrix4d declareMat4Param(
    const std::string& name,
    const std::array<double, 16>& default_value) {
    auto values = this->declare_parameter<std::vector<double>>(
      name,
      std::vector<double>(default_value.begin(), default_value.end()));
    if (values.size() != 16) {
      throw std::runtime_error(name + " must contain exactly 16 values");
    }

    Eigen::Matrix4d out;
    out << values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7],
           values[8], values[9], values[10], values[11],
           values[12], values[13], values[14], values[15];
    const Eigen::Matrix3d rot = out.block<3, 3>(0, 0);
    const double orthogonality_error =
      (rot.transpose() * rot - Eigen::Matrix3d::Identity()).norm();
    const double det = rot.determinant();
    const double bottom_row_error =
      (out.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).norm();
    if (orthogonality_error > 1e-3 || std::abs(det - 1.0) > 1e-3 ||
        bottom_row_error > 1e-6) {
      RCLCPP_WARN(
        this->get_logger(),
        "%s is not a proper SE3 transform: det=%.6f orthogonality_error=%.6f bottom_row_error=%.6f",
        name.c_str(),
        det,
        orthogonality_error,
        bottom_row_error);
    }
    return out;
  }

  struct ReplaySegment {
    std::string stage_name;
    trajectory_msgs::msg::JointTrajectory trajectory;
    bool e3_attached_during_segment{false};
  };

  geometry_msgs::msg::PoseStamped makePoseStamped(
    const std::array<double, 3>& xyz,
    const std::array<double, 3>& rpy,
    const std::string& frame_id) const {
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = frame_id;
    out.pose.position.x = xyz[0];
    out.pose.position.y = xyz[1];
    out.pose.position.z = xyz[2];
    out.pose.orientation = quaternionMsgFromRpy(rpy[0], rpy[1], rpy[2]);
    return out;
  }

  Eigen::VectorXd currentFullState() const {
    Eigen::VectorXd q_full = robot_model_->neutralConfiguration();
    const Eigen::VectorXd q_arm = Eigen::Map<const Eigen::VectorXd>(
      arm_state_.data(), static_cast<Eigen::Index>(arm_state_.size()));
    q_full = robot_model_->groupToFull(arm_group_, q_arm, q_full);

    for (size_t i = 0; i < gripper_joint_names_.size() && i < gripper_state_.size(); ++i) {
      try {
        q_full[robot_model_->qIndexOfJoint(gripper_joint_names_[i])] = gripper_state_[i];
      } catch (const std::exception&) {
        RCLCPP_DEBUG(
          this->get_logger(),
          "Gripper joint [%s] is not an active Pinocchio joint; skipping full-state sync.",
          gripper_joint_names_[i].c_str());
      }
    }
    return q_full;
  }

  pinocchio::SE3 currentFramePose(const std::string& frame_name) {
    const Eigen::VectorXd q_full = currentFullState();
    pinocchio::forwardKinematics(robot_model_->model(), robot_model_->data(), q_full);
    pinocchio::updateFramePlacements(robot_model_->model(), robot_model_->data());
    return robot_model_->data().oMf[robot_model_->getFrameIdChecked(frame_name)];
  }

  geometry_msgs::msg::PoseStamped currentFramePoseStamped(const std::string& frame_name) {
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = planning_frame_;
    out.pose = poseMsgFromSE3(currentFramePose(frame_name));
    return out;
  }

  geometry_msgs::msg::PoseStamped transformPoseToPlanningFrame(
    const geometry_msgs::msg::PoseStamped& in) {
    if (in.header.frame_id.empty() || in.header.frame_id == planning_frame_) {
      geometry_msgs::msg::PoseStamped out = in;
      out.header.frame_id = planning_frame_;
      return out;
    }

    const pinocchio::SE3 ref_in_planning = referenceFramePoseInPlanningFrame(in.header.frame_id);
    const pinocchio::SE3 pose_in_ref = poseMsgToSE3(in.pose);
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = planning_frame_;
    out.pose = poseMsgFromSE3(ref_in_planning * pose_in_ref);
    return out;
  }

  pinocchio::SE3 referenceFramePoseInPlanningFrame(const std::string& frame_name) {
    if (apply_reference_transform_to_frozen_frame_ &&
        !frozen_camera_frame_id_.empty() &&
        frame_name == frozen_camera_frame_id_) {
      pinocchio::SE3 shadow_pose;
      if (shadowCameraFramePose(shadow_pose)) {
        return shadow_pose;
      }
    }
    {
      std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
      if (have_frozen_pose_estimator_reference_ &&
          !pose_estimator_reference_frame_.empty() &&
          frame_name == pose_estimator_reference_frame_) {
        return frozen_pose_estimator_reference_in_planning_;
      }
    }
    return currentFramePose(frame_name);
  }

  pinocchio::SE3 poseEstimatorReferenceTransformSE3() const {
    return pinocchio::SE3(
      pose_estimator_reference_transform_.block<3, 3>(0, 0),
      pose_estimator_reference_transform_.block<3, 1>(0, 3));
  }

  geometry_msgs::msg::TransformStamped transformMsgFromSE3(
    const pinocchio::SE3& tf,
    const std::string& parent_frame,
    const std::string& child_frame) const {
    geometry_msgs::msg::TransformStamped out;
    out.header.stamp = this->now();
    out.header.frame_id = parent_frame;
    out.child_frame_id = child_frame;
    out.transform.translation.x = tf.translation().x();
    out.transform.translation.y = tf.translation().y();
    out.transform.translation.z = tf.translation().z();
    const Eigen::Quaterniond q(tf.rotation());
    out.transform.rotation.x = q.x();
    out.transform.rotation.y = q.y();
    out.transform.rotation.z = q.z();
    out.transform.rotation.w = q.w();
    return out;
  }

  bool shadowCameraFramePose(pinocchio::SE3& out) {
    if (pose_estimator_reference_frame_.empty()) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
      if (have_frozen_pose_estimator_reference_) {
        out = frozen_pose_estimator_reference_in_planning_;
        if (apply_reference_transform_to_frozen_frame_) {
          out = out * poseEstimatorReferenceTransformSE3();
        }
        return true;
      }
    }

    try {
      out = currentFramePose(pose_estimator_reference_frame_);
    } catch (const std::exception& e) {
      if (!shadow_camera_pose_error_logged_) {
        shadow_camera_pose_error_logged_ = true;
        RCLCPP_WARN(
          this->get_logger(),
          "Shadow camera frame [%s] is not available before observe: %s",
          pose_estimator_reference_frame_.c_str(),
          e.what());
      }
      return false;
    }

    if (apply_reference_transform_to_frozen_frame_) {
      out = out * poseEstimatorReferenceTransformSE3();
    }
    return true;
  }

  bool shadowCameraFramePoseStamped(geometry_msgs::msg::PoseStamped& out) {
    pinocchio::SE3 pose;
    if (!shadowCameraFramePose(pose)) {
      return false;
    }
    out.header.frame_id = planning_frame_;
    out.pose = poseMsgFromSE3(pose);
    return true;
  }

  void publishFrozenCameraFrame() {
    if (!frozen_camera_tf_broadcaster_) {
      return;
    }
    pinocchio::SE3 shadow_pose;
    if (!shadowCameraFramePose(shadow_pose)) {
      return;
    }
    frozen_camera_tf_broadcaster_->sendTransform(
      transformMsgFromSE3(shadow_pose, planning_frame_, frozen_camera_frame_id_));
  }

  void transformPointCloudToReferenceFrame(sensor_msgs::msg::PointCloud2& cloud) {
    if (!transform_frozen_pointcloud_to_reference_frame_ ||
        apply_reference_transform_to_frozen_frame_) {
      return;
    }

    try {
      sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
          continue;
        }
        const Eigen::Vector4d p_src(*iter_x, *iter_y, *iter_z, 1.0);
        const Eigen::Vector4d p_ref = pose_estimator_reference_transform_ * p_src;
        *iter_x = static_cast<float>(p_ref.x());
        *iter_y = static_cast<float>(p_ref.y());
        *iter_z = static_cast<float>(p_ref.z());
      }
    } catch (const std::exception& e) {
      if (!pointcloud_transform_error_logged_) {
        pointcloud_transform_error_logged_ = true;
        RCLCPP_WARN(
          this->get_logger(),
          "Frozen point cloud transform skipped: %s",
          e.what());
      }
    }
  }

  void onCameraPointCloud(const sensor_msgs::msg::PointCloud2& msg) {
    if (!frozen_camera_pointcloud_pub_) {
      return;
    }
    pinocchio::SE3 shadow_pose;
    if (!shadowCameraFramePose(shadow_pose)) {
      return;
    }

    sensor_msgs::msg::PointCloud2 out = msg;
    transformPointCloudToReferenceFrame(out);
    out.header.frame_id = frozen_camera_frame_id_;
    out.header.stamp = this->now();
    frozen_camera_pointcloud_pub_->publish(out);
    publishFrozenCameraFrame();

    const auto now_mono = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now_mono - last_frozen_pointcloud_log_time_).count() >
        frozen_camera_pointcloud_log_throttle_sec_) {
      last_frozen_pointcloud_log_time_ = now_mono;
      bool frozen = false;
      {
        std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
        frozen = have_frozen_pose_estimator_reference_;
      }
      RCLCPP_INFO(
        this->get_logger(),
        "Frozen camera cloud relay: source=%s source_frame=%s -> topic=%s frame=%s points=%u mode=%s",
        camera_pointcloud_topic_.c_str(),
        msg.header.frame_id.c_str(),
        frozen_camera_pointcloud_topic_.c_str(),
        frozen_camera_frame_id_.c_str(),
        msg.width * msg.height,
        frozen ? "frozen_at_observe" : "following_camera_link");
    }
  }

  geometry_msgs::msg::PoseStamped e3TargetPoseInPlanningFrame() {
    return transformPoseToPlanningFrame(
      makePoseStamped(e3_target_xyz_, e3_target_rpy_, e3_target_frame_));
  }

  void onPoseEstimatorTarget(const geometry_msgs::msg::PoseStamped& msg) {
    const auto now = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
      if (!pose_estimator_capture_enabled_ || have_pose_estimator_target_) {
        return;
      }
    }

    if (!isValidPoseEstimatorTarget(msg)) {
      std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
      resetPoseEstimatorContinuityLocked();
      return;
    }

    {
      std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
      if (!pose_estimator_capture_enabled_ || have_pose_estimator_target_) {
        return;
      }

      if (!pose_estimator_stream_active_ ||
          std::chrono::duration<double>(now - pose_estimator_last_msg_time_).count() >
            pose_estimator_continuous_max_gap_sec_) {
        pose_estimator_stream_active_ = true;
        pose_estimator_stream_start_time_ = now;
        RCLCPP_INFO(
          this->get_logger(),
          "Pose estimator target stream started; waiting %.2f s continuous publish before locking target.",
          pose_estimator_continuous_required_sec_);
      }
      pose_estimator_last_msg_time_ = now;

      const double continuous_sec =
        std::chrono::duration<double>(now - pose_estimator_stream_start_time_).count();
      if (continuous_sec < pose_estimator_continuous_required_sec_) {
        return;
      }

      latest_pose_estimator_target_ = msg;
      have_pose_estimator_target_ = true;
      RCLCPP_INFO(
        this->get_logger(),
        "Pose estimator target locked after %.3f s continuous publish; later poses will be ignored.",
        continuous_sec);
    }
    pose_estimator_cv_.notify_all();
  }

  void resetPoseEstimatorCaptureLocked() {
    have_pose_estimator_target_ = false;
    pose_estimator_stream_active_ = false;
    pose_estimator_capture_enabled_ = false;
    latest_pose_estimator_target_ = geometry_msgs::msg::PoseStamped();
  }

  bool enablePoseEstimatorCaptureAtCurrentReferencePose() {
    std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
    resetPoseEstimatorCaptureLocked();
    if (!pose_estimator_reference_frame_.empty()) {
      try {
        frozen_pose_estimator_reference_in_planning_ =
          currentFramePose(pose_estimator_reference_frame_);
        have_frozen_pose_estimator_reference_ = true;
      } catch (const std::exception& e) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Failed to freeze pose estimator reference frame [%s] in [%s]: %s",
          pose_estimator_reference_frame_.c_str(),
          planning_frame_.c_str(),
          e.what());
        return false;
      }
    } else {
      have_frozen_pose_estimator_reference_ = false;
    }
    pose_estimator_capture_enabled_ = true;
    RCLCPP_INFO(
      this->get_logger(),
      "Pose estimator capture enabled at observe pose; frozen_reference_frame=%s",
      pose_estimator_reference_frame_.empty() ? "<message_header>" : pose_estimator_reference_frame_.c_str());
    return true;
  }

  bool isValidPoseEstimatorTarget(const geometry_msgs::msg::PoseStamped& msg) const {
    const auto& p = msg.pose.position;
    const auto& q = msg.pose.orientation;
    const double q_norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
           std::isfinite(q.x) && std::isfinite(q.y) &&
           std::isfinite(q.z) && std::isfinite(q.w) &&
           q_norm_sq > 1e-12;
  }

  void resetPoseEstimatorContinuityLocked() {
    if (have_pose_estimator_target_) {
      return;
    }
    if (pose_estimator_stream_active_) {
      RCLCPP_WARN(
        this->get_logger(),
        "Pose estimator target stream continuity reset because an invalid pose was received.");
    }
    pose_estimator_stream_active_ = false;
  }

  geometry_msgs::msg::PoseStamped applyPoseEstimatorAxisConversion(
    const geometry_msgs::msg::PoseStamped& in) const {
    geometry_msgs::msg::PoseStamped out = in;
    const Eigen::Matrix3d converted_rotation =
      quaternionEigenFromMsg(in.pose.orientation).toRotationMatrix() *
      pose_estimator_axis_conversion_;
    Eigen::Quaterniond q(converted_rotation);
    q.normalize();
    out.pose.orientation.x = q.x();
    out.pose.orientation.y = q.y();
    out.pose.orientation.z = q.z();
    out.pose.orientation.w = q.w();
    return out;
  }

  geometry_msgs::msg::PoseStamped applyPoseEstimatorReferenceTransform(
    const geometry_msgs::msg::PoseStamped& in) const {
    if (pose_estimator_reference_frame_.empty()) {
      return in;
    }
    if (apply_reference_transform_to_frozen_frame_) {
      geometry_msgs::msg::PoseStamped out = in;
      out.header.frame_id = frozen_camera_frame_id_;
      return out;
    }

    const pinocchio::SE3 reference_from_estimator = poseEstimatorReferenceTransformSE3();
    geometry_msgs::msg::PoseStamped out;
    out.header = in.header;
    out.header.frame_id = pose_estimator_reference_frame_;
    out.pose = poseMsgFromSE3(reference_from_estimator * poseMsgToSE3(in.pose));
    return out;
  }

  geometry_msgs::msg::PoseStamped preparePoseEstimatorTarget(
    const geometry_msgs::msg::PoseStamped& in) const {
    return applyPoseEstimatorAxisConversion(applyPoseEstimatorReferenceTransform(in));
  }

  bool waitForPoseEstimatorTarget(geometry_msgs::msg::PoseStamped& out) {
    std::unique_lock<std::mutex> lock(pose_estimator_mutex_);
    const auto has_target = [this]() {
      return stop_requested_.load() || have_pose_estimator_target_;
    };
    if (!has_target()) {
      RCLCPP_INFO(
        this->get_logger(),
        "Waiting for pose estimator target on [%s] for up to %.2f s after %.2f s continuous publish.",
        pose_estimator_target_topic_.c_str(),
        pose_estimator_wait_timeout_sec_,
        pose_estimator_continuous_required_sec_);
      if (pose_estimator_wait_timeout_sec_ <= 0.0) {
        pose_estimator_cv_.wait(lock, has_target);
      } else if (!pose_estimator_cv_.wait_for(
          lock,
          std::chrono::duration<double>(pose_estimator_wait_timeout_sec_),
          has_target)) {
        RCLCPP_ERROR(
          this->get_logger(),
          "No locked pose estimator target received on [%s] within %.2f s.",
          pose_estimator_target_topic_.c_str(),
          pose_estimator_wait_timeout_sec_);
        return false;
      }
    }
    if (stop_requested_.load() || !have_pose_estimator_target_) {
      return false;
    }
    out = latest_pose_estimator_target_;
    return true;
  }

  bool lockedPoseEstimatorTargetInPlanningFrame(geometry_msgs::msg::PoseStamped& out) {
    geometry_msgs::msg::PoseStamped sensed_pose;
    {
      std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
      if (!have_pose_estimator_target_) {
        return false;
      }
      sensed_pose = latest_pose_estimator_target_;
    }

    geometry_msgs::msg::PoseStamped prepared_pose;
    try {
      prepared_pose = preparePoseEstimatorTarget(sensed_pose);
      out = transformPoseToPlanningFrame(prepared_pose);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to transform locked pose estimator target from source_frame [%s] prepared_frame [%s] to [%s] for visualization: %s",
        sensed_pose.header.frame_id.c_str(),
        prepared_pose.header.frame_id.c_str(),
        planning_frame_.c_str(),
        e.what());
      return false;
    }
    return true;
  }

  bool resolveE3TargetPose(geometry_msgs::msg::PoseStamped& out) {
    if (use_yaml_e3_target_) {
      out = e3TargetPoseInPlanningFrame();
      RCLCPP_INFO(this->get_logger(), "Using YAML E3 target pose.");
      return true;
    }

    geometry_msgs::msg::PoseStamped sensed_pose;
    if (!waitForPoseEstimatorTarget(sensed_pose)) {
      return false;
    }
    geometry_msgs::msg::PoseStamped prepared_pose;
    try {
      prepared_pose = preparePoseEstimatorTarget(sensed_pose);
      out = transformPoseToPlanningFrame(prepared_pose);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to transform pose estimator target from source_frame [%s] prepared_frame [%s] to [%s]: %s",
        sensed_pose.header.frame_id.c_str(),
        prepared_pose.header.frame_id.c_str(),
        planning_frame_.c_str(),
        e.what());
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Using pose estimator E3 target: source_frame=%s prepared_frame=%s xyz=(%.4f, %.4f, %.4f).",
      sensed_pose.header.frame_id.c_str(),
      prepared_pose.header.frame_id.c_str(),
      out.pose.position.x,
      out.pose.position.y,
      out.pose.position.z);
    return true;
  }

  geometry_msgs::msg::PoseStamped preGraspPoseForE3Target(
    const geometry_msgs::msg::PoseStamped& e3_target) const {
    geometry_msgs::msg::PoseStamped out = e3_target;
    const Eigen::Vector3d e3(
      e3_target.pose.position.x,
      e3_target.pose.position.y,
      e3_target.pose.position.z);
    const Eigen::Vector3d y_reference_point(0.0, e3.y(), 0.0);
    Eigen::Vector3d direction = y_reference_point - e3;
    if (direction.norm() < 1e-9) {
      direction =
        quaternionEigenFromMsg(e3_target.pose.orientation).toRotationMatrix() *
        Eigen::Vector3d::UnitZ();
      RCLCPP_WARN(
        this->get_logger(),
        "E3 target is already at the base Y-reference point; pre_grasp_point uses E3 local +Z fallback.");
    }
    direction.normalize();
    const Eigen::Vector3d pre = e3 + direction * pre_grasp_distance_m_;
    out.pose.position.x = pre.x();
    out.pose.position.y = pre.y();
    out.pose.position.z = pre.z();
    return out;
  }

  geometry_msgs::msg::Pose computeAttachRelativePose(
    const std::string& frame_name,
    const geometry_msgs::msg::PoseStamped& object_world_pose) {
    const pinocchio::SE3 frame_world = currentFramePose(frame_name);
    const pinocchio::SE3 object_world = poseMsgToSE3(object_world_pose.pose);
    return poseMsgFromSE3(frame_world.actInv(object_world));
  }

  sensor_msgs::msg::JointState makeJointStateMsg(
    const Eigen::VectorXd& q_full,
    const rclcpp::Time& stamp) const {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = stamp;
    msg.name = robot_model_->activeJointNames();
    msg.position.reserve(msg.name.size());
    for (const auto& joint_name : msg.name) {
      msg.position.push_back(q_full[robot_model_->qIndexOfJoint(joint_name)]);
    }
    return msg;
  }

  void publishCurrentTheoreticalJointState(const std::string& stage_name) {
    if (!joint_state_pub_ || !enable_visualization_) {
      return;
    }
    joint_state_pub_->publish(makeJointStateMsg(currentFullState(), this->now()));
    RCLCPP_DEBUG(
      this->get_logger(),
      "Stage [%s] published current theoretical joint state.",
      stage_name.c_str());
  }

  bool publishGroupHoldTrajectory(
    const std::string& group_name,
    const std::vector<double>& target_joints,
    const std::string& stage_name,
    bool record_for_replay = false) {
    if (!trajectory_pub_) {
      return true;
    }

    const auto& group = robot_model_->getPlanningGroup(group_name);
    if (target_joints.size() != group.joint_names.size()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Stage [%s] hold trajectory size mismatch: target=%zu group=%zu",
        stage_name.c_str(),
        target_joints.size(),
        group.joint_names.size());
      return false;
    }

    trajectory_msgs::msg::JointTrajectory trajectory;
    trajectory.header.stamp = this->now();
    trajectory.joint_names = group.joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = target_joints;
    point.velocities.assign(target_joints.size(), 0.0);
    point.accelerations.assign(target_joints.size(), 0.0);
    point.time_from_start = durationFromSec(std::max(0.02, post_action_wait_sec_));
    trajectory.points.push_back(std::move(point));

    trajectory_pub_->publish(trajectory);
    if (record_for_replay) {
      recordReplaySegment(stage_name, trajectory);
    }
    return true;
  }

  bool publishGripperTrajectory(
    const std::vector<double>& target_joints,
    const std::string& stage_name) {
    if (target_joints.size() != gripper_joint_names_.size()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Stage [%s] gripper target size=%zu, expected=%zu",
        stage_name.c_str(),
        target_joints.size(),
        gripper_joint_names_.size());
      return false;
    }

    trajectory_msgs::msg::JointTrajectory trajectory;
    trajectory.header.stamp = this->now();
    trajectory.joint_names = gripper_joint_names_;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = target_joints;
    point.velocities.assign(target_joints.size(), 0.0);
    point.accelerations.assign(target_joints.size(), 0.0);
    point.time_from_start = durationFromSec(std::max(0.02, gripper_motion_duration_sec_));
    trajectory.points.push_back(std::move(point));

    trajectory_pub_->publish(trajectory);
    recordReplaySegment(stage_name, trajectory);
    gripper_state_ = target_joints;
    RCLCPP_INFO(this->get_logger(), "Stage [%s] gripper command published.", stage_name.c_str());
    sleepAfterAction(gripper_motion_duration_sec_ + post_action_wait_sec_);
    publishCurrentTheoreticalJointState(stage_name + "_final_sync");
    return true;
  }

  template<typename ServiceT>
  bool waitForService(
    const typename rclcpp::Client<ServiceT>::SharedPtr& client,
    const std::string& service_name) {
    if (client->wait_for_service(std::chrono::duration<double>(service_wait_sec_))) {
      return true;
    }
    RCLCPP_ERROR(
      this->get_logger(),
      "Service [%s] not available within %.2f s",
      service_name.c_str(),
      service_wait_sec_);
    return false;
  }

  template<typename ServiceT>
  std::shared_ptr<typename ServiceT::Response> callService(
    const typename rclcpp::Client<ServiceT>::SharedPtr& client,
    const std::shared_ptr<typename ServiceT::Request>& request,
    const std::string& service_name) {
    auto future = client->async_send_request(request);
    if (future.wait_for(std::chrono::duration<double>(response_wait_sec_)) !=
        std::future_status::ready) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Timed out waiting for service [%s] response after %.2f s",
        service_name.c_str(),
        response_wait_sec_);
      return nullptr;
    }
    return future.get();
  }

  bool addMeshObject(
    const std::string& object_id,
    const std::string& mesh_resource,
    const geometry_msgs::msg::PoseStamped& object_pose) {
    auto req = std::make_shared<lite_motion_msgs::srv::AddSceneObject::Request>();
    req->object_id = object_id;
    req->mesh_resource = mesh_resource;
    req->pose = object_pose;
    req->collision_pose = object_pose.pose;
    req->mesh_scale = mesh_scale_;

    auto resp = callService<lite_motion_msgs::srv::AddSceneObject>(
      add_object_client_, req, "add_scene_object");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Add scene object [%s] failed: %s",
        object_id.c_str(),
        resp ? resp->message.c_str() : "no response");
      return false;
    }
    RCLCPP_INFO(this->get_logger(), "Scene object [%s] loaded from [%s]", object_id.c_str(), mesh_resource.c_str());
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool removeObject(const std::string& object_id, bool warn_on_failure = true) {
    auto req = std::make_shared<lite_motion_msgs::srv::RemoveSceneObject::Request>();
    req->object_id = object_id;
    auto resp = callService<lite_motion_msgs::srv::RemoveSceneObject>(
      remove_object_client_, req, "remove_scene_object");
    if (!resp || !resp->success) {
      if (warn_on_failure) {
        RCLCPP_WARN(
          this->get_logger(),
          "Remove scene object [%s] failed or skipped: %s",
          object_id.c_str(),
          resp ? resp->message.c_str() : "no response");
      }
      return false;
    }
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool setObjectAttached(
    const std::string& object_id,
    bool attach,
    const std::string& frame_id,
    const geometry_msgs::msg::Pose* relative_pose = nullptr,
    bool warn_on_failure = true) {
    auto req = std::make_shared<lite_motion_msgs::srv::AttachSceneObject::Request>();
    req->object_id = object_id;
    req->attach = attach;
    req->frame_id = frame_id;
    if (relative_pose != nullptr) {
      req->relative_pose = *relative_pose;
    } else {
      req->relative_pose.orientation.w = 1.0;
    }

    auto resp = callService<lite_motion_msgs::srv::AttachSceneObject>(
      attach_object_client_, req, "attach_scene_object");
    if (!resp || !resp->success) {
      if (warn_on_failure) {
        RCLCPP_ERROR(
          this->get_logger(),
          "%s object [%s] failed: %s",
          attach ? "Attach" : "Detach",
          object_id.c_str(),
          resp ? resp->message.c_str() : "no response");
      }
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "%s [%s] %s [%s]",
      attach ? "Attached" : "Detached",
      object_id.c_str(),
      attach ? "to" : "from",
      frame_id.c_str());
    if (object_id == e3_object_id_) {
      e3_attached_to_tcp_ = attach;
      if (attach && relative_pose != nullptr) {
        e3_relative_pose_in_tcp_ = *relative_pose;
      }
      publishReferenceMarkers(attach ? "after_attach_e3" : "after_detach_e3");
      publishE3Marker();
    }
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool setRuntimeSceneCollisionAllowance(
    const std::vector<std::string>& object_ids,
    const std::vector<std::string>& link_names,
    double allowed_penetration_m,
    const std::string& stage_name) {
    auto req = std::make_shared<lite_motion_msgs::srv::SetSceneCollisionAllowance::Request>();
    req->object_ids = object_ids;
    req->link_names = link_names;
    req->allowed_penetration = allowed_penetration_m;

    auto resp = callService<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
      set_scene_collision_allowance_client_, req, "set_scene_collision_allowance");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Set scene collision allowance failed at stage [%s]: %s",
        stage_name.c_str(),
        resp ? resp->message.c_str() : "no response");
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] applied collision allowance: objects=%zu links=%zu allowed_penetration=%.4f m",
      stage_name.c_str(),
      object_ids.size(),
      link_names.size(),
      allowed_penetration_m);
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool setGripperE3CollisionAllowance(double allowed_penetration_m, const std::string& stage_name) {
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{e3_object_id_},
      grasp_collision_links_,
      allowed_penetration_m,
      stage_name);
  }

  double effectiveExecutionWaitSec(const trajectory_msgs::msg::JointTrajectory& traj) const {
    double wait_sec = trajectoryDurationSec(traj);
    if (visualization_playback_enabled_) {
      wait_sec /= visualization_playback_speed_scale_;
    }
    return wait_sec + post_action_wait_sec_ + 0.25;
  }

  double maxJointError(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs) const {
    if (lhs.size() != rhs.size()) {
      return std::numeric_limits<double>::infinity();
    }
    double max_err = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
      max_err = std::max(max_err, std::abs(lhs[i] - rhs[i]));
    }
    return max_err;
  }

  void logArmJointState(const std::string& stage_name, const std::vector<double>& joints) const {
    std::ostringstream oss;
    oss << "Stage [" << stage_name << "] arm joints:";
    const auto& group = robot_model_->getPlanningGroup(arm_group_);
    for (size_t i = 0; i < joints.size() && i < group.joint_names.size(); ++i) {
      oss << " " << group.joint_names[i] << "=" << joints[i];
    }
    RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
  }

  void recordReplaySegment(
    const std::string& stage_name,
    const trajectory_msgs::msg::JointTrajectory& trajectory) {
    if (!replay_after_complete_ || trajectory.points.empty()) {
      return;
    }
    replay_segments_.push_back(ReplaySegment{stage_name, trajectory, e3_attached_to_tcp_});
  }

  trajectory_msgs::msg::JointTrajectory scaledTrajectoryForReplay(
    const trajectory_msgs::msg::JointTrajectory& trajectory) const {
    trajectory_msgs::msg::JointTrajectory out = trajectory;
    out.header.stamp = this->now();
    for (auto& point : out.points) {
      point.time_from_start =
        durationFromSec(pointTimeSec(point) / replay_speed_scale_);
    }
    return out;
  }

  double pointTimeSec(const trajectory_msgs::msg::JointTrajectoryPoint& point) const {
    return static_cast<double>(point.time_from_start.sec) +
           static_cast<double>(point.time_from_start.nanosec) * 1e-9;
  }

  std::vector<double> currentNamedJointPositions(
    const std::vector<std::string>& joint_names) const {
    std::vector<double> positions;
    positions.reserve(joint_names.size());
    const auto& arm_group = robot_model_->getPlanningGroup(arm_group_);
    for (const auto& joint_name : joint_names) {
      const auto arm_it =
        std::find(arm_group.joint_names.begin(), arm_group.joint_names.end(), joint_name);
      if (arm_it != arm_group.joint_names.end()) {
        positions.push_back(
          arm_state_[static_cast<size_t>(std::distance(arm_group.joint_names.begin(), arm_it))]);
        continue;
      }

      const auto gripper_it =
        std::find(gripper_joint_names_.begin(), gripper_joint_names_.end(), joint_name);
      if (gripper_it != gripper_joint_names_.end()) {
        positions.push_back(
          gripper_state_[static_cast<size_t>(std::distance(gripper_joint_names_.begin(), gripper_it))]);
        continue;
      }

      positions.push_back(0.0);
    }
    return positions;
  }

  void applyNamedJointPositions(
    const std::vector<std::string>& joint_names,
    const std::vector<double>& positions) {
    const size_t n = std::min(joint_names.size(), positions.size());
    const auto& arm_group = robot_model_->getPlanningGroup(arm_group_);
    for (size_t i = 0; i < n; ++i) {
      const auto arm_it =
        std::find(arm_group.joint_names.begin(), arm_group.joint_names.end(), joint_names[i]);
      if (arm_it != arm_group.joint_names.end()) {
        arm_state_[static_cast<size_t>(std::distance(arm_group.joint_names.begin(), arm_it))] =
          positions[i];
        continue;
      }

      const auto gripper_it =
        std::find(gripper_joint_names_.begin(), gripper_joint_names_.end(), joint_names[i]);
      if (gripper_it != gripper_joint_names_.end()) {
        gripper_state_[static_cast<size_t>(std::distance(gripper_joint_names_.begin(), gripper_it))] =
          positions[i];
      }
    }
  }

  std::vector<double> interpolatePositions(
    const std::vector<double>& from,
    const std::vector<double>& to,
    double ratio) const {
    const size_t n = std::min(from.size(), to.size());
    std::vector<double> out(n, 0.0);
    ratio = std::clamp(ratio, 0.0, 1.0);
    for (size_t i = 0; i < n; ++i) {
      out[i] = from[i] + (to[i] - from[i]) * ratio;
    }
    return out;
  }

  std::vector<double> sampleTrajectoryPositions(
    const trajectory_msgs::msg::JointTrajectory& trajectory,
    const std::vector<double>& segment_start_positions,
    double elapsed_sec) const {
    if (trajectory.points.empty()) {
      return segment_start_positions;
    }

    if (trajectory.points.size() == 1) {
      const double duration = std::max(0.02, pointTimeSec(trajectory.points.front()));
      return interpolatePositions(
        segment_start_positions,
        trajectory.points.front().positions,
        elapsed_sec / duration);
    }

    if (elapsed_sec <= pointTimeSec(trajectory.points.front())) {
      return trajectory.points.front().positions;
    }

    for (size_t i = 1; i < trajectory.points.size(); ++i) {
      const double t0 = pointTimeSec(trajectory.points[i - 1]);
      const double t1 = pointTimeSec(trajectory.points[i]);
      if (elapsed_sec <= t1) {
        const double denom = std::max(1e-6, t1 - t0);
        return interpolatePositions(
          trajectory.points[i - 1].positions,
          trajectory.points[i].positions,
          (elapsed_sec - t0) / denom);
      }
    }

    return trajectory.points.back().positions;
  }

  std::vector<double> applyArmCommandVelocityLimit(
    const std::vector<double>& desired,
    const std::vector<double>& previous,
    double dt_sec) const {
    if (desired.size() != previous.size() ||
        desired.size() != max_arm_command_velocity_rad_s_.size()) {
      return desired;
    }

    std::vector<double> limited = desired;
    for (size_t i = 0; i < desired.size(); ++i) {
      const double vmax = max_arm_command_velocity_rad_s_[i];
      if (vmax <= 0.0) {
        continue;
      }
      const double max_step = vmax * std::max(1e-3, dt_sec);
      limited[i] = std::clamp(desired[i], previous[i] - max_step, previous[i] + max_step);
    }
    return limited;
  }

  void publishArmCommand(
    const std::vector<std::string>& joint_names,
    const std::vector<double>& positions) {
    if (!arm_command_pub_ || joint_names.size() != positions.size()) {
      return;
    }

    sensor_msgs::msg::JointState msg;
    msg.header.stamp = this->now();
    msg.name = joint_names;
    msg.position = positions;
    arm_command_pub_->publish(msg);
  }

  void publishArmCommandCurrentState(const std::string& stage_name) {
    if (!arm_command_pub_) {
      return;
    }
    const auto& group = robot_model_->getPlanningGroup(arm_group_);
    publishArmCommand(group.joint_names, arm_state_);
    RCLCPP_DEBUG(
      this->get_logger(),
      "Stage [%s] published current arm command.",
      stage_name.c_str());
  }

  bool publishArmCommandTrajectory(
    const trajectory_msgs::msg::JointTrajectory& trajectory,
    const std::string& stage_name) {
    if (!arm_command_pub_ || trajectory.points.empty()) {
      return true;
    }

    const auto segment_start_positions = currentNamedJointPositions(trajectory.joint_names);
    std::vector<double> previous_command = segment_start_positions;
    const auto final_positions = trajectory.points.back().positions;
    const double original_duration = std::max(0.02, trajectoryDurationSec(trajectory));
    const double period_sec = 1.0 / arm_command_publish_hz_;
    const auto started_at = std::chrono::steady_clock::now();
    auto previous_tick = started_at;

    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] publishing arm command samples: hz=%.2f speed_scale=%.3f duration=%.3f s",
      stage_name.c_str(),
      arm_command_publish_hz_,
      arm_command_speed_scale_,
      original_duration);

    publishArmCommand(trajectory.joint_names, previous_command);

    while (!stop_requested_.load()) {
      const auto now_tp = std::chrono::steady_clock::now();
      const double wall_elapsed = std::chrono::duration<double>(now_tp - started_at).count();
      const double dt_sec = std::max(1e-3, std::chrono::duration<double>(now_tp - previous_tick).count());
      previous_tick = now_tp;

      const double trajectory_elapsed =
        std::min(original_duration, wall_elapsed * arm_command_speed_scale_);
      const auto desired = sampleTrajectoryPositions(
        trajectory,
        segment_start_positions,
        trajectory_elapsed);
      previous_command = applyArmCommandVelocityLimit(desired, previous_command, dt_sec);
      publishArmCommand(trajectory.joint_names, previous_command);

      if (trajectory_elapsed >= original_duration &&
          maxJointError(previous_command, final_positions) <= arm_command_goal_tolerance_rad_) {
        break;
      }

      std::this_thread::sleep_for(std::chrono::duration<double>(period_sec));
    }

    publishArmCommand(trajectory.joint_names, final_positions);
    return !stop_requested_.load();
  }

  bool executeArmJointPlan(
    const std::vector<double>& target_joints,
    const std::string& stage_name) {
    if (target_joints.size() == arm_state_.size()) {
      double max_err = 0.0;
      for (size_t i = 0; i < target_joints.size(); ++i) {
        max_err = std::max(max_err, std::abs(target_joints[i] - arm_state_[i]));
      }
      if (max_err < 1e-6) {
        RCLCPP_INFO(this->get_logger(), "Stage [%s] already at target arm joints.", stage_name.c_str());
        arm_state_ = target_joints;
        publishArmCommandCurrentState(stage_name);
        if (!publishGroupHoldTrajectory(arm_group_, target_joints, stage_name, true)) {
          return false;
        }
        publishCurrentTheoreticalJointState(stage_name + "_skip_sync");
        sleepAfterAction(post_action_wait_sec_);
        return true;
      }
    }

    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = arm_group_;
    req->ee_frame = tcp_frame_;
    req->use_pose_goal = false;
    req->start_joint_positions = arm_state_;
    req->goal_joint_positions = target_joints;
    req->plan_only = false;
    req->use_cartesian_path = false;
    req->cartesian_max_step = 0.0;
    req->use_request_solver_settings = true;
    req->request_planning_timeout = joint_planning_timeout_;
    req->request_path_check_step = path_check_step_m_;

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(
      plan_client_, req, "plan_arm_motion");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Arm joint planning failed at stage [%s]: %s",
        stage_name.c_str(),
        resp ? resp->message.c_str() : "no response");
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Arm joint planning returned empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }
    recordReplaySegment(stage_name, resp->trajectory);
    const auto final_arm_state = resp->trajectory.points.back().positions;
    const double target_err = maxJointError(final_arm_state, target_joints);
    logArmJointState(stage_name + "_planned_final", final_arm_state);
    if (target_err > joint_goal_tolerance_rad_) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Stage [%s] final joint state is %.6f rad away from requested target, tolerance %.6f rad.",
        stage_name.c_str(),
        target_err,
        joint_goal_tolerance_rad_);
      return false;
    }
    if (arm_command_pub_) {
      if (!publishArmCommandTrajectory(resp->trajectory, stage_name)) {
        return false;
      }
    } else {
      sleepAfterAction(effectiveExecutionWaitSec(resp->trajectory));
    }
    arm_state_ = final_arm_state;
    if (!publishGroupHoldTrajectory(arm_group_, arm_state_, stage_name + "_endpoint_hold")) {
      return false;
    }
    publishCurrentTheoreticalJointState(stage_name + "_final_sync");
    return true;
  }

  bool executeArmPosePlan(
    const geometry_msgs::msg::PoseStamped& target_pose,
    bool use_cartesian_path,
    const std::string& stage_name) {
    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = arm_group_;
    req->ee_frame = tcp_frame_;
    req->use_pose_goal = true;
    req->target_pose = target_pose;
    req->start_joint_positions = arm_state_;
    req->plan_only = false;
    req->use_cartesian_path = use_cartesian_path;
    req->cartesian_max_step = use_cartesian_path ? cartesian_max_step_m_ : 0.0;

    if (use_cartesian_path && cartesian_use_request_solver_settings_) {
      req->use_request_solver_settings = true;
      req->request_ik_max_iterations = cartesian_ik_max_iterations_;
      req->request_ik_pos_tolerance = cartesian_ik_pos_tolerance_;
      req->request_ik_rot_tolerance = cartesian_ik_rot_tolerance_;
      req->request_ik_damping = cartesian_ik_damping_;
      req->request_ik_alpha = cartesian_ik_alpha_;
      req->request_ik_max_step_norm = cartesian_ik_max_step_norm_;
      req->request_ik_max_seed_count = cartesian_ik_max_seed_count_;
      req->request_planning_timeout = cartesian_planning_timeout_;
      req->request_path_check_step = path_check_step_m_;
      req->request_use_sdls = cartesian_use_sdls_;
      req->request_cartesian_nullspace_gain = cartesian_nullspace_gain_;
      req->request_cartesian_singular_threshold = cartesian_singular_threshold_;
      req->request_cartesian_branch_jump_max_rad = cartesian_branch_jump_max_rad_;
      req->request_cartesian_branch_jump_norm_max = cartesian_branch_jump_norm_max_;
      req->request_cartesian_waypoint_pos_tolerance = cartesian_waypoint_pos_tolerance_;
      req->request_cartesian_waypoint_rot_tolerance = cartesian_waypoint_rot_tolerance_;
      req->request_cartesian_min_interpolation_steps = cartesian_min_interpolation_steps_;
      req->request_cartesian_refinement_max_steps = cartesian_refinement_max_steps_;
    } else {
      req->use_request_solver_settings = true;
      req->request_planning_timeout = joint_planning_timeout_;
      req->request_path_check_step = path_check_step_m_;
    }

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(
      plan_client_, req, "plan_arm_motion");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Arm pose planning failed at stage [%s]: %s",
        stage_name.c_str(),
        resp ? resp->message.c_str() : "no response");
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Arm pose planning returned empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }
    recordReplaySegment(stage_name, resp->trajectory);
    const auto final_arm_state = resp->trajectory.points.back().positions;
    logArmJointState(stage_name + "_planned_final", final_arm_state);
    if (arm_command_pub_) {
      if (!publishArmCommandTrajectory(resp->trajectory, stage_name)) {
        return false;
      }
    } else {
      sleepAfterAction(effectiveExecutionWaitSec(resp->trajectory));
    }
    arm_state_ = final_arm_state;
    if (!publishGroupHoldTrajectory(arm_group_, arm_state_, stage_name + "_endpoint_hold")) {
      return false;
    }
    publishCurrentTheoreticalJointState(stage_name + "_final_sync");
    return true;
  }

  std::vector<double> readInitialArmJointPositionsOrDefault() {
    RCLCPP_INFO(
      this->get_logger(),
      "Serial joint-state input is not available; using tzb.ini_joint_positions as initial arm state.");
    return initial_arm_positions_;
  }

  bool initializeSceneAndMoveToObserve() {
    removeObject(e3_object_id_, false);
    e3_attached_to_tcp_ = false;
    e3_target_world_pose_ = geometry_msgs::msg::PoseStamped();
    pre_grasp_world_pose_ = geometry_msgs::msg::PoseStamped();
    {
      std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
      resetPoseEstimatorCaptureLocked();
      have_frozen_pose_estimator_reference_ = false;
    }

    arm_state_ = readInitialArmJointPositionsOrDefault();
    gripper_state_ = gripper_home_positions_;
    publishCurrentTheoreticalJointState("initial_joint_state");
    publishReferenceMarkers("initial_joint_state");
    publishE3Marker();

    if (!publishGripperTrajectory(gripper_home_positions_, "move_gripper_to_home")) {
      return false;
    }

    if (!executeArmJointPlan(observe_joint_positions_, "move_arm_to_observe_before_perception")) {
      return false;
    }

    if (!use_yaml_e3_target_ && !enablePoseEstimatorCaptureAtCurrentReferencePose()) {
      return false;
    }
    return true;
  }

  bool runSequenceOnce() {
    replay_segments_.clear();
    if (!initializeSceneAndMoveToObserve()) {
      return false;
    }

    if (!resolveE3TargetPose(e3_target_world_pose_)) {
      return false;
    }
    e3_target_world_pose_.header.frame_id = planning_frame_;
    pre_grasp_world_pose_ = preGraspPoseForE3Target(e3_target_world_pose_);
    pre_grasp_world_pose_.header.frame_id = planning_frame_;
    publishReferenceMarkers("after_compute_e3_target_and_pre_grasp");
    publishE3Marker();

    if (!addMeshObject(e3_object_id_, e3_mesh_resource_, e3_target_world_pose_)) {
      return false;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Planning TCP to pre_grasp_point: E3=(%.4f, %.4f, %.4f), pre=(%.4f, %.4f, %.4f), pre_dis=%.4f.",
      e3_target_world_pose_.pose.position.x,
      e3_target_world_pose_.pose.position.y,
      e3_target_world_pose_.pose.position.z,
      pre_grasp_world_pose_.pose.position.x,
      pre_grasp_world_pose_.pose.position.y,
      pre_grasp_world_pose_.pose.position.z,
      pre_grasp_distance_m_);
    if (!executeArmPosePlan(pre_grasp_world_pose_, false, "move_tcp_to_pre_grasp_point")) {
      return false;
    }

    if (!publishGripperTrajectory(gripper_open_positions_, "open_gripper_at_pre_grasp")) {
      return false;
    }

    if (!setGripperE3CollisionAllowance(
          grasp_collision_allowed_penetration_m_,
          "allow_gripper_and_e3_before_final_approach")) {
      return false;
    }

    if (!executeArmPosePlan(e3_target_world_pose_, true, "move_tcp_to_e3_target")) {
      return false;
    }

    if (!publishGripperTrajectory(gripper_close_positions_, "close_gripper_at_e3_target")) {
      return false;
    }

    e3_relative_pose_in_tcp_ = computeAttachRelativePose(tcp_frame_, e3_target_world_pose_);
    if (!setObjectAttached(e3_object_id_, true, tcp_frame_, &e3_relative_pose_in_tcp_)) {
      return false;
    }

    if (!executeArmJointPlan(observe_joint_positions_, "return_attached_e3_to_observe")) {
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "tzb_catch sequence completed successfully.");
    return true;
  }

  void runSequence() {
    if (!waitForService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, "plan_arm_motion") ||
        !waitForService<lite_motion_msgs::srv::AddSceneObject>(add_object_client_, "add_scene_object") ||
        !waitForService<lite_motion_msgs::srv::RemoveSceneObject>(remove_object_client_, "remove_scene_object") ||
        !waitForService<lite_motion_msgs::srv::AttachSceneObject>(attach_object_client_, "attach_scene_object") ||
        !waitForService<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
          set_scene_collision_allowance_client_, "set_scene_collision_allowance")) {
      return;
    }

    do {
      (void)setGripperE3CollisionAllowance(0.0, "best_effort_cleanup_clear_gripper_e3_allowance");
      (void)setObjectAttached(e3_object_id_, false, tcp_frame_, nullptr, false);
      removeObject(e3_object_id_, false);
      if (!runSequenceOnce()) {
        return;
      }
      if (replay_after_complete_ && !replay_segments_.empty()) {
        replayRecordedSequence();
        return;
      }
      if (sequence_loop_ && !stop_requested_.load()) {
        sleepAfterAction(loop_pause_sec_);
      }
    } while (sequence_loop_ && !stop_requested_.load());
  }

  void replayRecordedSequence() {
    RCLCPP_INFO(
      this->get_logger(),
      "Starting tzb_catch recorded trajectory replay: segments=%zu loop=%s speed_scale=%.3f",
      replay_segments_.size(),
      replay_loop_ ? "true" : "false",
      replay_speed_scale_);

    do {
      arm_state_ = initial_arm_positions_;
      gripper_state_ = gripper_home_positions_;
      e3_attached_to_tcp_ = false;
      publishReferenceMarkers("replay_reset");
      publishE3Marker();
      publishCurrentTheoreticalJointState("replay_reset");

      for (const auto& segment : replay_segments_) {
        if (stop_requested_.load()) {
          return;
        }

        e3_attached_to_tcp_ = segment.e3_attached_during_segment;
        publishReferenceMarkers("replay_" + segment.stage_name);
        publishE3Marker();

        const auto scaled_traj = scaledTrajectoryForReplay(segment.trajectory);
        if (trajectory_pub_) {
          trajectory_pub_->publish(scaled_traj);
        }
        replayTrajectorySegment(segment);
        sleepAfterAction(replay_segment_pause_sec_);
      }

      if (replay_loop_ && !stop_requested_.load()) {
        sleepAfterAction(loop_pause_sec_);
      }
    } while (replay_loop_ && !stop_requested_.load());
  }

  void replayTrajectorySegment(const ReplaySegment& segment) {
    if (segment.trajectory.points.empty()) {
      return;
    }

    const auto segment_start_positions =
      currentNamedJointPositions(segment.trajectory.joint_names);
    const double duration = std::max(0.02, trajectoryDurationSec(segment.trajectory));
    const auto started_at = std::chrono::steady_clock::now();

    RCLCPP_INFO(
      this->get_logger(),
      "Replay segment [%s]: joints=%zu duration=%.3f s replay_duration=%.3f s",
      segment.stage_name.c_str(),
      segment.trajectory.joint_names.size(),
      duration,
      duration / replay_speed_scale_);

    while (!stop_requested_.load()) {
      const double wall_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
      const double trajectory_elapsed = std::min(duration, wall_elapsed * replay_speed_scale_);
      const auto positions = sampleTrajectoryPositions(
        segment.trajectory,
        segment_start_positions,
        trajectory_elapsed);
      applyNamedJointPositions(segment.trajectory.joint_names, positions);
      if (arm_command_publish_during_replay_ &&
          segment.trajectory.joint_names == robot_model_->getPlanningGroup(arm_group_).joint_names) {
        publishArmCommand(segment.trajectory.joint_names, positions);
      }
      publishCurrentTheoreticalJointState("replay_" + segment.stage_name);
      publishReferenceMarkers();
      publishE3Marker();

      if (trajectory_elapsed >= duration) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    const auto final_positions = segment.trajectory.points.back().positions;
    applyNamedJointPositions(segment.trajectory.joint_names, final_positions);
    if (arm_command_publish_during_replay_ &&
        segment.trajectory.joint_names == robot_model_->getPlanningGroup(arm_group_).joint_names) {
      publishArmCommand(segment.trajectory.joint_names, final_positions);
    }
    publishCurrentTheoreticalJointState("replay_" + segment.stage_name + "_final");
    publishReferenceMarkers();
    publishE3Marker();
  }

  void sleepAfterAction(double seconds) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!stop_requested_.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  void publishReferenceMarkers(const std::string& stage_name = "") {
    if (!ref_marker_pub_) {
      return;
    }
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);
    int id = 0;
    geometry_msgs::msg::PoseStamped e3_visual_pose;
    if (e3VisualMarkerPose(e3_visual_pose)) {
      appendE3MeshMarker(arr, e3_visual_pose, id);
    }
    geometry_msgs::msg::PoseStamped e3_pose;
    if (currentE3TargetPoseForVisualization(e3_pose)) {
      appendFrameMarkers(arr, e3_pose, "E3_target", id);
      appendFrameMarkers(arr, preGraspPoseForE3Target(e3_pose), "pre_grasp_point", id);
	    }
	    appendFrameMarkers(arr, currentFramePoseStamped(tcp_frame_), tcp_frame_, id);
    if (!pose_estimator_reference_frame_.empty()) {
      try {
        appendFrameMarkers(
          arr,
          currentFramePoseStamped(pose_estimator_reference_frame_),
          pose_estimator_reference_frame_,
          id);
      } catch (const std::exception& e) {
        RCLCPP_DEBUG(
          this->get_logger(),
          "Skipping camera reference markers for [%s]: %s",
          pose_estimator_reference_frame_.c_str(),
          e.what());
      }
    }
    geometry_msgs::msg::PoseStamped shadow_camera_pose;
    if (shadowCameraFramePoseStamped(shadow_camera_pose)) {
      appendFrameMarkers(arr, shadow_camera_pose, frozen_camera_frame_id_, id);
    }
	    ref_marker_pub_->publish(arr);
    if (!stage_name.empty()) {
      RCLCPP_INFO(this->get_logger(), "Published tzb_catch reference markers at stage [%s].", stage_name.c_str());
    }
  }

  void appendE3MeshMarker(
    visualization_msgs::msg::MarkerArray& arr,
    const geometry_msgs::msg::PoseStamped& pose,
    int& id) const {
    visualization_msgs::msg::Marker marker;
    marker = makeE3MeshMarker(pose, id++);
    arr.markers.push_back(marker);
  }

  visualization_msgs::msg::Marker makeE3MeshMarker(
    const geometry_msgs::msg::PoseStamped& pose,
    int id = 0) const {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = pose.header.frame_id;
    marker.header.stamp = this->now();
    marker.ns = "E3_visual";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = pose.pose;
    marker.frame_locked = e3_attached_to_tcp_;
    marker.mesh_resource = markerMeshResource(e3_mesh_resource_);
    marker.mesh_use_embedded_materials = true;
    marker.scale.x = mesh_scale_.size() == 3 ? mesh_scale_[0] : 1.0;
    marker.scale.y = mesh_scale_.size() == 3 ? mesh_scale_[1] : 1.0;
    marker.scale.z = mesh_scale_.size() == 3 ? mesh_scale_[2] : 1.0;
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;
    return marker;
  }

  void publishE3Marker() {
    if (!e3_marker_pub_) {
      return;
    }
    geometry_msgs::msg::PoseStamped pose;
    if (e3VisualMarkerPose(pose)) {
      e3_marker_pub_->publish(makeE3MeshMarker(pose));
      return;
    }

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = planning_frame_;
    marker.header.stamp = this->now();
    marker.ns = "E3_visual";
    marker.id = 0;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    e3_marker_pub_->publish(marker);
  }

  bool currentE3TargetPoseForVisualization(geometry_msgs::msg::PoseStamped& out) {
    if (!e3_target_world_pose_.header.frame_id.empty()) {
      out = e3_target_world_pose_;
      return true;
    }
    if (use_yaml_e3_target_) {
      out = e3TargetPoseInPlanningFrame();
      return true;
    }
    return lockedPoseEstimatorTargetInPlanningFrame(out);
  }

  bool e3VisualMarkerPose(geometry_msgs::msg::PoseStamped& out) {
    if (e3_attached_to_tcp_) {
      out.header.frame_id = tcp_frame_;
      out.pose = e3_relative_pose_in_tcp_;
      return true;
    }

    return currentE3TargetPoseForVisualization(out);
  }

  void appendFrameMarkers(
    visualization_msgs::msg::MarkerArray& arr,
    const geometry_msgs::msg::PoseStamped& pose,
    const std::string& ns,
    int& id) const {
    const Eigen::Quaterniond q = quaternionEigenFromMsg(pose.pose.orientation);
    const Eigen::Matrix3d rot = q.toRotationMatrix();
    const Eigen::Vector3d origin(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
    const std::array<Eigen::Vector3d, 3> axes = {
      rot * Eigen::Vector3d::UnitX(),
      rot * Eigen::Vector3d::UnitY(),
      rot * Eigen::Vector3d::UnitZ()};

    for (size_t i = 0; i < axes.size(); ++i) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = pose.header.frame_id;
      marker.header.stamp = this->now();
      marker.ns = ns + "_axis";
      marker.id = id++;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.scale.x = 0.008;
      marker.scale.y = 0.015;
      marker.scale.z = 0.015;
      marker.color.a = 1.0;
      marker.color.r = (i == 0) ? 1.0f : 0.0f;
      marker.color.g = (i == 1) ? 1.0f : 0.0f;
      marker.color.b = (i == 2) ? 1.0f : 0.0f;
      geometry_msgs::msg::Point p0;
      p0.x = origin.x();
      p0.y = origin.y();
      p0.z = origin.z();
      geometry_msgs::msg::Point p1;
      p1.x = origin.x() + 0.06 * axes[i].x();
      p1.y = origin.y() + 0.06 * axes[i].y();
      p1.z = origin.z() + 0.06 * axes[i].z();
      marker.points.push_back(p0);
      marker.points.push_back(p1);
      arr.markers.push_back(marker);
    }

    visualization_msgs::msg::Marker label;
    label.header.frame_id = pose.header.frame_id;
    label.header.stamp = this->now();
    label.ns = ns + "_label";
    label.id = id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose = pose.pose;
    label.pose.position.z += 0.05;
    label.scale.z = 0.03;
    label.color.r = 1.0f;
    label.color.g = 1.0f;
    label.color.b = 1.0f;
    label.color.a = 1.0f;
    label.text = ns;
    arr.markers.push_back(label);
  }

  std::unique_ptr<RobotModel> robot_model_;
  rclcpp::Client<lite_motion_msgs::srv::PlanArmMotion>::SharedPtr plan_client_;
  rclcpp::Client<lite_motion_msgs::srv::AddSceneObject>::SharedPtr add_object_client_;
  rclcpp::Client<lite_motion_msgs::srv::AttachSceneObject>::SharedPtr attach_object_client_;
  rclcpp::Client<lite_motion_msgs::srv::RemoveSceneObject>::SharedPtr remove_object_client_;
  rclcpp::Client<lite_motion_msgs::srv::SetSceneCollisionAllowance>::SharedPtr
    set_scene_collision_allowance_client_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ref_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr e3_marker_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr arm_command_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frozen_camera_pointcloud_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_estimator_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr camera_pointcloud_sub_;
  rclcpp::TimerBase::SharedPtr start_timer_;
  rclcpp::TimerBase::SharedPtr reference_marker_timer_;
  rclcpp::TimerBase::SharedPtr frozen_camera_tf_timer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> frozen_camera_tf_broadcaster_;

  std::thread worker_;
  std::atomic_bool stop_requested_{false};

  std::string planning_frame_;
  std::string arm_group_;
  std::string tcp_frame_;
  std::string joint_positions_unit_{"rad"};
  std::string e3_object_id_;
  std::string e3_mesh_resource_;
  std::vector<double> mesh_scale_;
  std::string e3_target_frame_;
  std::array<double, 3> e3_target_xyz_{};
  std::array<double, 3> e3_target_rpy_{};
  bool use_yaml_e3_target_{true};
  std::string pose_estimator_target_topic_{"/pose_estimator/object_pose"};
  double pose_estimator_wait_timeout_sec_{3.0};
  double pose_estimator_continuous_required_sec_{2.0};
  double pose_estimator_continuous_max_gap_sec_{0.75};
  std::string pose_estimator_reference_frame_;
  Eigen::Matrix4d pose_estimator_reference_transform_{Eigen::Matrix4d::Identity()};
  bool freeze_camera_pointcloud_enabled_{true};
  std::string camera_pointcloud_topic_{"/camera/depth/points"};
  std::string frozen_camera_pointcloud_topic_{"/tzb_catch/frozen_observe_camera_points"};
  std::string frozen_camera_frame_id_{"tzb_observe_camera_link"};
  bool apply_reference_transform_to_frozen_frame_{true};
  bool transform_frozen_pointcloud_to_reference_frame_{true};
  double frozen_camera_pointcloud_log_throttle_sec_{2.0};
  bool pointcloud_transform_error_logged_{false};
  bool shadow_camera_pose_error_logged_{false};
  std::chrono::steady_clock::time_point last_frozen_pointcloud_log_time_{};
  Eigen::Matrix3d pose_estimator_axis_conversion_{Eigen::Matrix3d::Identity()};
  double pre_grasp_distance_m_{0.10};
  std::vector<std::string> gripper_joint_names_;
  std::vector<double> gripper_home_positions_;
  std::vector<double> gripper_open_positions_;
  std::vector<double> gripper_close_positions_;
  std::vector<std::string> grasp_collision_links_;
  double grasp_collision_allowed_penetration_m_{-1.0};
  double joint_planning_timeout_{2.0};
  double cartesian_planning_timeout_{2.0};
  double path_check_step_m_{0.05};
  double cartesian_max_step_m_{0.02};
  int cartesian_min_interpolation_steps_{2};
  int cartesian_refinement_max_steps_{16};
  bool cartesian_use_request_solver_settings_{true};
  bool cartesian_use_sdls_{true};
  int cartesian_ik_max_iterations_{400};
  double cartesian_ik_pos_tolerance_{0.008};
  double cartesian_ik_rot_tolerance_{0.08};
  double cartesian_ik_damping_{1.0e-6};
  double cartesian_ik_alpha_{0.6};
  double cartesian_ik_max_step_norm_{0.5};
  int cartesian_ik_max_seed_count_{32};
  double cartesian_nullspace_gain_{0.0};
  double cartesian_singular_threshold_{0.03};
  double cartesian_branch_jump_max_rad_{0.75};
  double cartesian_branch_jump_norm_max_{1.20};
  double cartesian_waypoint_pos_tolerance_{0.002};
  double cartesian_waypoint_rot_tolerance_{0.05};
  double post_action_wait_sec_{0.05};
  double service_wait_sec_{30.0};
  double response_wait_sec_{30.0};
  std::string e3_marker_topic_{"/tzb_catch/e3_marker"};
  double reference_marker_republish_hz_{5.0};
  double gripper_motion_duration_sec_{0.35};
  bool enable_visualization_{true};
  bool visualization_playback_enabled_{true};
  double visualization_playback_speed_scale_{1.0};
  bool sequence_loop_{false};
  double loop_pause_sec_{0.5};
  bool replay_after_complete_{true};
  bool replay_loop_{true};
  double replay_speed_scale_{0.35};
  double replay_segment_pause_sec_{0.15};
  double joint_goal_tolerance_rad_{0.01};
  bool arm_command_enabled_{true};
  double arm_command_publish_hz_{20.0};
  double arm_command_speed_scale_{1.0};
  double arm_command_goal_tolerance_rad_{0.005};
  bool arm_command_publish_during_replay_{false};
  std::vector<double> initial_arm_positions_;
  std::vector<double> observe_joint_positions_;
  std::vector<double> max_arm_command_velocity_rad_s_;
  std::vector<double> arm_state_;
  std::vector<double> gripper_state_;
  std::mutex pose_estimator_mutex_;
  std::condition_variable pose_estimator_cv_;
  geometry_msgs::msg::PoseStamped latest_pose_estimator_target_;
  bool have_pose_estimator_target_{false};
  bool pose_estimator_capture_enabled_{false};
  bool pose_estimator_stream_active_{false};
  bool have_frozen_pose_estimator_reference_{false};
  pinocchio::SE3 frozen_pose_estimator_reference_in_planning_{pinocchio::SE3::Identity()};
  std::chrono::steady_clock::time_point pose_estimator_stream_start_time_{};
  std::chrono::steady_clock::time_point pose_estimator_last_msg_time_{};
  geometry_msgs::msg::PoseStamped e3_target_world_pose_;
  geometry_msgs::msg::PoseStamped pre_grasp_world_pose_;
  geometry_msgs::msg::Pose e3_relative_pose_in_tcp_;
  bool e3_attached_to_tcp_{false};
  std::vector<ReplaySegment> replay_segments_;
};

}  // namespace lite_motion_planner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lite_motion_planner::TzbCatchNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
