#pragma once

#include "lite_motion_planner/collision_scene.hpp"
#include "lite_motion_planner/kinematics_solver.hpp"
#include "lite_motion_planner/motion_planner.hpp"
#include "lite_motion_planner/robot_model.hpp"
#include "lite_motion_planner/trajectory_generator.hpp"

#include <lite_motion_msgs/srv/plan_arm_motion.hpp>
#include <lite_motion_msgs/srv/add_scene_object.hpp>
#include <lite_motion_msgs/srv/remove_scene_object.hpp>
#include <lite_motion_msgs/srv/attach_scene_object.hpp>
#include <lite_motion_msgs/srv/check_state_collision.hpp>
#include <lite_motion_msgs/srv/set_scene_collision_allowance.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <unordered_set>

namespace lite_motion_planner {

class PlanningNode : public rclcpp::Node {
public:
  PlanningNode();

private:
  void handlePlanRequest(
    const std::shared_ptr<lite_motion_msgs::srv::PlanArmMotion::Request> request,
    std::shared_ptr<lite_motion_msgs::srv::PlanArmMotion::Response> response);


  void handleAddSceneObject(
    const std::shared_ptr<lite_motion_msgs::srv::AddSceneObject::Request> request,
    std::shared_ptr<lite_motion_msgs::srv::AddSceneObject::Response> response);

  void handleRemoveSceneObject(
    const std::shared_ptr<lite_motion_msgs::srv::RemoveSceneObject::Request> request,
    std::shared_ptr<lite_motion_msgs::srv::RemoveSceneObject::Response> response);

  void handleAttachSceneObject(
    const std::shared_ptr<lite_motion_msgs::srv::AttachSceneObject::Request> request,
    std::shared_ptr<lite_motion_msgs::srv::AttachSceneObject::Response> response);

  void handleCheckStateCollision(
    const std::shared_ptr<lite_motion_msgs::srv::CheckStateCollision::Request> request,
    std::shared_ptr<lite_motion_msgs::srv::CheckStateCollision::Response> response);

  void handleSetSceneCollisionAllowance(
    const std::shared_ptr<lite_motion_msgs::srv::SetSceneCollisionAllowance::Request> request,
    std::shared_ptr<lite_motion_msgs::srv::SetSceneCollisionAllowance::Response> response);

  void handleObservedJointState(const sensor_msgs::msg::JointState::SharedPtr msg);

  void publishVisualizationState(
    const std::string& group_name,
    const Eigen::VectorXd& q_start,
    const Eigen::VectorXd& q_goal,
    const trajectory_msgs::msg::JointTrajectory& traj,
    const std::vector<std::string>& obstacle_ids,
    const std::vector<shape_msgs::msg::SolidPrimitive>& obstacle_shapes,
    const std::vector<geometry_msgs::msg::PoseStamped>& obstacle_poses);

  sensor_msgs::msg::JointState makeJointStateMsg(
    const Eigen::VectorXd& q_full,
    const rclcpp::Time& stamp) const;

  sensor_msgs::msg::JointState makeJointStateMsgFromGroupPoint(
    const std::string& group_name,
    const Eigen::VectorXd& base_q_full,
    const trajectory_msgs::msg::JointTrajectoryPoint& point,
    const trajectory_msgs::msg::JointTrajectory& traj,
    const rclcpp::Time& stamp) const;

  Eigen::VectorXd makeFullStateFromGroupPoint(
    const std::string& group_name,
    const Eigen::VectorXd& base_q_full,
    const trajectory_msgs::msg::JointTrajectoryPoint& point) const;

  moveit_msgs::msg::RobotState makeMoveItRobotStateMsg(
    const Eigen::VectorXd& q_full,
    const rclcpp::Time& stamp,
    bool is_diff = false) const;

  moveit_msgs::msg::RobotTrajectory makeMoveItRobotTrajectoryMsg(
    const trajectory_msgs::msg::JointTrajectory& traj) const;

  moveit_msgs::msg::PlanningScene makePlanningSceneMsg(
    const Eigen::VectorXd& q_full,
    const std::vector<std::string>& obstacle_ids,
    const std::vector<shape_msgs::msg::SolidPrimitive>& obstacle_shapes,
    const std::vector<geometry_msgs::msg::PoseStamped>& obstacle_poses,
    bool is_diff) const;

  moveit_msgs::msg::CollisionObject makeCollisionObjectMsg(
    const std::string& object_id,
    const shape_msgs::msg::SolidPrimitive& primitive,
    const geometry_msgs::msg::PoseStamped& pose_stamped) const;

  moveit_msgs::msg::CollisionObject makeCollisionObjectMeshMsg(
    const std::string& object_id,
    const std::string& mesh_resource,
    const std::array<double, 3>& mesh_scale,
    const geometry_msgs::msg::PoseStamped& pose_stamped) const;

  shape_msgs::msg::Mesh loadMeshMsg(
    const std::string& mesh_resource,
    const std::array<double, 3>& mesh_scale) const;

  moveit_msgs::msg::AttachedCollisionObject makeAttachedCollisionObjectMsg(
    const CollisionScene::SceneObjectInfo& info) const;

  void publishInitialVisualizationState();
  void publishCurrentPlanningScene(bool is_diff);
  void logSceneObjectAttachmentState(const std::string& object_id) const;
  void publishCollisionMarkers(const Eigen::VectorXd& q_full);
  void publishE3Marker();
  void collisionMarkerTimerTick();
  void e3MarkerTimerTick();
  std::vector<std::string> inferTouchLinks(const std::string& frame_id) const;

  void startJointStatePlayback(
    const std::string& group_name,
    const Eigen::VectorXd& q_start_full,
    const trajectory_msgs::msg::JointTrajectory& traj);

  void stopJointStatePlayback();

  void playbackTimerTick();

  void publishPlaybackPlanningSceneDiff(const Eigen::VectorXd& q_full);

  double pointTimeSec(const trajectory_msgs::msg::JointTrajectoryPoint& point) const;

  std::unique_ptr<RobotModel> robot_model_;
  std::unique_ptr<KinematicsSolver> kinematics_solver_;
  std::unique_ptr<CollisionScene> collision_scene_;
  std::unique_ptr<MotionPlanner> motion_planner_;
  std::unique_ptr<TrajectoryGenerator> trajectory_generator_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_traj_pub_;
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_pub_;
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr monitored_planning_scene_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr collision_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr e3_marker_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Service<lite_motion_msgs::srv::PlanArmMotion>::SharedPtr service_;
  rclcpp::Service<lite_motion_msgs::srv::AddSceneObject>::SharedPtr add_scene_object_service_;
  rclcpp::Service<lite_motion_msgs::srv::RemoveSceneObject>::SharedPtr remove_scene_object_service_;
  rclcpp::Service<lite_motion_msgs::srv::AttachSceneObject>::SharedPtr attach_scene_object_service_;
  rclcpp::Service<lite_motion_msgs::srv::CheckStateCollision>::SharedPtr check_state_collision_service_;
  rclcpp::Service<lite_motion_msgs::srv::SetSceneCollisionAllowance>::SharedPtr set_scene_collision_allowance_service_;
  rclcpp::CallbackGroup::SharedPtr planning_cb_group_;

  // playback state
  rclcpp::TimerBase::SharedPtr playback_timer_;
  rclcpp::TimerBase::SharedPtr collision_marker_timer_;
  rclcpp::TimerBase::SharedPtr e3_marker_timer_;
  trajectory_msgs::msg::JointTrajectory playback_traj_;
  Eigen::VectorXd playback_base_q_full_;
  std::string playback_group_name_;
  rclcpp::Time playback_started_at_{0, 0, RCL_ROS_TIME};
  bool playback_active_{false};
  bool playback_loop_{false};
  double playback_speed_scale_{1.0};
  size_t playback_last_index_{0};
  Eigen::VectorXd current_visual_q_full_;

  // Avoid repeatedly publishing REMOVE diffs for an attached object that RViz
  // has already removed from the world scene. Repeated REMOVE diffs are valid
  // but MoveIt/RViz prints a warning every cycle, flooding the terminal.
  mutable std::unordered_set<std::string> world_remove_published_for_attached_ids_;
};

}  // namespace lite_motion_planner
