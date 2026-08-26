#pragma once

#include "lite_motion_planner/robot_model.hpp"

#include <hpp/fcl/collision.h>
#include <hpp/fcl/collision_object.h>
#include <pinocchio/multibody/geometry.hpp>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <array>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace lite_motion_planner {

class CollisionScene {
public:
  struct SceneObjectInfo {
    std::string id;
    bool uses_mesh{false};
    std::string mesh_resource;
    std::array<double, 3> mesh_scale{{1.0, 1.0, 1.0}};
    shape_msgs::msg::SolidPrimitive primitive;
    geometry_msgs::msg::Pose world_pose;
    bool attached{false};
    std::string frame_id;
    geometry_msgs::msg::Pose relative_pose;
    std::vector<std::string> touch_links;
  };

  explicit CollisionScene(const RobotModel& robot_model);

  struct CollisionReport {
    bool in_collision{false};
    std::string category;
    std::string object_a;
    std::string object_b;
    std::string detail;
  };

  bool checkSelfCollision(const Eigen::VectorXd& q_full) const;
  bool checkEnvironmentCollision(const Eigen::VectorXd& q_full) const;
  bool isStateValid(const Eigen::VectorXd& q_full) const;
  CollisionReport getCollisionReport(const Eigen::VectorXd& q_full) const;
  CollisionReport getCollisionReportThreadSafe(const Eigen::VectorXd& q_full) const;

  void clearTransientObstacles();
  void addTransientObstacleFromPrimitive(
    const std::string& id,
    const shape_msgs::msg::SolidPrimitive& primitive,
    const geometry_msgs::msg::Pose& pose);

  bool upsertSceneObject(
    const std::string& id,
    const std::string& mesh_resource,
    const std::vector<double>& mesh_scale,
    const shape_msgs::msg::SolidPrimitive& primitive,
    const geometry_msgs::msg::Pose& pose);
  bool removeSceneObject(const std::string& id);
  bool setObjectAttached(
    const std::string& id,
    bool attach,
    const std::string& frame_id,
    const geometry_msgs::msg::Pose& relative_pose,
    const std::vector<std::string>& touch_links);
  std::vector<SceneObjectInfo> sceneObjectInfos() const;

  const pinocchio::GeometryModel& collisionModel() const { return collision_model_; }
  pinocchio::GeometryData& collisionData() { return collision_data_; }
  const pinocchio::GeometryData& collisionData() const { return collision_data_; }
  void updateGeometry(const Eigen::VectorXd& q_full) const;
  void setSceneObjectLinkCollisionAllowance(
    const std::string& object_id,
    const std::vector<std::string>& link_names,
    double allowed_penetration);
  void setPersistentSceneObjectLinkCollisionAllowance(
    const std::string& object_id,
    const std::vector<std::string>& link_names,
    double allowed_penetration);
  void setSelfCollisionAllowedPenetration(double allowed_penetration);
  void setGlobalCollisionAllowedPenetration(double allowed_penetration);
  double globalCollisionAllowedPenetration() const { return global_collision_allowed_penetration_; }
  void setRobotLinkCollisionAllowance(
    const std::vector<std::string>& link_names,
    double allowed_penetration);
  void setBaseGroupCollisionAllowance(
    const std::vector<std::string>& link_names,
    double allowed_penetration);
  void setGlobalIgnoredRobotCollisionLinks(
    const std::vector<std::string>& link_names,
    bool ignore);

private:
  struct RelaxedSceneCollisionPair {
    std::string object_id;
    std::string link_name;
    double allowed_penetration{0.0};
  };

  struct Obstacle {
    std::string id;
    std::shared_ptr<hpp::fcl::CollisionGeometry> geometry;
    std::shared_ptr<hpp::fcl::CollisionObject> object;
  };

  struct SceneObject {
    std::string id;
    bool uses_mesh{false};
    std::string mesh_resource;
    std::array<double, 3> mesh_scale{{1.0, 1.0, 1.0}};
    shape_msgs::msg::SolidPrimitive primitive;
    geometry_msgs::msg::Pose world_pose;
    bool attached{false};
    std::string frame_id;
    geometry_msgs::msg::Pose relative_pose;
    std::unordered_set<std::string> touch_links;
    std::shared_ptr<hpp::fcl::CollisionGeometry> geometry;
    pinocchio::JointIndex parent_joint{0};
    pinocchio::FrameIndex parent_frame{0};
    pinocchio::SE3 geometry_placement{pinocchio::SE3::Identity()};
    bool active_in_collision_model{false};
  };

  static hpp::fcl::Transform3f poseMsgToFclTransform(const geometry_msgs::msg::Pose& pose);
  static std::shared_ptr<hpp::fcl::CollisionGeometry> primitiveToGeometry(
    const shape_msgs::msg::SolidPrimitive& primitive);

  bool isAllowedCollisionPair(const std::string& a, const std::string& b) const;
  double allowedPenetrationForPair(
    const SceneObject* scene_a,
    const SceneObject* scene_b,
    const std::string& link_a,
    const std::string& link_b) const;
  double selfCollisionAllowedPenetrationForPair(
    const std::string& link_a,
    const std::string& link_b) const;
  CollisionReport getCollisionReportWithData(
    const Eigen::VectorXd& q_full,
    pinocchio::Data& model_data,
    pinocchio::GeometryData& geometry_data) const;
  void updateRobotGeometry(const Eigen::VectorXd& q_full) const;
  void refreshGeometryData();
  void activateSceneObject(SceneObject& object);
  void deactivateSceneObject(SceneObject& object);
  const SceneObject* findSceneObjectByGeometryName(const std::string& geometry_name) const;
  geometry_msgs::msg::Pose computeAttachedWorldPose(const SceneObject& object) const;

  const RobotModel& robot_model_;
  pinocchio::GeometryModel collision_model_;
  mutable pinocchio::GeometryData collision_data_;
  mutable Eigen::VectorXd last_q_full_;
  std::vector<Obstacle> transient_obstacles_;
  std::vector<RelaxedSceneCollisionPair> relaxed_scene_collision_pairs_;
  std::vector<RelaxedSceneCollisionPair> persistent_scene_collision_pairs_;
  double global_collision_allowed_penetration_{0.0};
  double self_collision_allowed_penetration_{0.0};
  std::unordered_set<std::string> robot_link_collision_allowance_links_;
  double robot_link_collision_allowed_penetration_{0.0};
  std::unordered_set<std::string> base_group_collision_allowance_links_;
  double base_group_collision_allowed_penetration_{0.0};
  std::unordered_set<std::string> globally_ignored_robot_collision_links_;
  std::vector<SceneObject> scene_objects_;
};

}  // namespace lite_motion_planner
