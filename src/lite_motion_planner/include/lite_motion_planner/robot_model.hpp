#pragma once

#include "lite_motion_planner/common.hpp"

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyxml2 {
class XMLDocument;
class XMLElement;
}

namespace lite_motion_planner {

class RobotModel {
public:
  RobotModel(
    const std::string& urdf_path,
    const std::string& srdf_path,
    const std::vector<std::string>& mesh_package_dirs);

  const pinocchio::Model& model() const { return model_; }
  pinocchio::Data& data() { return data_; }
  const pinocchio::Data& data() const { return data_; }

  const std::string& urdfPath() const { return urdf_path_; }
  const std::vector<std::string>& meshPackageDirs() const { return mesh_package_dirs_; }

  const PlanningGroupInfo& getPlanningGroup(const std::string& group_name) const;
  bool hasPlanningGroup(const std::string& group_name) const;
  const std::map<std::string, PlanningGroupInfo>& planningGroups() const { return planning_groups_; }
  const AllowedCollisionMatrix& allowedCollisionMatrix() const { return allowed_collision_matrix_; }

  Eigen::VectorXd neutralConfiguration() const;
  Eigen::VectorXd lowerBoundsForGroup(const std::string& group_name) const;
  Eigen::VectorXd upperBoundsForGroup(const std::string& group_name) const;
  Eigen::VectorXd velocityLimitsForGroup(const std::string& group_name) const;
  Eigen::VectorXd accelerationLimitsForGroup(const std::string& group_name) const;
  Eigen::VectorXd jerkLimitsForGroup(const std::string& group_name) const;

  Eigen::VectorXd fullToGroup(const std::string& group_name, const Eigen::VectorXd& q_full) const;
  Eigen::VectorXd groupToFull(const std::string& group_name, const Eigen::VectorXd& q_group, const Eigen::VectorXd& seed_full) const;

  int qIndexOfJoint(const std::string& joint_name) const;
  int vIndexOfJoint(const std::string& joint_name) const;
  pinocchio::FrameIndex getFrameIdChecked(const std::string& frame_name) const;
  std::vector<std::string> activeJointNames() const;
  std::string robotName() const;

private:
  struct UrdfJointInfo {
    std::string name;
    std::string type;
    std::string parent_link;
    std::string child_link;
  };

  void loadPinocchioModel();
  void parseUrdfStructure();
  void parseSrdf();
  void parseGroups(tinyxml2::XMLElement* robot_el);
  void parseAllowedCollisions(tinyxml2::XMLElement* robot_el);
  PlanningGroupInfo resolveChainGroup(
    const std::string& group_name,
    const std::string& base_link,
    const std::string& tip_link) const;
  void buildJointIndexMaps();

  std::string urdf_path_;
  std::string srdf_path_;
  std::vector<std::string> mesh_package_dirs_;

  pinocchio::Model model_;
  pinocchio::Data data_;

  std::map<std::string, PlanningGroupInfo> planning_groups_;
  AllowedCollisionMatrix allowed_collision_matrix_;

  std::unordered_map<std::string, UrdfJointInfo> urdf_joints_by_name_;
  std::unordered_map<std::string, std::string> child_link_to_joint_;
  std::unordered_map<std::string, JointLimit> joint_limits_;
  std::unordered_map<std::string, int> joint_to_qidx_;
  std::unordered_map<std::string, int> joint_to_vidx_;
};

}  // namespace lite_motion_planner
