#pragma once

#include <Eigen/Core>
#include <pinocchio/spatial/se3.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace lite_motion_planner {

struct JointLimit {
  double lower{0.0};
  double upper{0.0};
  double velocity{0.0};
  double acceleration{0.0};
  double jerk{0.0};
};

struct PlanningGroupInfo {
  std::string name;
  std::string base_link;
  std::string tip_link;
  std::vector<std::string> joint_names;
  std::vector<std::string> link_names;
};

struct PlanRequestData {
  std::string group_name;
  std::string ee_frame;
  bool use_pose_goal{false};
  pinocchio::SE3 target_pose;
  Eigen::VectorXd start;
  Eigen::VectorXd goal;
};

using AllowedCollisionMatrix = std::set<std::pair<std::string, std::string>>;

inline std::pair<std::string, std::string> makeOrderedLinkPair(
  const std::string& a,
  const std::string& b)
{
  return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

}  // namespace lite_motion_planner
