#pragma once

#include "lite_motion_planner/collision_scene.hpp"

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace ompl {
namespace base {
class State;
class SpaceInformation;
}
}  // namespace ompl

namespace lite_motion_planner {

class MotionPlanner {
public:
  struct PlanResult {
    bool success{false};
    std::string reason;
    std::vector<Eigen::VectorXd> path;
  };

  MotionPlanner(const RobotModel& robot_model, const CollisionScene& collision_scene);

  PlanResult plan(
    const std::string& group_name,
    const Eigen::VectorXd& start_group,
    const Eigen::VectorXd& goal_group,
    const Eigen::VectorXd& base_q_full,
    double timeout_sec = 2.0,
    double path_check_step = 0.05,
    int max_plan_attempts = 4) const;

  bool isPathCollisionFree(
    const std::string& group_name,
    const std::vector<Eigen::VectorXd>& path,
    const Eigen::VectorXd& base_q_full,
    double max_segment_step = 0.05,
    std::string* collision_reason = nullptr,
    bool thread_safe = false) const;

private:
  class StateValidityChecker;

  const RobotModel& robot_model_;
  const CollisionScene& collision_scene_;
};

}  // namespace lite_motion_planner
