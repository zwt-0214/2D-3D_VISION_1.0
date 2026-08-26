#include "lite_motion_planner/motion_planner.hpp"

#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace ob = ompl::base;
namespace og = ompl::geometric;

namespace lite_motion_planner {
namespace {

double maxAbsJointError(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double max_error = 0.0;
  for (Eigen::Index i = 0; i < lhs.size(); ++i) {
    max_error = std::max(max_error, std::abs(lhs[i] - rhs[i]));
  }
  return max_error;
}

}  // namespace

class MotionPlanner::StateValidityChecker : public ob::StateValidityChecker {
public:
  StateValidityChecker(
    const ob::SpaceInformationPtr& si,
    const RobotModel& robot_model,
    const CollisionScene& collision_scene,
    std::string group_name,
    Eigen::VectorXd base_q_full)
  : ob::StateValidityChecker(si),
    robot_model_(robot_model),
    collision_scene_(collision_scene),
    group_name_(std::move(group_name)),
    base_q_full_(std::move(base_q_full)) {}

  bool isValid(const ob::State* state) const override {
    const auto* rv_state = state->as<ob::RealVectorStateSpace::StateType>();
    const auto& group = robot_model_.getPlanningGroup(group_name_);

    Eigen::VectorXd q_group(static_cast<Eigen::Index>(group.joint_names.size()));
    for (size_t i = 0; i < group.joint_names.size(); ++i) {
      q_group[static_cast<Eigen::Index>(i)] = rv_state->values[i];
    }

    Eigen::VectorXd q_full = robot_model_.groupToFull(group_name_, q_group, base_q_full_);
    return collision_scene_.isStateValid(q_full);
  }

private:
  const RobotModel& robot_model_;
  const CollisionScene& collision_scene_;
  std::string group_name_;
  Eigen::VectorXd base_q_full_;
};

MotionPlanner::MotionPlanner(const RobotModel& robot_model, const CollisionScene& collision_scene)
: robot_model_(robot_model), collision_scene_(collision_scene) {}

MotionPlanner::PlanResult MotionPlanner::plan(
  const std::string& group_name,
  const Eigen::VectorXd& start_group,
  const Eigen::VectorXd& goal_group,
  const Eigen::VectorXd& base_q_full,
  double timeout_sec,
  double path_check_step,
  int max_plan_attempts) const
{
  const auto& group = robot_model_.getPlanningGroup(group_name);

  auto space = std::make_shared<ob::RealVectorStateSpace>(group.joint_names.size());
  ob::RealVectorBounds bounds(group.joint_names.size());
  const auto lower = robot_model_.lowerBoundsForGroup(group_name);
  const auto upper = robot_model_.upperBoundsForGroup(group_name);
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    bounds.setLow(i, lower[static_cast<Eigen::Index>(i)]);
    bounds.setHigh(i, upper[static_cast<Eigen::Index>(i)]);
  }
  space->setBounds(bounds);

  og::SimpleSetup ss(space);
  ss.setStateValidityChecker(std::make_shared<StateValidityChecker>(
    ss.getSpaceInformation(), robot_model_, collision_scene_, group_name, base_q_full));
  ss.getSpaceInformation()->setStateValidityCheckingResolution(0.02);
  ss.setPlanner(std::make_shared<og::RRTConnect>(ss.getSpaceInformation()));

  ob::ScopedState<ob::RealVectorStateSpace> start(space), goal(space);
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    start[i] = start_group[static_cast<Eigen::Index>(i)];
    goal[i] = goal_group[static_cast<Eigen::Index>(i)];
  }

  const Eigen::VectorXd q_start_full = robot_model_.groupToFull(group_name, start_group, base_q_full);
  const Eigen::VectorXd q_goal_full = robot_model_.groupToFull(group_name, goal_group, base_q_full);
  if (!collision_scene_.isStateValid(q_start_full)) {
    const auto report = collision_scene_.getCollisionReport(q_start_full);
    return {false, "start_state_in_collision_or_invalid: " + report.category + " [" + report.object_a + " vs " + report.object_b + "] " + report.detail, {}};
  }
  if (!collision_scene_.isStateValid(q_goal_full)) {
    const auto report = collision_scene_.getCollisionReport(q_goal_full);
    return {false, "goal_state_in_collision_or_invalid: " + report.category + " [" + report.object_a + " vs " + report.object_b + "] " + report.detail, {}};
  }

  const int bounded_max_plan_attempts = std::max(1, max_plan_attempts);
  bool saw_solution = false;
  bool saw_approximate_solution = false;
  std::string last_collision_reason;
  std::string last_approximate_reason;
  for (int attempt = 0; attempt < bounded_max_plan_attempts; ++attempt) {
    ss.clear();
    ss.setStartAndGoalStates(start, goal);
    const auto solved = ss.solve(timeout_sec);
    if (!solved || !ss.haveSolutionPath()) {
      continue;
    }
    saw_solution = true;

    auto& path = ss.getSolutionPath();
    og::PathSimplifier simplifier(ss.getSpaceInformation());
    simplifier.simplifyMax(path);
    if (path.getStateCount() < 2) {
      path.interpolate(2);
    }

    std::vector<Eigen::VectorXd> points;
    points.reserve(path.getStateCount());
    for (size_t i = 0; i < path.getStateCount(); ++i) {
      const auto* s = path.getState(i)->as<ob::RealVectorStateSpace::StateType>();
      Eigen::VectorXd q(static_cast<Eigen::Index>(group.joint_names.size()));
      for (size_t j = 0; j < group.joint_names.size(); ++j) {
        q[static_cast<Eigen::Index>(j)] = s->values[j];
      }
      points.push_back(q);
    }

    const double final_goal_error = maxAbsJointError(points.back(), goal_group);
    constexpr double kExactGoalToleranceRad = 1.0e-4;
    if (final_goal_error > kExactGoalToleranceRad) {
      saw_approximate_solution = true;
      std::ostringstream oss;
      oss << "approximate_solution_rejected: final joint max error "
          << final_goal_error << " rad exceeds "
          << kExactGoalToleranceRad << " rad";
      last_approximate_reason = oss.str();
      continue;
    }

    std::string collision_reason;
    if (isPathCollisionFree(group_name, points, base_q_full, path_check_step, &collision_reason)) {
      return {true, "success", points};
    }
    if (!collision_reason.empty()) {
      last_collision_reason = collision_reason;
    }
  }

  if (saw_solution) {
    if (!last_collision_reason.empty()) {
      return {false, "path_collision_validation_failed_after_retries: " + last_collision_reason, {}};
    }
    if (saw_approximate_solution && !last_approximate_reason.empty()) {
      return {false, last_approximate_reason, {}};
    }
    return {false, "path_collision_validation_failed_after_retries", {}};
  }
  return {false, "planning_timeout", {}};
}

bool MotionPlanner::isPathCollisionFree(
  const std::string& group_name,
  const std::vector<Eigen::VectorXd>& path,
  const Eigen::VectorXd& base_q_full,
  double max_segment_step,
  std::string* collision_reason,
  bool thread_safe) const
{
  if (path.empty()) {
    if (collision_reason != nullptr) {
      *collision_reason = "empty_path";
    }
    return false;
  }

  for (size_t i = 0; i < path.size(); ++i) {
    const Eigen::VectorXd q_full = robot_model_.groupToFull(group_name, path[i], base_q_full);
    const auto waypoint_report = thread_safe ?
      collision_scene_.getCollisionReportThreadSafe(q_full) :
      collision_scene_.getCollisionReport(q_full);
    if (waypoint_report.in_collision) {
      if (collision_reason != nullptr) {
        *collision_reason =
          "path_waypoint_in_collision idx=" + std::to_string(i) + ": " +
          waypoint_report.category + " [" + waypoint_report.object_a + " vs " + waypoint_report.object_b + "] " +
          waypoint_report.detail;
      }
      return false;
    }

    if (i == 0) {
      continue;
    }

    const Eigen::VectorXd delta = path[i] - path[i - 1];
    const double dist = delta.norm();
    const int substeps = std::max(1, static_cast<int>(std::ceil(dist / std::max(1e-6, max_segment_step))));
    for (int s = 1; s < substeps; ++s) {
      const double t = static_cast<double>(s) / static_cast<double>(substeps);
      const Eigen::VectorXd q_interp = path[i - 1] + t * delta;
      const Eigen::VectorXd q_full_interp = robot_model_.groupToFull(group_name, q_interp, base_q_full);
      const auto interp_report = thread_safe ?
        collision_scene_.getCollisionReportThreadSafe(q_full_interp) :
        collision_scene_.getCollisionReport(q_full_interp);
      if (interp_report.in_collision) {
        if (collision_reason != nullptr) {
          *collision_reason =
            "path_interp_in_collision seg=" + std::to_string(i - 1) + "->" + std::to_string(i) +
            " substep=" + std::to_string(s) + "/" + std::to_string(substeps) + ": " +
            interp_report.category + " [" + interp_report.object_a + " vs " + interp_report.object_b + "] " +
            interp_report.detail;
        }
        return false;
      }
    }
  }
  return true;
}

}  // namespace lite_motion_planner
