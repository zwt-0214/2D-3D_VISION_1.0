#include "lite_motion_planner/trajectory_generator.hpp"

#include <builtin_interfaces/msg/duration.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <algorithm>
#include <cmath>

namespace lite_motion_planner {
namespace {

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

}  // namespace

TrajectoryGenerator::TrajectoryGenerator(
  const RobotModel& robot_model,
  double control_period_sec)
: robot_model_(robot_model),
  control_period_sec_(std::max(1e-3, control_period_sec)) {}

trajectory_msgs::msg::JointTrajectory TrajectoryGenerator::generate(
  const std::string& group_name,
  const std::vector<Eigen::VectorXd>& path_points) const {
  trajectory_msgs::msg::JointTrajectory traj;
  const auto& group = robot_model_.getPlanningGroup(group_name);
  traj.joint_names = group.joint_names;
  if (path_points.empty()) {
    return traj;
  }

  const Eigen::VectorXd velocity_limits = robot_model_.velocityLimitsForGroup(group_name);
  double elapsed = 0.0;
  Eigen::VectorXd previous = path_points.front();

  for (size_t i = 0; i < path_points.size(); ++i) {
    const Eigen::VectorXd& q = path_points[i];
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.resize(static_cast<size_t>(q.size()));
    point.velocities.assign(static_cast<size_t>(q.size()), 0.0);
    point.accelerations.assign(static_cast<size_t>(q.size()), 0.0);

    for (Eigen::Index j = 0; j < q.size(); ++j) {
      point.positions[static_cast<size_t>(j)] = q[j];
    }

    if (i > 0) {
      double segment_duration = control_period_sec_;
      for (Eigen::Index j = 0; j < q.size() && j < velocity_limits.size(); ++j) {
        const double vmax = std::max(1e-3, std::abs(velocity_limits[j]));
        segment_duration = std::max(segment_duration, std::abs(q[j] - previous[j]) / vmax);
      }
      elapsed += std::max(control_period_sec_, segment_duration);
      previous = q;
    }

    point.time_from_start = durationFromSec(elapsed);
    traj.points.push_back(std::move(point));
  }

  return traj;
}

}  // namespace lite_motion_planner
