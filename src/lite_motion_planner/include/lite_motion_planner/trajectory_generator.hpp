#pragma once

#include "lite_motion_planner/robot_model.hpp"

#include <Eigen/Core>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <string>
#include <vector>

namespace lite_motion_planner {

class TrajectoryGenerator {
public:
  explicit TrajectoryGenerator(const RobotModel& robot_model, double control_period_sec = 0.01);

  trajectory_msgs::msg::JointTrajectory generate(
    const std::string& group_name,
    const std::vector<Eigen::VectorXd>& path_points) const;

private:
  const RobotModel& robot_model_;
  double control_period_sec_{0.01};
};

}  // namespace lite_motion_planner
