#pragma once

#include "lite_motion_planner/robot_model.hpp"

#include <Eigen/Core>
#include <pinocchio/spatial/se3.hpp>

#include <atomic>
#include <string>
#include <utility>
#include <vector>

namespace lite_motion_planner {

class KinematicsSolver {
public:
  explicit KinematicsSolver(const RobotModel& robot_model);

  pinocchio::SE3 solveFK(
    const std::string& group_name,
    const Eigen::VectorXd& q_group,
    const std::string& ee_frame = "") const;

  std::pair<bool, Eigen::VectorXd> solveIK(
    const pinocchio::SE3& target_pose,
    const std::string& group_name,
    const Eigen::VectorXd& q_seed_group,
    const std::string& ee_frame = "",
    int max_iterations = 300,
    double pos_eps = 5e-3,
    double rot_eps = 5e-2,
    double damping = 1e-4,
    double alpha = 0.2,
    double max_step_norm = 0.2,
    const std::atomic_bool* cancel = nullptr,
    bool use_sdls = false) const;

  std::pair<bool, Eigen::VectorXd> solveIKWeighted(
    const pinocchio::SE3& target_pose,
    const std::string& group_name,
    const Eigen::VectorXd& q_seed_group,
    const Eigen::Matrix<double, 6, 1>& task_weights,
    const std::string& ee_frame = "",
    int max_iterations = 300,
    double pos_eps = 5e-3,
    double weighted_rot_eps = 5e-2,
    double damping = 1e-4,
    double alpha = 0.2,
    double max_step_norm = 0.2,
    const std::atomic_bool* cancel = nullptr,
    bool use_sdls = false) const;

  struct IKCandidate {
  Eigen::VectorXd q_group;
  double distance_to_seed{0.0};
};

  struct IKDiagnostics {
    double min_singular_value{0.0};
    double max_singular_value{0.0};
    double condition_number{0.0};
    double manipulability{0.0};
    double final_position_error{0.0};
    double final_rotation_error{0.0};
    int iterations{0};
    bool near_singular{false};
    bool joint_limit_clamped{false};
    int joint_limit_index{-1};
    double joint_limit_value{0.0};
    double joint_limit_lower{0.0};
    double joint_limit_upper{0.0};
  };

  std::vector<IKCandidate> solveIKMultiSeed(
    const pinocchio::SE3& target_pose,
    const std::string& group_name,
    const std::vector<Eigen::VectorXd>& q_seed_groups,
    const std::string& ee_frame = "",
    int max_iterations = 300,
    double pos_eps = 5e-3,
    double rot_eps = 5e-2,
    double damping = 1e-4,
    double alpha = 0.2,
    double max_step_norm = 0.2,
    const std::atomic_bool* cancel = nullptr,
    bool use_sdls = false) const;

  std::vector<IKCandidate> solveIKWeightedMultiSeed(
    const pinocchio::SE3& target_pose,
    const std::string& group_name,
    const std::vector<Eigen::VectorXd>& q_seed_groups,
    const Eigen::Matrix<double, 6, 1>& task_weights,
    const std::string& ee_frame = "",
    int max_iterations = 300,
    double pos_eps = 5e-3,
    double weighted_rot_eps = 5e-2,
    double damping = 1e-4,
    double alpha = 0.2,
    double max_step_norm = 0.2,
    const std::atomic_bool* cancel = nullptr,
    bool use_sdls = false) const;

  std::pair<bool, Eigen::VectorXd> solveIKContinuous(
    const pinocchio::SE3& target_pose,
    const std::string& group_name,
    const Eigen::VectorXd& q_seed_group,
    const std::string& ee_frame = "",
    int max_iterations = 300,
    double pos_eps = 5e-3,
    double rot_eps = 5e-2,
    double damping = 1e-4,
    double alpha = 0.2,
    double max_step_norm = 0.2,
    bool use_sdls = false,
    double nullspace_gain = 0.0,
    double singular_threshold = 0.04,
    const std::atomic_bool* cancel = nullptr,
    IKDiagnostics* diagnostics = nullptr) const;

private:
  const RobotModel& robot_model_;
};

}  // namespace lite_motion_planner
