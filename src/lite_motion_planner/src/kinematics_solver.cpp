#include "lite_motion_planner/kinematics_solver.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/spatial/log.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <Eigen/SVD>

namespace lite_motion_planner {
namespace {

Eigen::VectorXd solveDampedLeastSquaresStep(
  const Eigen::MatrixXd& jacobian,
  const Eigen::VectorXd& error,
  double damping) {
  const Eigen::Index task_dim = jacobian.rows();
  const Eigen::MatrixXd A =
    jacobian * jacobian.transpose() +
    damping * Eigen::MatrixXd::Identity(task_dim, task_dim);
  return -jacobian.transpose() * A.ldlt().solve(error);
}

Eigen::MatrixXd dampedPseudoinverse(
  const Eigen::MatrixXd& jacobian,
  double damping) {
  const Eigen::Index task_dim = jacobian.rows();
  const Eigen::MatrixXd A =
    jacobian * jacobian.transpose() +
    damping * Eigen::MatrixXd::Identity(task_dim, task_dim);
  return jacobian.transpose() * A.ldlt().solve(Eigen::MatrixXd::Identity(task_dim, task_dim));
}

void clampMaxAbsInPlace(Eigen::VectorXd& v, double limit) {
  const double safe_limit = std::max(0.0, limit);
  if (safe_limit <= 0.0 || v.size() == 0) {
    v.setZero();
    return;
  }
  const double max_abs = v.cwiseAbs().maxCoeff();
  if (max_abs > safe_limit && max_abs > 1e-12) {
    v *= safe_limit / max_abs;
  }
}

Eigen::VectorXd solveSelectivelyDampedLeastSquaresStep(
  const Eigen::MatrixXd& jacobian,
  const Eigen::VectorXd& error,
  double max_step_norm) {
  const Eigen::Index joint_dim = jacobian.cols();
  if (jacobian.rows() == 0 || joint_dim == 0 || error.size() != jacobian.rows()) {
    return Eigen::VectorXd::Zero(joint_dim);
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
    jacobian,
    Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto singular_values = svd.singularValues();
  const Eigen::MatrixXd& U = svd.matrixU();
  const Eigen::MatrixXd& V = svd.matrixV();

  Eigen::VectorXd dq = Eigen::VectorXd::Zero(joint_dim);
  const double eps = 1e-9;
  const double gamma_max = std::max(1e-4, max_step_norm);
  std::vector<std::pair<Eigen::Index, Eigen::Index>> task_blocks;
  for (Eigen::Index start = 0; start < jacobian.rows(); start += 3) {
    task_blocks.emplace_back(start, std::min<Eigen::Index>(3, jacobian.rows() - start));
  }

  for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
    const double sigma = singular_values[i];
    if (sigma <= eps || !std::isfinite(sigma)) {
      continue;
    }

    double n_i = 0.0;
    double m_i = 0.0;
    for (const auto& block : task_blocks) {
      const Eigen::Index start = block.first;
      const Eigen::Index size = block.second;
      n_i += U.col(i).segment(start, size).norm();
      for (Eigen::Index j = 0; j < joint_dim; ++j) {
        const double rho = jacobian.col(j).segment(start, size).norm();
        m_i += std::abs(V(j, i)) * rho / sigma;
      }
    }

    const double gamma_i =
      m_i > eps ? std::min(1.0, n_i / m_i) * gamma_max : gamma_max;
    const double task_component = U.col(i).dot(error);
    Eigen::VectorXd phi_i = -(task_component / sigma) * V.col(i);
    clampMaxAbsInPlace(phi_i, gamma_i);

    dq += phi_i;
  }
  clampMaxAbsInPlace(dq, gamma_max);

  return dq;
}

Eigen::MatrixXd selectivelyDampedPseudoinverse(
  const Eigen::MatrixXd& jacobian,
  double max_step_norm) {
  const Eigen::Index joint_dim = jacobian.cols();
  const Eigen::Index task_dim = jacobian.rows();
  if (task_dim == 0 || joint_dim == 0) {
    return Eigen::MatrixXd::Zero(joint_dim, task_dim);
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
    jacobian,
    Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto singular_values = svd.singularValues();
  const Eigen::MatrixXd& U = svd.matrixU();
  const Eigen::MatrixXd& V = svd.matrixV();

  Eigen::MatrixXd pinv = Eigen::MatrixXd::Zero(joint_dim, task_dim);
  const double eps = 1e-9;
  const double gamma_max = std::max(1e-4, max_step_norm);
  std::vector<std::pair<Eigen::Index, Eigen::Index>> task_blocks;
  for (Eigen::Index start = 0; start < task_dim; start += 3) {
    task_blocks.emplace_back(start, std::min<Eigen::Index>(3, task_dim - start));
  }

  for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
    const double sigma = singular_values[i];
    if (sigma <= eps || !std::isfinite(sigma)) {
      continue;
    }
    double n_i = 0.0;
    double m_i = 0.0;
    for (const auto& block : task_blocks) {
      const Eigen::Index start = block.first;
      const Eigen::Index size = block.second;
      n_i += U.col(i).segment(start, size).norm();
      for (Eigen::Index j = 0; j < joint_dim; ++j) {
        const double rho = jacobian.col(j).segment(start, size).norm();
        m_i += std::abs(V(j, i)) * rho / sigma;
      }
    }
    const double gain_scale = m_i > eps ? std::min(1.0, n_i / m_i) : 1.0;
    const double per_direction_scale = std::min(1.0, gain_scale * gamma_max / std::max(gamma_max, eps));
    pinv += per_direction_scale * (V.col(i) * U.col(i).transpose()) / sigma;
  }

  return pinv;
}

Eigen::VectorXd solveIkStep(
  const Eigen::MatrixXd& jacobian,
  const Eigen::VectorXd& error,
  double damping,
  double max_step_norm,
  bool use_sdls) {
  if (use_sdls) {
    return solveSelectivelyDampedLeastSquaresStep(
      jacobian,
      error,
      max_step_norm);
  }
  return solveDampedLeastSquaresStep(jacobian, error, damping);
}

void updateSingularityDiagnostics(
  const Eigen::MatrixXd& jacobian,
  double singular_threshold,
  KinematicsSolver::IKDiagnostics* diagnostics) {
  if (diagnostics == nullptr || jacobian.rows() == 0 || jacobian.cols() == 0) {
    return;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto singular_values = svd.singularValues();
  if (singular_values.size() == 0) {
    return;
  }
  const double max_sigma = singular_values[0];
  const double min_sigma = singular_values[singular_values.size() - 1];
  double manipulability = 1.0;
  for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
    manipulability *= std::max(0.0, singular_values[i]);
  }
  diagnostics->max_singular_value = max_sigma;
  diagnostics->min_singular_value = min_sigma;
  diagnostics->condition_number =
    min_sigma > 1e-12 ? max_sigma / min_sigma : std::numeric_limits<double>::infinity();
  diagnostics->manipulability = manipulability;
  diagnostics->near_singular = min_sigma < std::max(1e-9, singular_threshold);
}

}  // namespace

KinematicsSolver::KinematicsSolver(const RobotModel& robot_model)
: robot_model_(robot_model) {}

pinocchio::SE3 KinematicsSolver::solveFK(
  const std::string& group_name,
  const Eigen::VectorXd& q_group,
  const std::string& ee_frame) const
{
  const auto& group = robot_model_.getPlanningGroup(group_name);
  const std::string target_frame = ee_frame.empty() ? group.tip_link : ee_frame;

  auto q_full = robot_model_.groupToFull(group_name, q_group, robot_model_.neutralConfiguration());
  auto& data = const_cast<RobotModel&>(robot_model_).data();
  pinocchio::forwardKinematics(robot_model_.model(), data, q_full);
  pinocchio::updateFramePlacements(robot_model_.model(), data);
  return data.oMf[robot_model_.getFrameIdChecked(target_frame)];
}

std::pair<bool, Eigen::VectorXd> KinematicsSolver::solveIK(
  const pinocchio::SE3& target_pose,
  const std::string& group_name,
  const Eigen::VectorXd& q_seed_group,
  const std::string& ee_frame,
  int max_iterations,
  double pos_eps,
  double rot_eps,
  double damping,
  double alpha,
  double max_step_norm,
  const std::atomic_bool* cancel,
  bool use_sdls) const
{
  const auto& group = robot_model_.getPlanningGroup(group_name);
  const std::string target_frame = ee_frame.empty() ? group.tip_link : ee_frame;
  auto q_full = robot_model_.groupToFull(group_name, q_seed_group, robot_model_.neutralConfiguration());

  const auto lower = robot_model_.lowerBoundsForGroup(group_name);
  const auto upper = robot_model_.upperBoundsForGroup(group_name);

  pinocchio::Data data(robot_model_.model());
  const auto frame_id = robot_model_.getFrameIdChecked(target_frame);

  for (int iter = 0; iter < max_iterations; ++iter) {
    if (cancel != nullptr && cancel->load()) {
      break;
    }
    pinocchio::forwardKinematics(robot_model_.model(), data, q_full);
    pinocchio::updateFramePlacements(robot_model_.model(), data);

    const pinocchio::SE3 current = data.oMf[frame_id];
    const pinocchio::SE3 iMd = current.actInv(target_pose);
    const Eigen::Matrix<double, 6, 1> err = pinocchio::log6(iMd).toVector();
    const double pos_err = err.head<3>().norm();
    const double rot_err = err.tail<3>().norm();

    if (pos_err < pos_eps && rot_err < rot_eps) {
      return {true, robot_model_.fullToGroup(group_name, q_full)};
    }

    Eigen::Matrix<double, 6, Eigen::Dynamic> J_local(6, robot_model_.model().nv);
    J_local.setZero();
    pinocchio::computeFrameJacobian(
      robot_model_.model(), data, q_full, frame_id, pinocchio::LOCAL, J_local);

    pinocchio::Data::Matrix6 Jlog;
    pinocchio::Jlog6(iMd.inverse(), Jlog);
    const Eigen::Matrix<double, 6, Eigen::Dynamic> J_task = -Jlog * J_local;

    Eigen::MatrixXd Jg(6, static_cast<Eigen::Index>(group.joint_names.size()));
    Jg.setZero();
    for (size_t c = 0; c < group.joint_names.size(); ++c) {
      Jg.col(static_cast<Eigen::Index>(c)) = J_task.col(robot_model_.vIndexOfJoint(group.joint_names[c]));
    }

    Eigen::Matrix<double, 6, 6> W = Eigen::Matrix<double, 6, 6>::Identity();
    W.topLeftCorner<3,3>() *= 1.0;
    W.bottomRightCorner<3,3>() *= 0.3;

    const Eigen::MatrixXd JW = W * Jg;
    const Eigen::Matrix<double, 6, 1> eW = W * err;
    Eigen::VectorXd dq_group = alpha * solveIkStep(
      JW,
      eW,
      damping,
      max_step_norm,
      use_sdls);

    const double step_norm = dq_group.norm();
    if (step_norm > max_step_norm && step_norm > 1e-12) {
      dq_group *= (max_step_norm / step_norm);
    }

    Eigen::VectorXd v_full = Eigen::VectorXd::Zero(robot_model_.model().nv);
    for (size_t i = 0; i < group.joint_names.size(); ++i) {
      v_full(robot_model_.vIndexOfJoint(group.joint_names[i])) = dq_group(static_cast<Eigen::Index>(i));
    }

    q_full = pinocchio::integrate(robot_model_.model(), q_full, v_full);

    Eigen::VectorXd q_group = robot_model_.fullToGroup(group_name, q_full);
    for (Eigen::Index i = 0; i < q_group.size(); ++i) {
      q_group.coeffRef(i) = std::min(std::max(q_group.coeff(i), lower.coeff(i)), upper.coeff(i));
    }
    q_full = robot_model_.groupToFull(group_name, q_group, q_full);
  }

  return {false, robot_model_.fullToGroup(group_name, q_full)};
}


std::pair<bool, Eigen::VectorXd> KinematicsSolver::solveIKWeighted(
  const pinocchio::SE3& target_pose,
  const std::string& group_name,
  const Eigen::VectorXd& q_seed_group,
  const Eigen::Matrix<double, 6, 1>& task_weights,
  const std::string& ee_frame,
  int max_iterations,
  double pos_eps,
  double weighted_rot_eps,
  double damping,
  double alpha,
  double max_step_norm,
  const std::atomic_bool* cancel,
  bool use_sdls) const
{
  const auto& group = robot_model_.getPlanningGroup(group_name);
  const std::string target_frame = ee_frame.empty() ? group.tip_link : ee_frame;
  auto q_full = robot_model_.groupToFull(group_name, q_seed_group, robot_model_.neutralConfiguration());

  const auto lower = robot_model_.lowerBoundsForGroup(group_name);
  const auto upper = robot_model_.upperBoundsForGroup(group_name);

  Eigen::Matrix<double, 6, 1> weights = task_weights;
  for (Eigen::Index i = 0; i < weights.size(); ++i) {
    if (!std::isfinite(weights[i]) || weights[i] < 0.0) {
      weights[i] = 0.0;
    }
  }
  // Keep the translational part strongly constrained even if a caller passes zeros.
  for (Eigen::Index i = 0; i < 3; ++i) {
    if (weights[i] <= 0.0) {
      weights[i] = 1.0;
    }
  }

  pinocchio::Data data(robot_model_.model());
  const auto frame_id = robot_model_.getFrameIdChecked(target_frame);

  for (int iter = 0; iter < max_iterations; ++iter) {
    if (cancel != nullptr && cancel->load()) {
      break;
    }
    pinocchio::forwardKinematics(robot_model_.model(), data, q_full);
    pinocchio::updateFramePlacements(robot_model_.model(), data);

    const pinocchio::SE3 current = data.oMf[frame_id];
    const pinocchio::SE3 iMd = current.actInv(target_pose);
    const Eigen::Matrix<double, 6, 1> err = pinocchio::log6(iMd).toVector();
    const Eigen::Matrix<double, 6, 1> weighted_err = weights.asDiagonal() * err;
    const double pos_err = err.head<3>().norm();
    const double weighted_rot_err = weighted_err.tail<3>().norm();

    if (pos_err < pos_eps && weighted_rot_err < weighted_rot_eps) {
      return {true, robot_model_.fullToGroup(group_name, q_full)};
    }

    Eigen::Matrix<double, 6, Eigen::Dynamic> J_local(6, robot_model_.model().nv);
    J_local.setZero();
    pinocchio::computeFrameJacobian(
      robot_model_.model(), data, q_full, frame_id, pinocchio::LOCAL, J_local);

    pinocchio::Data::Matrix6 Jlog;
    pinocchio::Jlog6(iMd.inverse(), Jlog);
    const Eigen::Matrix<double, 6, Eigen::Dynamic> J_task = -Jlog * J_local;

    Eigen::MatrixXd Jg(6, static_cast<Eigen::Index>(group.joint_names.size()));
    Jg.setZero();
    for (size_t c = 0; c < group.joint_names.size(); ++c) {
      Jg.col(static_cast<Eigen::Index>(c)) = J_task.col(robot_model_.vIndexOfJoint(group.joint_names[c]));
    }

    const Eigen::Matrix<double, 6, 6> W = weights.asDiagonal();
    const Eigen::MatrixXd JW = W * Jg;
    const Eigen::Matrix<double, 6, 1> eW = W * err;
    Eigen::VectorXd dq_group = alpha * solveIkStep(
      JW,
      eW,
      damping,
      max_step_norm,
      use_sdls);

    const double step_norm = dq_group.norm();
    if (step_norm > max_step_norm && step_norm > 1e-12) {
      dq_group *= (max_step_norm / step_norm);
    }

    Eigen::VectorXd v_full = Eigen::VectorXd::Zero(robot_model_.model().nv);
    for (size_t i = 0; i < group.joint_names.size(); ++i) {
      v_full(robot_model_.vIndexOfJoint(group.joint_names[i])) = dq_group(static_cast<Eigen::Index>(i));
    }

    q_full = pinocchio::integrate(robot_model_.model(), q_full, v_full);

    Eigen::VectorXd q_group = robot_model_.fullToGroup(group_name, q_full);
    for (Eigen::Index i = 0; i < q_group.size(); ++i) {
      q_group.coeffRef(i) = std::min(std::max(q_group.coeff(i), lower.coeff(i)), upper.coeff(i));
    }
    q_full = robot_model_.groupToFull(group_name, q_group, q_full);
  }

  return {false, robot_model_.fullToGroup(group_name, q_full)};
}

std::vector<KinematicsSolver::IKCandidate> KinematicsSolver::solveIKMultiSeed(
  const pinocchio::SE3& target_pose,
  const std::string& group_name,
  const std::vector<Eigen::VectorXd>& q_seed_groups,
  const std::string& ee_frame,
  int max_iterations,
  double pos_eps,
  double rot_eps,
  double damping,
  double alpha,
  double max_step_norm,
  const std::atomic_bool* cancel,
  bool use_sdls) const
{
  std::vector<IKCandidate> candidates;
  if (q_seed_groups.empty()) {
    return candidates;
  }

  const Eigen::VectorXd& primary_seed = q_seed_groups.front();
  for (const auto& seed : q_seed_groups) {
    auto result = solveIK(target_pose, group_name, seed, ee_frame,
                          max_iterations, pos_eps, rot_eps, damping, alpha, max_step_norm, cancel, use_sdls);
    if (!result.first) {
      continue;
    }

    bool duplicate = false;
    for (const auto& existing : candidates) {
      if ((existing.q_group - result.second).norm() < 1e-3) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }

    IKCandidate cand;
    cand.q_group = result.second;
    cand.distance_to_seed = (cand.q_group - primary_seed).norm();
    candidates.push_back(std::move(cand));
  }

  std::sort(candidates.begin(), candidates.end(), [](const IKCandidate& a, const IKCandidate& b) {
    return a.distance_to_seed < b.distance_to_seed;
  });
  return candidates;
}


std::vector<KinematicsSolver::IKCandidate> KinematicsSolver::solveIKWeightedMultiSeed(
  const pinocchio::SE3& target_pose,
  const std::string& group_name,
  const std::vector<Eigen::VectorXd>& q_seed_groups,
  const Eigen::Matrix<double, 6, 1>& task_weights,
  const std::string& ee_frame,
  int max_iterations,
  double pos_eps,
  double weighted_rot_eps,
  double damping,
  double alpha,
  double max_step_norm,
  const std::atomic_bool* cancel,
  bool use_sdls) const
{
  std::vector<IKCandidate> candidates;
  if (q_seed_groups.empty()) {
    return candidates;
  }

  const Eigen::VectorXd& primary_seed = q_seed_groups.front();
  for (const auto& seed : q_seed_groups) {
    auto result = solveIKWeighted(target_pose, group_name, seed, task_weights, ee_frame,
                                  max_iterations, pos_eps, weighted_rot_eps,
                                  damping, alpha, max_step_norm, cancel, use_sdls);
    if (!result.first) {
      continue;
    }

    bool duplicate = false;
    for (const auto& existing : candidates) {
      if ((existing.q_group - result.second).norm() < 1e-3) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }

    IKCandidate cand;
    cand.q_group = result.second;
    cand.distance_to_seed = (cand.q_group - primary_seed).norm();
    candidates.push_back(std::move(cand));
  }

  std::sort(candidates.begin(), candidates.end(), [](const IKCandidate& a, const IKCandidate& b) {
    return a.distance_to_seed < b.distance_to_seed;
  });
  return candidates;
}

std::pair<bool, Eigen::VectorXd> KinematicsSolver::solveIKContinuous(
  const pinocchio::SE3& target_pose,
  const std::string& group_name,
  const Eigen::VectorXd& q_seed_group,
  const std::string& ee_frame,
  int max_iterations,
  double pos_eps,
  double rot_eps,
  double damping,
  double alpha,
  double max_step_norm,
  bool use_sdls,
  double nullspace_gain,
  double singular_threshold,
  const std::atomic_bool* cancel,
  IKDiagnostics* diagnostics) const
{
  if (diagnostics != nullptr) {
    *diagnostics = IKDiagnostics{};
  }

  const auto& group = robot_model_.getPlanningGroup(group_name);
  const std::string target_frame = ee_frame.empty() ? group.tip_link : ee_frame;
  auto q_full = robot_model_.groupToFull(group_name, q_seed_group, robot_model_.neutralConfiguration());
  const auto lower = robot_model_.lowerBoundsForGroup(group_name);
  const auto upper = robot_model_.upperBoundsForGroup(group_name);

  pinocchio::Data data(robot_model_.model());
  const auto frame_id = robot_model_.getFrameIdChecked(target_frame);
  Eigen::MatrixXd last_jacobian;
  Eigen::Matrix<double, 6, 1> last_error = Eigen::Matrix<double, 6, 1>::Zero();
  double last_pos_err = std::numeric_limits<double>::infinity();
  double last_rot_err = std::numeric_limits<double>::infinity();

  for (int iter = 0; iter < max_iterations; ++iter) {
    if (cancel != nullptr && cancel->load()) {
      break;
    }

    pinocchio::forwardKinematics(robot_model_.model(), data, q_full);
    pinocchio::updateFramePlacements(robot_model_.model(), data);

    const pinocchio::SE3 current = data.oMf[frame_id];
    const pinocchio::SE3 iMd = current.actInv(target_pose);
    const Eigen::Matrix<double, 6, 1> err = pinocchio::log6(iMd).toVector();
    const double pos_err = err.head<3>().norm();
    const double rot_err = err.tail<3>().norm();
    last_error = err;
    last_pos_err = pos_err;
    last_rot_err = rot_err;
    if (diagnostics != nullptr) {
      diagnostics->iterations = iter + 1;
      diagnostics->final_position_error = pos_err;
      diagnostics->final_rotation_error = rot_err;
    }

    if (pos_err < pos_eps && rot_err < rot_eps) {
      if (diagnostics != nullptr && last_jacobian.size() > 0) {
        updateSingularityDiagnostics(last_jacobian, singular_threshold, diagnostics);
      }
      return {true, robot_model_.fullToGroup(group_name, q_full)};
    }

    Eigen::Matrix<double, 6, Eigen::Dynamic> J_local(6, robot_model_.model().nv);
    J_local.setZero();
    pinocchio::computeFrameJacobian(
      robot_model_.model(), data, q_full, frame_id, pinocchio::LOCAL, J_local);

    pinocchio::Data::Matrix6 Jlog;
    pinocchio::Jlog6(iMd.inverse(), Jlog);
    const Eigen::Matrix<double, 6, Eigen::Dynamic> J_task = -Jlog * J_local;

    Eigen::MatrixXd Jg(6, static_cast<Eigen::Index>(group.joint_names.size()));
    Jg.setZero();
    for (size_t c = 0; c < group.joint_names.size(); ++c) {
      Jg.col(static_cast<Eigen::Index>(c)) = J_task.col(robot_model_.vIndexOfJoint(group.joint_names[c]));
    }

    Eigen::Matrix<double, 6, 6> W = Eigen::Matrix<double, 6, 6>::Identity();
    W.topLeftCorner<3,3>() *= 1.0;
    W.bottomRightCorner<3,3>() *= 0.3;
    const Eigen::MatrixXd JW = W * Jg;
    const Eigen::Matrix<double, 6, 1> eW = W * err;
    last_jacobian = JW;

    const Eigen::MatrixXd pinv = use_sdls ?
      selectivelyDampedPseudoinverse(JW, max_step_norm) :
      dampedPseudoinverse(JW, damping);
    Eigen::VectorXd dq_group = -pinv * eW;

    if (nullspace_gain > 0.0 && group.joint_names.size() > 6) {
      const Eigen::MatrixXd projector =
        Eigen::MatrixXd::Identity(Jg.cols(), Jg.cols()) - pinv * JW;
      dq_group += projector * (nullspace_gain * (q_seed_group - robot_model_.fullToGroup(group_name, q_full)));
    }

    dq_group *= alpha;
    const double step_norm = dq_group.norm();
    if (step_norm > max_step_norm && step_norm > 1e-12) {
      dq_group *= (max_step_norm / step_norm);
    }

    Eigen::VectorXd v_full = Eigen::VectorXd::Zero(robot_model_.model().nv);
    for (size_t i = 0; i < group.joint_names.size(); ++i) {
      v_full(robot_model_.vIndexOfJoint(group.joint_names[i])) = dq_group(static_cast<Eigen::Index>(i));
    }

    q_full = pinocchio::integrate(robot_model_.model(), q_full, v_full);

    Eigen::VectorXd q_group = robot_model_.fullToGroup(group_name, q_full);
    for (Eigen::Index i = 0; i < q_group.size(); ++i) {
      const double clamped = std::min(std::max(q_group.coeff(i), lower.coeff(i)), upper.coeff(i));
      if (std::abs(clamped - q_group.coeff(i)) > 1e-10 && diagnostics != nullptr) {
        diagnostics->joint_limit_clamped = true;
        diagnostics->joint_limit_index = static_cast<int>(i);
        diagnostics->joint_limit_value = clamped;
        diagnostics->joint_limit_lower = lower.coeff(i);
        diagnostics->joint_limit_upper = upper.coeff(i);
      }
      q_group.coeffRef(i) = clamped;
    }
    q_full = robot_model_.groupToFull(group_name, q_group, q_full);
  }

  if (diagnostics != nullptr) {
    diagnostics->final_position_error = last_pos_err;
    diagnostics->final_rotation_error = last_rot_err;
    if (last_jacobian.size() > 0) {
      updateSingularityDiagnostics(last_jacobian, singular_threshold, diagnostics);
    }
  }

  (void)last_error;
  return {false, robot_model_.fullToGroup(group_name, q_full)};
}

}  // namespace lite_motion_planner
