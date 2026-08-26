#include "lite_motion_planner/planning_node.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometric_shapes/geometric_shapes/mesh_operations.h>
#include <geometric_shapes/geometric_shapes/shape_operations.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <moveit_msgs/msg/allowed_collision_entry.hpp>
#include <moveit_msgs/msg/allowed_collision_matrix.hpp>
#include <moveit_msgs/msg/planning_scene_world.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/spatial/log.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/qos.hpp>

#include <boost/variant/get.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <filesystem>
#include <future>
#include <limits>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <Eigen/SVD>

namespace lite_motion_planner {
namespace {

constexpr const char* kFilePrefix = "file://";
constexpr const char* kPackagePrefix = "package://";
constexpr const char* kGlobalRobotSelfCollisionIgnoreObjectId = "__robot_self_collision_ignore__";
constexpr const char* kGlobalRobotSelfCollisionAllowanceObjectId =
  "__robot_self_collision_allowance__";
constexpr const char* kBaseGroupCollisionAllowanceObjectId =
  "__base_group_collision_allowance__";

bool fileExists(const std::string& p) {
  std::error_code ec;
  return std::filesystem::exists(std::filesystem::path(p), ec);
}

std::string trimLeadingSlash(std::string s) {
  while (!s.empty() && s.front() == '/') {
    s.erase(s.begin());
  }
  return s;
}

std::string resolvePackageMeshPath(
  const std::string& mesh_resource,
  const std::vector<std::string>& mesh_package_dirs)
{
  if (mesh_resource.rfind(kPackagePrefix, 0) != 0) {
    return std::string();
  }

  const std::string rest = mesh_resource.substr(std::char_traits<char>::length(kPackagePrefix));
  const auto slash_pos = rest.find('/');
  if (slash_pos == std::string::npos) {
    return std::string();
  }

  const std::string package_name = rest.substr(0, slash_pos);
  const std::string relative_path = trimLeadingSlash(rest.substr(slash_pos + 1));

  try {
    const auto share_dir = ament_index_cpp::get_package_share_directory(package_name);
    const auto candidate = (std::filesystem::path(share_dir) / relative_path).string();
    if (fileExists(candidate)) {
      return candidate;
    }
  } catch (const std::exception&) {
    // Try mesh_package_dirs when the package index lookup is unavailable.
  }

  for (const auto& base_dir_raw : mesh_package_dirs) {
    const auto base_dir = std::filesystem::path(base_dir_raw);
    const auto candidate_direct = (base_dir / relative_path).string();
    if (fileExists(candidate_direct)) {
      return candidate_direct;
    }

    const auto candidate_with_pkg = (base_dir / package_name / relative_path).string();
    if (fileExists(candidate_with_pkg)) {
      return candidate_with_pkg;
    }
  }

  return std::string();
}

std::string normalizeMeshResource(
  const std::string& mesh_resource,
  const std::vector<std::string>& mesh_package_dirs)
{
  if (mesh_resource.empty()) {
    return mesh_resource;
  }
  if (mesh_resource.rfind(kFilePrefix, 0) == 0) {
    return mesh_resource;
  }
  if (mesh_resource.rfind(kPackagePrefix, 0) == 0) {
    const auto resolved = resolvePackageMeshPath(mesh_resource, mesh_package_dirs);
    if (!resolved.empty()) {
      return std::string(kFilePrefix) + resolved;
    }
    return mesh_resource;
  }
  return std::string(kFilePrefix) + mesh_resource;
}

pinocchio::SE3 poseMsgToSE3(const geometry_msgs::msg::Pose& pose) {
  Eigen::Quaterniond q(
    pose.orientation.w,
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z);
  return pinocchio::SE3(
    q.toRotationMatrix(),
    Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z));
}

Eigen::Matrix3d quaternionMsgToRotation(const geometry_msgs::msg::Quaternion& q_msg) {
  Eigen::Quaterniond q(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
  if (q.norm() < 1e-12) {
    q = Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q.toRotationMatrix();
}

Eigen::Vector3d rpyFromRotationMatrix(const Eigen::Matrix3d& rotation) {
  return rotation.eulerAngles(0, 1, 2);
}

Eigen::MatrixXd dampedPseudoinverse(
  const Eigen::MatrixXd& jacobian,
  double damping) {
  const Eigen::Index task_dim = jacobian.rows();
  if (task_dim == 0 || jacobian.cols() == 0) {
    return Eigen::MatrixXd::Zero(jacobian.cols(), task_dim);
  }
  const double lambda = std::max(1e-12, damping);
  const Eigen::MatrixXd A =
    jacobian * jacobian.transpose() +
    lambda * Eigen::MatrixXd::Identity(task_dim, task_dim);
  return jacobian.transpose() * A.ldlt().solve(Eigen::MatrixXd::Identity(task_dim, task_dim));
}

void pushUniqueSeed(
  std::vector<Eigen::VectorXd>& seeds,
  const Eigen::VectorXd& candidate,
  double duplicate_norm = 1e-3) {
  for (const auto& existing : seeds) {
    if ((existing - candidate).norm() < duplicate_norm) {
      return;
    }
  }
  seeds.push_back(candidate);
}

std::vector<Eigen::VectorXd> buildIkSeeds(const Eigen::VectorXd& q_start) {
  std::vector<Eigen::VectorXd> seeds;
  pushUniqueSeed(seeds, q_start);

  // A reachable TCP goal can still fail from a single local IK branch. Provide
  // deterministic, low-cost seeds across shoulder/elbow/wrist branches before
  // giving up.
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(q_start.size());
  pushUniqueSeed(seeds, zero);

  const std::vector<int> preferred_indices = {5, 0, 1, 2, 3, 4, 6};
  const std::vector<double> offsets = {1.57, -1.57, 0.85, -0.85};
  for (int idx : preferred_indices) {
    if (idx < 0 || idx >= q_start.size()) {
      continue;
    }
    for (double offset : offsets) {
      Eigen::VectorXd s = q_start;
      s[idx] += offset;
      pushUniqueSeed(seeds, s);
    }
  }

  if (q_start.size() >= 7) {
    const std::vector<std::pair<int, int>> branch_pairs = {{0, 2}, {1, 3}, {2, 4}, {3, 5}};
    for (const auto& pair : branch_pairs) {
      for (double sign : {1.0, -1.0}) {
        Eigen::VectorXd s = q_start;
        s[pair.first] += sign * 0.85;
        s[pair.second] -= sign * 0.85;
        pushUniqueSeed(seeds, s);
      }
    }
  }

  return seeds;
}


pinocchio::SE3 interpolatePose(
  const pinocchio::SE3& start_pose,
  const pinocchio::SE3& goal_pose,
  double alpha) {
  const Eigen::Vector3d t =
    (1.0 - alpha) * start_pose.translation() + alpha * goal_pose.translation();

  const Eigen::Quaterniond q_start(start_pose.rotation());
  const Eigen::Quaterniond q_goal(goal_pose.rotation());
  Eigen::Quaterniond q_interp = q_start.slerp(alpha, q_goal);
  q_interp.normalize();

  return pinocchio::SE3(q_interp.toRotationMatrix(), t);
}

double wrapToPi(double value) {
  while (value > M_PI) {
    value -= 2.0 * M_PI;
  }
  while (value < -M_PI) {
    value += 2.0 * M_PI;
  }
  return value;
}

double relativeYawInReference(
  const Eigen::Matrix3d& reference_rotation,
  const Eigen::Matrix3d& child_rotation) {
  const Eigen::Matrix3d relative_rotation = reference_rotation.transpose() * child_rotation;
  return wrapToPi(std::atan2(relative_rotation(1, 0), relative_rotation(0, 0)));
}

double yawTargetError(double relative_yaw_rad, double target_yaw_rad) {
  return std::abs(wrapToPi(relative_yaw_rad - target_yaw_rad));
}

double shortestAngularDistance(double from, double to) {
  return wrapToPi(to - from);
}

bool yawWithinReferenceRange(double yaw_rad, double min_rad, double max_rad) {
  const double yaw = wrapToPi(yaw_rad);
  const double min_yaw = wrapToPi(min_rad);
  const double max_yaw = wrapToPi(max_rad);
  constexpr double kEps = 1e-9;
  if (min_yaw <= max_yaw) {
    return yaw >= min_yaw - kEps && yaw <= max_yaw + kEps;
  }
  return yaw >= min_yaw - kEps || yaw <= max_yaw + kEps;
}

std::vector<double> generateYawOffsets(double tolerance_rad, int sample_count) {
  std::vector<double> offsets;
  offsets.push_back(0.0);
  if (sample_count <= 1 || tolerance_rad < 1e-6) {
    return offsets;
  }

  const double tol = std::min(std::max(0.0, tolerance_rad), 2.0 * M_PI);
  const bool full_circle = tol >= (2.0 * M_PI - 1e-3);
  if (full_circle) {
    const double step = 2.0 * M_PI / static_cast<double>(sample_count);
    for (int k = 1; static_cast<int>(offsets.size()) < sample_count; ++k) {
      offsets.push_back(wrapToPi(-static_cast<double>(k) * step));
      if (static_cast<int>(offsets.size()) >= sample_count) {
        break;
      }
      offsets.push_back(wrapToPi(static_cast<double>(k) * step));
    }
    offsets.resize(std::min(offsets.size(), static_cast<size_t>(sample_count)));
    return offsets;
  }

  const double half_span = 0.5 * tol;
  const int side_count = std::max(1, sample_count / 2);
  const double step = half_span / static_cast<double>(side_count);
  for (int k = 1; static_cast<int>(offsets.size()) < sample_count; ++k) {
    const double delta = std::min(half_span, static_cast<double>(k) * step);
    offsets.push_back(-delta);
    if (static_cast<int>(offsets.size()) >= sample_count) {
      break;
    }
    offsets.push_back(delta);
  }
  return offsets;
}

pinocchio::SE3 rotatePoseAboutWorldAxis(
  const pinocchio::SE3& pose,
  const Eigen::Vector3d& axis,
  double angle_rad) {
  Eigen::Vector3d normalized_axis = axis;
  if (normalized_axis.norm() < 1e-12) {
    normalized_axis = Eigen::Vector3d::UnitZ();
  } else {
    normalized_axis.normalize();
  }
  const Eigen::AngleAxisd aa(angle_rad, normalized_axis);
  const Eigen::Matrix3d rotated = aa.toRotationMatrix() * pose.rotation();
  return pinocchio::SE3(rotated, pose.translation());
}

std::vector<double> generateYawOffsetsForReferenceRange(
  const pinocchio::SE3& base_pose,
  const Eigen::Vector3d& yaw_axis,
  const Eigen::Matrix3d& reference_rotation,
  double min_yaw_rad,
  double max_yaw_rad,
  int sample_count) {
  std::vector<double> offsets;
  if (sample_count <= 0) {
    return offsets;
  }

  const double base_yaw = relativeYawInReference(reference_rotation, base_pose.rotation());
  const double min_yaw = wrapToPi(min_yaw_rad);
  const double max_yaw = wrapToPi(max_yaw_rad);
  double span = wrapToPi(max_yaw - min_yaw);
  if (span < 0.0) {
    span += 2.0 * M_PI;
  }

  offsets.reserve(static_cast<size_t>(sample_count));
  if (sample_count == 1 || span < 1e-9) {
    offsets.push_back(wrapToPi(min_yaw - base_yaw));
  } else {
    for (int i = 0; i < sample_count; ++i) {
      const double ratio = static_cast<double>(i) / static_cast<double>(sample_count - 1);
      const double desired_yaw = wrapToPi(min_yaw + span * ratio);
      offsets.push_back(wrapToPi(desired_yaw - base_yaw));
    }
  }

  std::sort(
    offsets.begin(),
    offsets.end(),
    [](double a, double b) {
      const double aw = wrapToPi(a);
      const double bw = wrapToPi(b);
      const bool a_negative = aw < -1e-9;
      const bool b_negative = bw < -1e-9;
      if (a_negative != b_negative) {
        return a_negative;
      }
      if (a_negative) {
        return aw > bw;
      }
      return aw < bw;
    });

  std::vector<double> unique_offsets;
  unique_offsets.reserve(offsets.size());
  for (double offset : offsets) {
    const double wrapped = wrapToPi(offset);
    const bool duplicate = std::any_of(
      unique_offsets.begin(),
      unique_offsets.end(),
      [wrapped](double existing) {
        return std::abs(wrapToPi(existing - wrapped)) < 1e-9;
      });
    if (!duplicate) {
      unique_offsets.push_back(wrapped);
    }
  }

  (void)yaw_axis;
  return unique_offsets;
}

size_t computeCartesianInterpolationSteps(
  const pinocchio::SE3& start_pose,
  const pinocchio::SE3& goal_pose,
  double max_translation_step,
  double max_rotation_step_rad,
  size_t min_steps,
  size_t max_steps) {
  const double distance = (goal_pose.translation() - start_pose.translation()).norm();
  const Eigen::Quaterniond q_start(start_pose.rotation());
  const Eigen::Quaterniond q_goal(goal_pose.rotation());
  const double angle = q_start.angularDistance(q_goal);

  size_t steps = std::max<size_t>(1, min_steps);
  while (true) {
    const double translation_step = distance / static_cast<double>(steps);
    const double rotation_step = angle / static_cast<double>(steps);
    const bool translation_ok = translation_step <= std::max(1e-6, max_translation_step) + 1e-9;
    const bool rotation_ok = rotation_step <= std::max(1e-6, max_rotation_step_rad) + 1e-9;
    if (translation_ok && rotation_ok) {
      break;
    }
    if (max_steps > 0 && steps >= max_steps) {
      break;
    }
    const size_t next_steps = steps + 2;
    steps = max_steps > 0 ? std::min(next_steps, max_steps) : next_steps;
  }
  return std::max<size_t>(1, steps);
}

std::vector<Eigen::VectorXd> buildSequentialIkSeeds(
  const Eigen::VectorXd& q_seed,
  int max_seed_count) {
  auto seeds = buildIkSeeds(q_seed);
  if (max_seed_count > 0 && static_cast<int>(seeds.size()) > max_seed_count) {
    seeds.resize(static_cast<size_t>(max_seed_count));
  }
  return seeds;
}

pinocchio::SE3 framePoseForGroupState(
  const RobotModel& robot_model,
  const std::string& group_name,
  const Eigen::VectorXd& q_group,
  const Eigen::VectorXd& base_q_full,
  const std::string& frame_name) {
  const Eigen::VectorXd q_full = robot_model.groupToFull(group_name, q_group, base_q_full);
  pinocchio::Data data(robot_model.model());
  pinocchio::forwardKinematics(robot_model.model(), data, q_full);
  pinocchio::updateFramePlacements(robot_model.model(), data);
  return data.oMf[robot_model.getFrameIdChecked(frame_name)];
}

double frameRelativeYawForGroupState(
  const RobotModel& robot_model,
  const std::string& group_name,
  const Eigen::VectorXd& q_group,
  const Eigen::VectorXd& base_q_full,
  const std::string& frame_name,
  const Eigen::Matrix3d& reference_rotation) {
  const auto frame_pose =
    framePoseForGroupState(robot_model, group_name, q_group, base_q_full, frame_name);
  return relativeYawInReference(reference_rotation, frame_pose.rotation());
}

bool measureCartesianSegmentPathLength(
  const RobotModel& robot_model,
  const std::string& group_name,
  const Eigen::VectorXd& previous_q,
  const Eigen::VectorXd& candidate_q,
  const Eigen::VectorXd& base_q_full,
  const std::string& frame_name,
  double path_check_step,
  double max_path_length_m,
  double* measured_path_length_m,
  std::string* reject_reason,
  const std::atomic_bool* cancel = nullptr) {
  if (measured_path_length_m != nullptr) {
    *measured_path_length_m = 0.0;
  }
  if (max_path_length_m <= 0.0 && measured_path_length_m == nullptr) {
    return true;
  }
  if (cancel != nullptr && cancel->load()) {
    return false;
  }

  const Eigen::VectorXd delta = candidate_q - previous_q;
  const double joint_distance = delta.norm();
  const int substeps = std::max(
    1,
    static_cast<int>(std::ceil(joint_distance / std::max(1e-6, path_check_step))));

  Eigen::Vector3d last_position =
    framePoseForGroupState(robot_model, group_name, previous_q, base_q_full, frame_name).translation();
  double total_length = 0.0;
  for (int s = 1; s <= substeps; ++s) {
    if (cancel != nullptr && cancel->load()) {
      return false;
    }
    const double t = static_cast<double>(s) / static_cast<double>(substeps);
    const Eigen::VectorXd q_interp = previous_q + t * delta;
    const Eigen::Vector3d position =
      framePoseForGroupState(robot_model, group_name, q_interp, base_q_full, frame_name).translation();
    total_length += (position - last_position).norm();
    last_position = position;

    if (measured_path_length_m != nullptr) {
      *measured_path_length_m = total_length;
    }
  }

  if (max_path_length_m > 0.0 && total_length > max_path_length_m + 1e-9) {
    if (reject_reason != nullptr) {
      std::ostringstream oss;
      oss << "cartesian segment frame [" << frame_name << "] path length "
          << total_length << " m exceeds hard limit "
          << max_path_length_m << " m";
      *reject_reason = oss.str();
    }
    return false;
  }

  return true;
}

class ScopedGlobalCollisionAllowance {
public:
  ScopedGlobalCollisionAllowance(CollisionScene& collision_scene, double requested_allowance)
  : collision_scene_(collision_scene),
    previous_allowance_(collision_scene.globalCollisionAllowedPenetration()),
    active_(requested_allowance > 0.0)
  {
    if (active_) {
      collision_scene_.setGlobalCollisionAllowedPenetration(requested_allowance);
    }
  }

  ~ScopedGlobalCollisionAllowance()
  {
    if (active_) {
      collision_scene_.setGlobalCollisionAllowedPenetration(previous_allowance_);
    }
  }

  ScopedGlobalCollisionAllowance(const ScopedGlobalCollisionAllowance&) = delete;
  ScopedGlobalCollisionAllowance& operator=(const ScopedGlobalCollisionAllowance&) = delete;

private:
  CollisionScene& collision_scene_;
  double previous_allowance_{0.0};
  bool active_{false};
};

struct CartesianWaypointCandidate {
  bool found{false};
  Eigen::VectorXd q;
  pinocchio::SE3 target_pose{pinocchio::SE3::Identity()};
  double distance{std::numeric_limits<double>::infinity()};
  double cartesian_segment_path_length{0.0};
  double endpoint_chord_length{0.0};
  double target_position_error{std::numeric_limits<double>::infinity()};
  double target_rotation_error{std::numeric_limits<double>::infinity()};
  double joint_delta_max_abs{0.0};
  int joint_delta_max_index{-1};
  int ik_candidate_count{0};
  int endpoint_collision_reject_count{0};
  int path_collision_reject_count{0};
  std::string endpoint_collision_reject_reason;
  std::string path_collision_reject_reason;
  double yaw_offset{0.0};
  KinematicsSolver::IKDiagnostics ik_diagnostics;
  bool fallback_multiseed_used{false};
  std::string reject_reason;
};

struct NullspaceRecoveryResult {
  bool found{false};
  Eigen::VectorXd q;
  int focus_joint_index{-1};
  std::string focus_joint_name{"n/a"};
  std::string reason;
  bool used_reference{false};
  double focus_joint_delta{0.0};
  double reference_max_error{std::numeric_limits<double>::infinity()};
  double pos_error{0.0};
  double roll_error{0.0};
  double pitch_error{0.0};
  double relative_yaw{0.0};
  int iterations{0};
};

void logNullspaceRecoveryJoints(
  rclcpp::Logger logger,
  const PlanningGroupInfo& group,
  const Eigen::VectorXd& q) {
  std::ostringstream joint_oss;
  joint_oss << "Null-space recovery joint angles:";
  for (size_t ji = 0; ji < group.joint_names.size() &&
       ji < static_cast<size_t>(q.size()); ++ji) {
    joint_oss << " " << group.joint_names[ji]
              << "=" << q[static_cast<Eigen::Index>(ji)];
  }
  RCLCPP_INFO(logger, "%s", joint_oss.str().c_str());
}

int chooseLargestReferenceDeltaJoint(
  const PlanningGroupInfo& group,
  const Eigen::VectorXd& q,
  const Eigen::VectorXd* reference_q) {
  if (reference_q == nullptr ||
      reference_q->size() != static_cast<Eigen::Index>(group.joint_names.size())) {
    return group.joint_names.empty() ? -1 : static_cast<int>(group.joint_names.size() / 2);
  }

  for (const auto& preferred_joint : {"arm_right_J5_joint", "arm_right_J7_joint"}) {
    for (size_t i = 0; i < group.joint_names.size(); ++i) {
      if (group.joint_names[i] != preferred_joint) {
        continue;
      }
      const Eigen::Index idx = static_cast<Eigen::Index>(i);
      const double reference_delta = std::abs(shortestAngularDistance(q[idx], (*reference_q)[idx]));
      if (q[idx] > 2.85 && reference_delta > 0.04) {
        return static_cast<int>(i);
      }
    }
  }

  double best_abs_delta = -1.0;
  int best_index = -1;
  for (Eigen::Index i = 0; i < reference_q->size(); ++i) {
    const double delta = std::abs(shortestAngularDistance(q[i], (*reference_q)[i]));
    if (delta > best_abs_delta) {
      best_abs_delta = delta;
      best_index = static_cast<int>(i);
    }
  }
  return best_index;
}

struct PoseGoalCandidate {
  bool found{false};
  Eigen::VectorXd q;
  double distance{std::numeric_limits<double>::infinity()};
  double yaw_offset{0.0};
  std::string reject_reason;
};

PoseGoalCandidate solvePoseGoalCandidate(
  const KinematicsSolver& kinematics_solver,
  const CollisionScene& collision_scene,
  const RobotModel& robot_model,
  const pinocchio::SE3& target_pose,
  const std::string& group_name,
  const Eigen::VectorXd& q_seed,
  const Eigen::VectorXd& base_q_full,
  const std::string& ee_frame,
  double yaw_offset_rad,
  const Eigen::Vector3d& yaw_axis,
  bool use_weighted_orientation_goal,
  const Eigen::Matrix<double, 6, 1>& task_weights,
  int ik_max_iterations,
  double ik_pos_tolerance,
  double ik_rot_tolerance,
  double ik_damping,
  double ik_alpha,
  double ik_max_step_norm,
  int ik_max_seed_count,
  const std::atomic_bool* cancel = nullptr,
  bool use_sdls = false) {
  PoseGoalCandidate result;
  result.yaw_offset = yaw_offset_rad;
  if (cancel != nullptr && cancel->load()) {
    return result;
  }

  const pinocchio::SE3 candidate_pose =
    std::abs(wrapToPi(yaw_offset_rad)) > 1e-9 ?
    rotatePoseAboutWorldAxis(target_pose, yaw_axis, yaw_offset_rad) :
    target_pose;

  std::vector<KinematicsSolver::IKCandidate> candidates;
  const auto seeds = buildSequentialIkSeeds(q_seed, ik_max_seed_count);
  if (use_weighted_orientation_goal) {
    candidates = kinematics_solver.solveIKWeightedMultiSeed(
      candidate_pose,
      group_name,
      seeds,
      task_weights,
      ee_frame,
      ik_max_iterations,
      ik_pos_tolerance,
      ik_rot_tolerance,
      ik_damping,
      ik_alpha,
      ik_max_step_norm,
      cancel,
      use_sdls);
  } else {
    candidates = kinematics_solver.solveIKMultiSeed(
      candidate_pose,
      group_name,
      seeds,
      ee_frame,
      ik_max_iterations,
      ik_pos_tolerance,
      ik_rot_tolerance,
      ik_damping,
      ik_alpha,
      ik_max_step_norm,
      cancel,
      use_sdls);
  }

  for (const auto& cand : candidates) {
    if (cancel != nullptr && cancel->load()) {
      break;
    }

    const Eigen::VectorXd q_full =
      robot_model.groupToFull(group_name, cand.q_group, base_q_full);
    const auto report = collision_scene.getCollisionReportThreadSafe(q_full);
    if (report.in_collision) {
      std::ostringstream oss;
      oss << report.category << " [" << report.object_a << " vs " << report.object_b << "] "
          << report.detail;
      result.reject_reason = oss.str();
      continue;
    }

    const double d = (cand.q_group - q_seed).norm();
    if (d < result.distance) {
      result.distance = d;
      result.q = cand.q_group;
      result.found = true;
    }
  }

  if (!result.found && result.reject_reason.empty()) {
    result.reject_reason = "no IK candidate";
  }
  return result;
}

int jointIndexFromNameToken(
  const PlanningGroupInfo& group,
  const std::string& text) {
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    if (text.find(group.joint_names[i]) != std::string::npos) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int chooseNullspaceFocusJoint(
  const CartesianWaypointCandidate& failure,
  const PlanningGroupInfo& group) {
  if (failure.ik_diagnostics.joint_limit_index >= 0 &&
      static_cast<size_t>(failure.ik_diagnostics.joint_limit_index) < group.joint_names.size()) {
    return failure.ik_diagnostics.joint_limit_index;
  }
  int idx = jointIndexFromNameToken(group, failure.reject_reason);
  if (idx >= 0) {
    return idx;
  }
  idx = jointIndexFromNameToken(group, failure.endpoint_collision_reject_reason);
  if (idx >= 0) {
    return idx;
  }
  idx = jointIndexFromNameToken(group, failure.path_collision_reject_reason);
  if (idx >= 0) {
    return idx;
  }
  return group.joint_names.empty() ? -1 : static_cast<int>(group.joint_names.size() / 2);
}

bool isArcLimitGuardJointName(const std::string& joint_name) {
  return joint_name.find("J5_joint") != std::string::npos ||
         joint_name.find("J7_joint") != std::string::npos;
}

int chooseArcUpperLimitGuardJoint(
  const PlanningGroupInfo& group,
  const Eigen::VectorXd& q,
  const Eigen::VectorXd& upper,
  double upper_margin_threshold,
  double* selected_margin) {
  int best_index = -1;
  double best_score = std::numeric_limits<double>::infinity();
  double best_margin = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    if (!isArcLimitGuardJointName(group.joint_names[i])) {
      continue;
    }
    const auto idx = static_cast<Eigen::Index>(i);
    if (idx >= q.size() || idx >= upper.size()) {
      continue;
    }
    const double margin = upper[idx] - q[idx];
    if (margin > upper_margin_threshold) {
      continue;
    }
    const bool is_j5 = group.joint_names[i].find("J5_joint") != std::string::npos;
    const double score = margin + (is_j5 ? -0.02 : 0.0);
    if (score < best_score) {
      best_score = score;
      best_margin = margin;
      best_index = static_cast<int>(i);
    }
  }
  if (selected_margin != nullptr) {
    *selected_margin = best_margin;
  }
  return best_index;
}

double yawDistanceToWrappedRange(double yaw, double min_yaw, double max_yaw) {
  if (yawWithinReferenceRange(yaw, min_yaw, max_yaw)) {
    return 0.0;
  }
  return std::min(
    std::abs(shortestAngularDistance(yaw, min_yaw)),
    std::abs(shortestAngularDistance(yaw, max_yaw)));
}

bool nullspacePoseWithinTolerance(
  const pinocchio::SE3& reference_pose,
  const pinocchio::SE3& candidate_pose,
  const Eigen::Matrix3d& yaw_reference_rotation,
  double pos_tol,
  double roll_pitch_tol,
  double yaw_min,
  double yaw_max,
  double* pos_error,
  double* roll_error,
  double* pitch_error,
  double* relative_yaw) {
  const double p_err =
    (candidate_pose.translation() - reference_pose.translation()).norm();
  const Eigen::Matrix3d relative_rotation =
    reference_pose.rotation().transpose() * candidate_pose.rotation();
  const Eigen::Vector3d rpy = rpyFromRotationMatrix(relative_rotation);
  const double r_err = std::abs(wrapToPi(rpy.x()));
  const double pch_err = std::abs(wrapToPi(rpy.y()));
  const double rel_yaw =
    relativeYawInReference(yaw_reference_rotation, candidate_pose.rotation());
  if (pos_error != nullptr) {
    *pos_error = p_err;
  }
  if (roll_error != nullptr) {
    *roll_error = r_err;
  }
  if (pitch_error != nullptr) {
    *pitch_error = pch_err;
  }
  if (relative_yaw != nullptr) {
    *relative_yaw = rel_yaw;
  }
  return p_err <= pos_tol &&
    r_err <= roll_pitch_tol &&
    pch_err <= roll_pitch_tol &&
    yawWithinReferenceRange(rel_yaw, yaw_min, yaw_max);
}

NullspaceRecoveryResult solveNullspaceRecovery(
  const RobotModel& robot_model,
  const CollisionScene& collision_scene,
  const MotionPlanner& motion_planner,
  const std::string& group_name,
  const std::string& ee_frame,
  const Eigen::VectorXd& previous_q,
  const Eigen::VectorXd& base_q_full,
  const CartesianWaypointCandidate& failure,
  int max_attempts,
  int max_iterations,
  double pos_tolerance,
  double roll_pitch_tolerance,
  double yaw_reference_min,
  double yaw_reference_max,
  const Eigen::Matrix3d& yaw_reference_rotation,
  double step_norm,
  double joint_gain,
  double min_joint_change,
  const Eigen::VectorXd* reference_q,
  double reference_gain,
  double damping,
  double path_check_step,
  int focus_joint_override = -1) {
  NullspaceRecoveryResult result;
  const auto& group = robot_model.getPlanningGroup(group_name);
  int focus_index = focus_joint_override;
  if (focus_index < 0 ||
      static_cast<size_t>(focus_index) >= group.joint_names.size()) {
    focus_index = chooseNullspaceFocusJoint(failure, group);
  }
  result.focus_joint_index = focus_index;
  result.focus_joint_name =
    focus_index >= 0 && static_cast<size_t>(focus_index) < group.joint_names.size() ?
    group.joint_names[static_cast<size_t>(focus_index)] : std::string("n/a");
  if (focus_index < 0 || group.joint_names.size() <= 6) {
    result.reason = "no redundant joint available for null-space recovery";
    return result;
  }

  const auto lower = robot_model.lowerBoundsForGroup(group_name);
  const auto upper = robot_model.upperBoundsForGroup(group_name);
  const std::string target_frame = ee_frame.empty() ? group.tip_link : ee_frame;
  const auto frame_id = robot_model.getFrameIdChecked(target_frame);
  const pinocchio::SE3 reference_pose =
    framePoseForGroupState(robot_model, group_name, previous_q, base_q_full, target_frame);
  const double focus_lower = lower[focus_index];
  const double focus_upper = upper[focus_index];
  const double focus_mid = 0.5 * (focus_lower + focus_upper);
  const double focus_range = std::max(1e-6, focus_upper - focus_lower);
  const double initial_focus_value = previous_q[focus_index];
  const double base_direction =
    initial_focus_value >= focus_mid ? -1.0 : 1.0;

  const int attempts = std::max(1, max_attempts);
  const int iterations = std::max(1, max_iterations);
  const std::vector<double> direction_order = {base_direction, -base_direction};
  const std::vector<double> gain_scale = {1.0, 0.5, 1.5, 0.25};
  const bool has_reference =
    reference_q != nullptr &&
    reference_q->size() == static_cast<Eigen::Index>(group.joint_names.size()) &&
    reference_gain > 0.0;
  result.used_reference = has_reference;
  double best_score = std::numeric_limits<double>::infinity();
  std::string last_reject;

  for (int attempt = 0; attempt < attempts; ++attempt) {
    const double direction = direction_order[static_cast<size_t>(attempt) % direction_order.size()];
    const double scale = gain_scale[static_cast<size_t>(attempt / direction_order.size()) % gain_scale.size()];
    Eigen::VectorXd q_group = previous_q;

    for (int iter = 0; iter < iterations; ++iter) {
      pinocchio::Data data(robot_model.model());
      Eigen::VectorXd q_full = robot_model.groupToFull(group_name, q_group, base_q_full);
      pinocchio::forwardKinematics(robot_model.model(), data, q_full);
      pinocchio::updateFramePlacements(robot_model.model(), data);

      const pinocchio::SE3 current = data.oMf[frame_id];
      const pinocchio::SE3 iMd = current.actInv(reference_pose);
      const Eigen::Matrix<double, 6, 1> err = pinocchio::log6(iMd).toVector();

      Eigen::Matrix<double, 6, Eigen::Dynamic> J_local(6, robot_model.model().nv);
      J_local.setZero();
      pinocchio::computeFrameJacobian(
        robot_model.model(), data, q_full, frame_id, pinocchio::LOCAL, J_local);

      pinocchio::Data::Matrix6 Jlog;
      pinocchio::Jlog6(iMd.inverse(), Jlog);
      const Eigen::Matrix<double, 6, Eigen::Dynamic> J_task = -Jlog * J_local;

      Eigen::MatrixXd Jg(6, static_cast<Eigen::Index>(group.joint_names.size()));
      Jg.setZero();
      for (size_t c = 0; c < group.joint_names.size(); ++c) {
        Jg.col(static_cast<Eigen::Index>(c)) =
          J_task.col(robot_model.vIndexOfJoint(group.joint_names[c]));
      }

      Eigen::Matrix<double, 6, 6> W = Eigen::Matrix<double, 6, 6>::Identity();
      W.topLeftCorner<3,3>() *= 1.0;
      W.bottomRightCorner<3,3>() *= 0.3;
      const Eigen::MatrixXd JW = W * Jg;
      const Eigen::Matrix<double, 6, 1> eW = W * err;
      const Eigen::MatrixXd pinv = dampedPseudoinverse(JW, damping);
      Eigen::VectorXd dq_group = -pinv * eW;
      const Eigen::MatrixXd projector =
        Eigen::MatrixXd::Identity(Jg.cols(), Jg.cols()) - pinv * JW;
      Eigen::VectorXd secondary = Eigen::VectorXd::Zero(Jg.cols());
      secondary[focus_index] =
        direction * std::max(1e-4, joint_gain) * scale * focus_range;
      for (Eigen::Index j = 0; j < secondary.size(); ++j) {
        const double lower_margin = q_group[j] - lower[j];
        const double upper_margin = upper[j] - q_group[j];
        const double limit_bias =
          lower_margin < 0.15 ? 0.15 - lower_margin :
          (upper_margin < 0.15 ? -(0.15 - upper_margin) : 0.0);
        secondary[j] += 0.15 * limit_bias;
      }
      if (has_reference) {
        Eigen::VectorXd reference_delta = *reference_q - q_group;
        for (Eigen::Index j = 0; j < reference_delta.size(); ++j) {
          reference_delta[j] = shortestAngularDistance(q_group[j], (*reference_q)[j]);
        }
        secondary += std::max(0.0, reference_gain) * reference_delta;
      }
      dq_group += projector * secondary;

      const double safe_step = std::max(1e-4, step_norm);
      const double dq_norm = dq_group.norm();
      if (dq_norm > safe_step && dq_norm > 1e-12) {
        dq_group *= safe_step / dq_norm;
      }

      q_group += dq_group;
      for (Eigen::Index j = 0; j < q_group.size(); ++j) {
        q_group[j] = std::min(std::max(q_group[j], lower[j]), upper[j]);
      }

      const Eigen::VectorXd candidate_full =
        robot_model.groupToFull(group_name, q_group, base_q_full);
      const auto report = collision_scene.getCollisionReport(candidate_full);
      if (report.in_collision) {
        std::ostringstream oss;
        oss << report.category << " [" << report.object_a << " vs " << report.object_b
            << "] " << report.detail;
        last_reject = oss.str();
        continue;
      }

      const std::vector<Eigen::VectorXd> segment{previous_q, q_group};
      std::string path_collision_reason;
      if (!motion_planner.isPathCollisionFree(
            group_name,
            segment,
            base_q_full,
            path_check_step,
            &path_collision_reason,
            true)) {
        last_reject = path_collision_reason.empty() ?
          "null-space interpolation path collision" :
          ("null-space interpolation path collision: " + path_collision_reason);
        continue;
      }

      const pinocchio::SE3 candidate_pose =
        framePoseForGroupState(robot_model, group_name, q_group, base_q_full, target_frame);
      double pos_err = 0.0;
      double roll_err = 0.0;
      double pitch_err = 0.0;
      double relative_yaw = 0.0;
      const bool within_tol = nullspacePoseWithinTolerance(
        reference_pose,
        candidate_pose,
        yaw_reference_rotation,
        pos_tolerance,
        roll_pitch_tolerance,
        yaw_reference_min,
        yaw_reference_max,
        &pos_err,
        &roll_err,
        &pitch_err,
        &relative_yaw);
      const double focus_delta = q_group[focus_index] - initial_focus_value;
      double reference_max_error = std::numeric_limits<double>::infinity();
      if (has_reference) {
        reference_max_error = 0.0;
        for (Eigen::Index j = 0; j < q_group.size(); ++j) {
          reference_max_error = std::max(
            reference_max_error,
            std::abs(shortestAngularDistance(q_group[j], (*reference_q)[j])));
        }
      }
      const double yaw_outside =
        yawDistanceToWrappedRange(relative_yaw, yaw_reference_min, yaw_reference_max);
      const double score =
        1000.0 * pos_err + 100.0 * (roll_err + pitch_err) +
        10.0 * yaw_outside +
        (has_reference ? 2.0 * reference_max_error : 0.0) -
        std::abs(focus_delta);
      if (score < best_score) {
        best_score = score;
        result.q = q_group;
        result.focus_joint_delta = focus_delta;
        result.reference_max_error = reference_max_error;
        result.pos_error = pos_err;
        result.roll_error = roll_err;
        result.pitch_error = pitch_err;
        result.relative_yaw = relative_yaw;
        result.iterations = iter + 1;
      }
      if (within_tol && std::abs(focus_delta) >= std::max(0.0, min_joint_change)) {
        result.found = true;
        result.reason.clear();
        return result;
      }
      last_reject = "candidate kept TCP pose but did not move focus joint enough or yaw was outside allowed range";
    }
  }

  result.reason = last_reject.empty() ? "no null-space recovery candidate found" : last_reject;
  return result;
}

CartesianWaypointCandidate solveCartesianWaypointCandidate(
  const KinematicsSolver& kinematics_solver,
  const MotionPlanner& motion_planner,
  const RobotModel& robot_model,
  const CollisionScene& collision_scene,
  const pinocchio::SE3& target_pose,
  const std::string& group_name,
  const std::string& ee_frame,
  const Eigen::VectorXd& q_seed,
  const Eigen::VectorXd& previous_q,
  const Eigen::VectorXd& base_q_full,
  double yaw_offset_rad,
  const Eigen::Vector3d& yaw_axis,
  int ik_max_iterations,
  double ik_pos_tolerance,
  double ik_rot_tolerance,
  double ik_damping,
  double ik_alpha,
  double ik_max_step_norm,
  int ik_max_seed_count,
  double path_check_step,
  bool use_sdls,
  double cartesian_nullspace_gain,
  double cartesian_singular_threshold,
  double cartesian_branch_jump_max_rad,
  double cartesian_branch_jump_norm_max,
  std::atomic_bool* cancel) {
  CartesianWaypointCandidate result;
  result.yaw_offset = yaw_offset_rad;
  if (cancel != nullptr && cancel->load()) {
    return result;
  }
  const auto& group = robot_model.getPlanningGroup(group_name);

  const pinocchio::SE3 candidate_pose =
    std::abs(wrapToPi(yaw_offset_rad)) > 1e-9 ?
    rotatePoseAboutWorldAxis(target_pose, yaw_axis, yaw_offset_rad) :
    target_pose;
  result.target_pose = candidate_pose;

  auto continuous = kinematics_solver.solveIKContinuous(
    candidate_pose,
    group_name,
    previous_q,
    ee_frame,
    ik_max_iterations,
    ik_pos_tolerance,
    ik_rot_tolerance,
    ik_damping,
    ik_alpha,
    ik_max_step_norm,
    false,
    cartesian_nullspace_gain,
    cartesian_singular_threshold,
    cancel,
    &result.ik_diagnostics);

  std::vector<KinematicsSolver::IKCandidate> candidates;
  auto append_continuous_candidate =
    [&](const Eigen::VectorXd& q_candidate) {
    KinematicsSolver::IKCandidate cand;
    cand.q_group = q_candidate;
    cand.distance_to_seed = (q_candidate - previous_q).norm();
    candidates.push_back(std::move(cand));
  };

  if (continuous.first) {
    append_continuous_candidate(continuous.second);
  } else if (use_sdls && result.ik_diagnostics.near_singular) {
    KinematicsSolver::IKDiagnostics sdls_diagnostics;
    auto sdls_retry = kinematics_solver.solveIKContinuous(
      candidate_pose,
      group_name,
      previous_q,
      ee_frame,
      ik_max_iterations,
      ik_pos_tolerance,
      ik_rot_tolerance,
      ik_damping,
      ik_alpha,
      ik_max_step_norm,
      true,
      cartesian_nullspace_gain,
      cartesian_singular_threshold,
      cancel,
      &sdls_diagnostics);
    if (sdls_retry.first) {
      result.ik_diagnostics = sdls_diagnostics;
      append_continuous_candidate(sdls_retry.second);
      result.reject_reason = "continuous WDLS failed near singularity; adaptive SDLS retry succeeded";
    }
  }

  if (candidates.empty()) {
    const bool use_sdls_for_fallback = use_sdls && result.ik_diagnostics.near_singular;
    const auto diagnostic_candidates = kinematics_solver.solveIKMultiSeed(
      candidate_pose,
      group_name,
      buildSequentialIkSeeds(previous_q, ik_max_seed_count),
      ee_frame,
      std::max(50, ik_max_iterations / 2),
      ik_pos_tolerance,
      ik_rot_tolerance,
      ik_damping,
      ik_alpha,
      ik_max_step_norm,
      cancel,
      use_sdls_for_fallback);
    result.fallback_multiseed_used = !diagnostic_candidates.empty();
    if (result.fallback_multiseed_used) {
      result.ik_candidate_count = static_cast<int>(diagnostic_candidates.size());
      double best_rejected_branch_norm = std::numeric_limits<double>::infinity();
      double best_rejected_branch_max_abs = std::numeric_limits<double>::infinity();
      int best_rejected_branch_joint_index = -1;
      for (const auto& cand : diagnostic_candidates) {
        const Eigen::VectorXd branch_delta = cand.q_group - previous_q;
        const double branch_norm = branch_delta.norm();
        const double branch_max_abs = branch_delta.cwiseAbs().maxCoeff();
        if (branch_max_abs <= cartesian_branch_jump_max_rad &&
            branch_norm <= cartesian_branch_jump_norm_max) {
          candidates.push_back(cand);
        } else if (branch_norm < best_rejected_branch_norm) {
          best_rejected_branch_norm = branch_norm;
          best_rejected_branch_max_abs = branch_max_abs;
          Eigen::Index max_index = 0;
          branch_delta.cwiseAbs().maxCoeff(&max_index);
          best_rejected_branch_joint_index = static_cast<int>(max_index);
        }
      }
      if (candidates.empty()) {
        const std::string best_rejected_branch_joint =
          best_rejected_branch_joint_index >= 0 &&
          static_cast<size_t>(best_rejected_branch_joint_index) < group.joint_names.size() ?
          group.joint_names[static_cast<size_t>(best_rejected_branch_joint_index)] :
          std::string("n/a");
        std::ostringstream oss;
        oss << "continuous WDLS"
            << (use_sdls_for_fallback ? "/adaptive-SDLS" : "")
            << " IK failed and detached multi-seed IK found "
            << diagnostic_candidates.size()
            << " candidate(s), but all exceeded branch-jump limits"
            << " (max_joint_limit=" << cartesian_branch_jump_max_rad
            << ", norm_limit=" << cartesian_branch_jump_norm_max
            << ", best_rejected_branch=" << best_rejected_branch_joint
            << ":" << best_rejected_branch_max_abs
            << ", best_rejected_norm=" << best_rejected_branch_norm
            << ", min_sigma=" << result.ik_diagnostics.min_singular_value
            << ", cond=" << result.ik_diagnostics.condition_number
            << ", pos_err=" << result.ik_diagnostics.final_position_error
            << ", rot_err=" << result.ik_diagnostics.final_rotation_error;
        if (result.ik_diagnostics.joint_limit_clamped) {
          const std::string limit_joint_name =
            result.ik_diagnostics.joint_limit_index >= 0 &&
            static_cast<size_t>(result.ik_diagnostics.joint_limit_index) < group.joint_names.size() ?
            group.joint_names[static_cast<size_t>(result.ik_diagnostics.joint_limit_index)] :
            std::string("n/a");
          oss << ", joint_limit=" << limit_joint_name
              << " value=" << result.ik_diagnostics.joint_limit_value
              << " range=[" << result.ik_diagnostics.joint_limit_lower
              << ", " << result.ik_diagnostics.joint_limit_upper << "]";
        }
        oss << ")";
        result.reject_reason = oss.str();
      } else {
        std::ostringstream oss;
        oss << "continuous WDLS"
            << (use_sdls_for_fallback ? "/adaptive-SDLS" : "")
            << " IK failed; accepted "
            << candidates.size()
            << " safe branch candidate(s) from "
            << diagnostic_candidates.size()
            << " detached multi-seed IK candidate(s)"
            << " (min_sigma=" << result.ik_diagnostics.min_singular_value
            << ", cond=" << result.ik_diagnostics.condition_number << ")";
        result.reject_reason = oss.str();
      }
    } else {
      std::ostringstream oss;
      oss << "continuous WDLS"
          << (use_sdls && result.ik_diagnostics.near_singular ? "/adaptive-SDLS" : "")
          << " IK failed and diagnostic multi-seed IK found no candidate; likely unreachable/limited waypoint"
          << " (near_singular=" << (result.ik_diagnostics.near_singular ? "true" : "false")
          << ", joint_limit_clamped=" << (result.ik_diagnostics.joint_limit_clamped ? "true" : "false")
          << ", min_sigma=" << result.ik_diagnostics.min_singular_value
          << ", cond=" << result.ik_diagnostics.condition_number
          << ", pos_err=" << result.ik_diagnostics.final_position_error
          << ", rot_err=" << result.ik_diagnostics.final_rotation_error;
      if (result.ik_diagnostics.joint_limit_clamped) {
        const std::string limit_joint_name =
          result.ik_diagnostics.joint_limit_index >= 0 &&
          static_cast<size_t>(result.ik_diagnostics.joint_limit_index) < group.joint_names.size() ?
          group.joint_names[static_cast<size_t>(result.ik_diagnostics.joint_limit_index)] :
          std::string("n/a");
        oss << ", joint_limit=" << limit_joint_name
            << " value=" << result.ik_diagnostics.joint_limit_value
            << " range=[" << result.ik_diagnostics.joint_limit_lower
            << ", " << result.ik_diagnostics.joint_limit_upper << "]";
      }
      oss << ")";
      result.reject_reason = oss.str();
    }
  }

  if (!candidates.empty()) {
    result.ik_candidate_count = static_cast<int>(candidates.size());
  }
  const auto previous_frame_pose =
    framePoseForGroupState(robot_model, group_name, previous_q, base_q_full, ee_frame);

  for (const auto& cand : candidates) {
    if (cancel != nullptr && cancel->load()) {
      break;
    }

    const Eigen::VectorXd q_full =
      robot_model.groupToFull(group_name, cand.q_group, base_q_full);
    const auto report = collision_scene.getCollisionReportThreadSafe(q_full);
    if (report.in_collision) {
      std::ostringstream oss;
      oss << report.category << " [" << report.object_a << " vs " << report.object_b << "] "
          << report.detail;
      result.reject_reason = oss.str();
      result.endpoint_collision_reject_reason = result.reject_reason;
      ++result.endpoint_collision_reject_count;
      continue;
    }

    const std::vector<Eigen::VectorXd> segment{previous_q, cand.q_group};
    std::string path_collision_reason;
    if (!motion_planner.isPathCollisionFree(
          group_name,
          segment,
          base_q_full,
          path_check_step,
          &path_collision_reason,
          true)) {
      result.reject_reason = path_collision_reason.empty() ?
        "joint-space interpolation between cartesian waypoints is in collision" :
        ("joint-space interpolation between cartesian waypoints is in collision: " + path_collision_reason);
      result.path_collision_reject_reason = result.reject_reason;
      ++result.path_collision_reject_count;
      continue;
    }

    double cartesian_segment_path_length = 0.0;
    (void)measureCartesianSegmentPathLength(
      robot_model,
      group_name,
      previous_q,
      cand.q_group,
      base_q_full,
      ee_frame,
      path_check_step,
      0.0,
      &cartesian_segment_path_length,
      nullptr,
      cancel);

    const double d = (cand.q_group - q_seed).norm();
    const Eigen::VectorXd joint_delta = cand.q_group - previous_q;
    double joint_delta_max_abs = 0.0;
    int joint_delta_max_index = -1;
    for (Eigen::Index i = 0; i < joint_delta.size(); ++i) {
      const double abs_delta = std::abs(joint_delta[i]);
      if (abs_delta > joint_delta_max_abs) {
        joint_delta_max_abs = abs_delta;
        joint_delta_max_index = static_cast<int>(i);
      }
    }
    const auto actual_frame_pose =
      framePoseForGroupState(robot_model, group_name, cand.q_group, base_q_full, ee_frame);
    const double endpoint_chord_length =
      (actual_frame_pose.translation() - previous_frame_pose.translation()).norm();
    const double target_position_error =
      (actual_frame_pose.translation() - candidate_pose.translation()).norm();
    const double target_rotation_error =
      Eigen::Quaterniond(actual_frame_pose.rotation()).angularDistance(
        Eigen::Quaterniond(candidate_pose.rotation()));
    const bool better =
      !result.found ||
      cartesian_segment_path_length < result.cartesian_segment_path_length - 1e-9 ||
      (std::abs(cartesian_segment_path_length - result.cartesian_segment_path_length) <= 1e-9 &&
       d < result.distance);
    if (better) {
      result.distance = d;
      result.q = cand.q_group;
      result.cartesian_segment_path_length = cartesian_segment_path_length;
      result.endpoint_chord_length = endpoint_chord_length;
      result.target_position_error = target_position_error;
      result.target_rotation_error = target_rotation_error;
      result.joint_delta_max_abs = joint_delta_max_abs;
      result.joint_delta_max_index = joint_delta_max_index;
      result.found = true;
      if (cancel != nullptr) {
        cancel->store(true);
        break;
      }
    }
  }

  if (!result.found && result.reject_reason.empty()) {
    result.reject_reason = "no IK candidate";
  }
  return result;
}

moveit_msgs::msg::AllowedCollisionMatrix buildAllowedCollisionMatrixMsg(
  const RobotModel& robot_model) {
  moveit_msgs::msg::AllowedCollisionMatrix acm_msg;

  std::set<std::string> name_set;
  for (const auto& kv : robot_model.allowedCollisionMatrix()) {
    name_set.insert(kv.first);
    name_set.insert(kv.second);
  }
  for (const auto& group_kv : robot_model.planningGroups()) {
    for (const auto& link_name : group_kv.second.link_names) {
      name_set.insert(link_name);
    }
  }

  std::vector<std::string> names(name_set.begin(), name_set.end());
  acm_msg.entry_names = names;
  acm_msg.entry_values.reserve(names.size());

  for (const auto& row_name : names) {
    moveit_msgs::msg::AllowedCollisionEntry row;
    row.enabled.resize(names.size(), false);
    for (size_t j = 0; j < names.size(); ++j) {
      if (robot_model.allowedCollisionMatrix().count(
            makeOrderedLinkPair(row_name, names[j])) > 0) {
        row.enabled[j] = true;
      }
    }
    acm_msg.entry_values.push_back(std::move(row));
  }

  return acm_msg;
}

geometry_msgs::msg::Pose se3ToPoseMsg(const pinocchio::SE3& pose) {
  geometry_msgs::msg::Pose out;
  out.position.x = pose.translation().x();
  out.position.y = pose.translation().y();
  out.position.z = pose.translation().z();
  const Eigen::Quaterniond q(pose.rotation());
  out.orientation.x = q.x();
  out.orientation.y = q.y();
  out.orientation.z = q.z();
  out.orientation.w = q.w();
  return out;
}

}  // namespace

PlanningNode::PlanningNode()
: Node("planning_node") {
  const auto urdf_path = this->declare_parameter<std::string>("urdf_path", "");
  const auto srdf_path = this->declare_parameter<std::string>("srdf_path", "");

  rcl_interfaces::msg::ParameterDescriptor mesh_desc;
  mesh_desc.description = "Directories to search for mesh files";
  const auto mesh_package_dirs = this->declare_parameter<std::vector<std::string>>(
    "mesh_package_dirs", std::vector<std::string>(), mesh_desc);

  const auto joint_traj_topic = this->declare_parameter<std::string>(
    "joint_trajectory_topic", "/joint_trajectory_controller/joint_trajectory");
  const auto control_period = this->declare_parameter<double>("control_period", 0.01);

  (void)this->declare_parameter<std::string>("planning_frame", "base_link");
  (void)this->declare_parameter<double>("planning_timeout", 2.0);
  (void)this->declare_parameter<int>("ompl_max_plan_attempts", 4);
  (void)this->declare_parameter<std::string>("default_ee_frame", "tcp_link");
  (void)this->declare_parameter<std::string>("default_left_ee_frame", "tcp_link");
  (void)this->declare_parameter<std::string>("default_right_ee_frame", "tcp_link");

  (void)this->declare_parameter<int>("ik_max_iterations", 100);
  (void)this->declare_parameter<double>("ik_pos_tolerance", 1e-3);
  (void)this->declare_parameter<double>("ik_rot_tolerance", 1e-3);
  (void)this->declare_parameter<double>("ik_damping", 1e-6);
  (void)this->declare_parameter<double>("ik_alpha", 0.5);
  (void)this->declare_parameter<double>("ik_max_step_norm", 0.5);
  (void)this->declare_parameter<int>("ik_max_seed_count", 3);

  (void)this->declare_parameter<double>("trajectory_collision_check_step", 0.05);
  (void)this->declare_parameter<double>("cartesian_max_rotation_step_rad", 0.7853981633974483);
  (void)this->declare_parameter<int>("cartesian_min_interpolation_steps", 1);
  (void)this->declare_parameter<int>("cartesian_max_interpolation_steps", 0);
  (void)this->declare_parameter<int>("cartesian_segment_length_refinement_max_steps", 96);
  (void)this->declare_parameter<bool>("cartesian_log_each_waypoint", false);
  (void)this->declare_parameter<bool>("cartesian_diagnostics_enabled", false);
  (void)this->declare_parameter<double>("cartesian_diagnostics_suspicious_path_m", 0.10);
  (void)this->declare_parameter<double>("cartesian_diagnostics_suspicious_ratio", 5.0);

  // visualization switches
  (void)this->declare_parameter<bool>("enable_visualization", true);
  (void)this->declare_parameter<bool>("enable_moveit_visualization", true);
  (void)this->declare_parameter<bool>("publish_moveit_display_trajectory", true);
  (void)this->declare_parameter<bool>("publish_moveit_planning_scene", true);
  (void)this->declare_parameter<bool>("publish_moveit_monitored_planning_scene", true);
  (void)this->declare_parameter<bool>("publish_joint_states_for_visualization", true);
  (void)this->declare_parameter<bool>("publish_collision_markers", true);
  (void)this->declare_parameter<bool>("publish_robot_collision_markers", true);
  (void)this->declare_parameter<std::string>("collision_marker_topic", "/lite_motion_planner/collision_markers");
  (void)this->declare_parameter<double>("collision_marker_republish_hz", 2.0);
  (void)this->declare_parameter<bool>("publish_collision_labels", true);
  (void)this->declare_parameter<bool>("publish_e3_marker", true);
  (void)this->declare_parameter<std::string>("e3_marker_topic", "/tzb_catch/e3_marker");
  (void)this->declare_parameter<double>("e3_marker_republish_hz", 5.0);
  (void)this->declare_parameter<std::string>("e3_marker_object_id", "E3");
  (void)this->declare_parameter<std::vector<std::string>>(
    "relaxed_scene_collision_links",
    std::vector<std::string>{"tcp_link", "J6_link", "claw1_link", "claw2_link"});
  (void)this->declare_parameter<std::string>("relaxed_scene_collision_object_id", "E3");
  (void)this->declare_parameter<std::vector<std::string>>(
    "relaxed_scene_collision_object_ids",
    std::vector<std::string>());
  (void)this->declare_parameter<double>("relaxed_scene_collision_allowed_penetration", 0.005);
  (void)this->declare_parameter<std::vector<std::string>>(
    "gripper_scene_collision_object_ids",
    std::vector<std::string>());
  (void)this->declare_parameter<std::vector<std::string>>(
    "gripper_scene_collision_links",
    std::vector<std::string>{"claw1_link", "claw2_link"});
  (void)this->declare_parameter<double>("gripper_scene_collision_allowed_penetration", 0.0);
  (void)this->declare_parameter<std::vector<std::string>>(
    "gripper_self_collision_links",
    std::vector<std::string>{"claw1_link", "claw2_link"});
  (void)this->declare_parameter<double>("gripper_self_collision_allowed_penetration", 0.0);
  (void)this->declare_parameter<double>("self_collision_allowed_penetration", 0.0);

  // joint-state playback
  (void)this->declare_parameter<bool>("enable_joint_state_playback", true);
  (void)this->declare_parameter<bool>("joint_state_playback_loop", false);
  (void)this->declare_parameter<double>("joint_state_playback_speed_scale", 1.0);

  (void)this->declare_parameter<std::string>(
    "moveit_display_trajectory_topic", "/display_planned_path");
  (void)this->declare_parameter<std::string>(
    "moveit_planning_scene_topic", "/planning_scene");
  (void)this->declare_parameter<std::string>(
    "moveit_monitored_planning_scene_topic", "/monitored_planning_scene");
  (void)this->declare_parameter<std::string>(
    "visualization_joint_states_topic", "/joint_states");

  planning_cb_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  if (urdf_path.empty()) {
    RCLCPP_ERROR(
      this->get_logger(),
      "URDF path is not set. Please check your config file.");
    throw std::runtime_error("URDF path missing");
  }

  robot_model_ = std::make_unique<RobotModel>(urdf_path, srdf_path, mesh_package_dirs);
  kinematics_solver_ = std::make_unique<KinematicsSolver>(*robot_model_);
  collision_scene_ = std::make_unique<CollisionScene>(*robot_model_);
  RCLCPP_INFO(
    this->get_logger(),
    "Robot collision model initialized: geometries=%zu, active_collision_pairs=%zu",
    collision_scene_->collisionModel().geometryObjects.size(),
    collision_scene_->collisionModel().collisionPairs.size());
  auto relaxed_object_ids =
    this->get_parameter("relaxed_scene_collision_object_ids").as_string_array();
  if (relaxed_object_ids.empty()) {
    relaxed_object_ids.push_back(this->get_parameter("relaxed_scene_collision_object_id").as_string());
  }
  const auto relaxed_links = this->get_parameter("relaxed_scene_collision_links").as_string_array();
  const double relaxed_allowed_penetration =
    this->get_parameter("relaxed_scene_collision_allowed_penetration").as_double();
  for (const auto& object_id : relaxed_object_ids) {
    if (object_id.empty()) {
      continue;
    }
    collision_scene_->setSceneObjectLinkCollisionAllowance(
      object_id,
      relaxed_links,
      relaxed_allowed_penetration);
    RCLCPP_INFO(
      this->get_logger(),
      "Relaxed scene collision allowance: object=[%s], links=%zu, allowed_penetration=%.4f m",
      object_id.c_str(),
      relaxed_links.size(),
      relaxed_allowed_penetration);
  }
  auto gripper_object_ids =
    this->get_parameter("gripper_scene_collision_object_ids").as_string_array();
  if (gripper_object_ids.empty()) {
    gripper_object_ids = relaxed_object_ids;
  }
  const auto gripper_links =
    this->get_parameter("gripper_scene_collision_links").as_string_array();
  const double gripper_allowed_penetration =
    std::max(0.0, this->get_parameter("gripper_scene_collision_allowed_penetration").as_double());
  if (gripper_allowed_penetration > 0.0 && !gripper_links.empty()) {
    for (const auto& object_id : gripper_object_ids) {
      if (object_id.empty()) {
        continue;
      }
      collision_scene_->setPersistentSceneObjectLinkCollisionAllowance(
        object_id,
        gripper_links,
        gripper_allowed_penetration);
      RCLCPP_INFO(
        this->get_logger(),
        "Persistent gripper scene collision allowance: object=[%s], links=%zu, allowed_penetration=%.4f m",
        object_id.c_str(),
        gripper_links.size(),
        gripper_allowed_penetration);
    }
  }
  const double self_collision_allowed_penetration =
    std::max(0.0, this->get_parameter("self_collision_allowed_penetration").as_double());
  collision_scene_->setSelfCollisionAllowedPenetration(self_collision_allowed_penetration);
  RCLCPP_INFO(
    this->get_logger(),
    "Self-collision penetration allowance: %.4f m",
    self_collision_allowed_penetration);
  const auto gripper_self_links =
    this->get_parameter("gripper_self_collision_links").as_string_array();
  const double gripper_self_allowed_penetration =
    std::max(0.0, this->get_parameter("gripper_self_collision_allowed_penetration").as_double());
  if (gripper_self_allowed_penetration > 0.0 && !gripper_self_links.empty()) {
    collision_scene_->setRobotLinkCollisionAllowance(
      gripper_self_links,
      gripper_self_allowed_penetration);
    RCLCPP_INFO(
      this->get_logger(),
      "Persistent gripper self-collision allowance: links=%zu, allowed_penetration=%.4f m",
      gripper_self_links.size(),
      gripper_self_allowed_penetration);
  }
  motion_planner_ = std::make_unique<MotionPlanner>(*robot_model_, *collision_scene_);
  trajectory_generator_ =
    std::make_unique<TrajectoryGenerator>(*robot_model_, control_period);

  trajectory_pub_ =
    this->create_publisher<trajectory_msgs::msg::JointTrajectory>(joint_traj_topic, 10);

  const bool enable_visualization =
    this->get_parameter("enable_visualization").as_bool();
  const bool enable_moveit_visualization =
    this->get_parameter("enable_moveit_visualization").as_bool();
  const bool publish_joint_states =
    this->get_parameter("publish_joint_states_for_visualization").as_bool();
  const auto vis_joint_topic =
    this->get_parameter("visualization_joint_states_topic").as_string();

  if (enable_visualization && publish_joint_states) {
    // Reliable QoS improves compatibility with robot_state_publisher and tools.
    rclcpp::QoS joint_state_qos(rclcpp::KeepLast(50));
    joint_state_qos.reliable().durability_volatile();
    joint_state_pub_ =
      this->create_publisher<sensor_msgs::msg::JointState>(
        vis_joint_topic,
        joint_state_qos);

    rclcpp::SubscriptionOptions joint_state_sub_options;
    joint_state_sub_options.callback_group = planning_cb_group_;
    joint_state_sub_ =
      this->create_subscription<sensor_msgs::msg::JointState>(
        vis_joint_topic,
        joint_state_qos,
        std::bind(&PlanningNode::handleObservedJointState, this, std::placeholders::_1),
        joint_state_sub_options);
  }

  rclcpp::QoS latched_qos(1);
  latched_qos.transient_local().reliable();

  if (enable_visualization &&
      enable_moveit_visualization &&
      this->get_parameter("publish_moveit_display_trajectory").as_bool()) {
    display_traj_pub_ =
      this->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
        this->get_parameter("moveit_display_trajectory_topic").as_string(),
        latched_qos);
  }

  if (enable_visualization &&
      enable_moveit_visualization &&
      this->get_parameter("publish_moveit_planning_scene").as_bool()) {
    planning_scene_pub_ =
      this->create_publisher<moveit_msgs::msg::PlanningScene>(
        this->get_parameter("moveit_planning_scene_topic").as_string(),
        latched_qos);
  }

  if (enable_visualization &&
      enable_moveit_visualization &&
      this->get_parameter("publish_moveit_monitored_planning_scene").as_bool()) {
    monitored_planning_scene_pub_ =
      this->create_publisher<moveit_msgs::msg::PlanningScene>(
        this->get_parameter("moveit_monitored_planning_scene_topic").as_string(),
        latched_qos);
  }

  if (enable_visualization &&
      this->get_parameter("publish_collision_markers").as_bool()) {
    collision_marker_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->get_parameter("collision_marker_topic").as_string(),
        latched_qos);

    const double republish_hz =
      std::max(0.0, this->get_parameter("collision_marker_republish_hz").as_double());
    if (republish_hz > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(1.0 / republish_hz));
      collision_marker_timer_ = this->create_wall_timer(
        std::max(std::chrono::milliseconds(100), period),
        std::bind(&PlanningNode::collisionMarkerTimerTick, this));
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Collision marker publisher active: topic=%s republish_hz=%.2f",
      this->get_parameter("collision_marker_topic").as_string().c_str(),
      republish_hz);
  }

  if (enable_visualization &&
      this->get_parameter("publish_e3_marker").as_bool()) {
    e3_marker_pub_ =
      this->create_publisher<visualization_msgs::msg::Marker>(
        this->get_parameter("e3_marker_topic").as_string(),
        latched_qos);

    const double republish_hz =
      std::max(0.0, this->get_parameter("e3_marker_republish_hz").as_double());
    if (republish_hz > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(1.0 / republish_hz));
      e3_marker_timer_ = this->create_wall_timer(
        std::max(std::chrono::milliseconds(100), period),
        std::bind(&PlanningNode::e3MarkerTimerTick, this));
    }

    RCLCPP_INFO(
      this->get_logger(),
      "E3 marker publisher active: topic=%s republish_hz=%.2f",
      this->get_parameter("e3_marker_topic").as_string().c_str(),
      republish_hz);
  }

  playback_loop_ = this->get_parameter("joint_state_playback_loop").as_bool();
  playback_speed_scale_ =
    std::max(1e-3, this->get_parameter("joint_state_playback_speed_scale").as_double());

  if (enable_visualization &&
      publish_joint_states &&
      this->get_parameter("enable_joint_state_playback").as_bool()) {
    playback_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&PlanningNode::playbackTimerTick, this));
    playback_timer_->cancel();
  }

  service_ = this->create_service<lite_motion_msgs::srv::PlanArmMotion>(
    "plan_arm_motion",
    std::bind(
      &PlanningNode::handlePlanRequest,
      this,
      std::placeholders::_1,
      std::placeholders::_2),
    rmw_qos_profile_services_default,
    planning_cb_group_);

  add_scene_object_service_ = this->create_service<lite_motion_msgs::srv::AddSceneObject>(
    "add_scene_object",
    std::bind(
      &PlanningNode::handleAddSceneObject,
      this,
      std::placeholders::_1,
      std::placeholders::_2),
    rmw_qos_profile_services_default,
    planning_cb_group_);

  remove_scene_object_service_ = this->create_service<lite_motion_msgs::srv::RemoveSceneObject>(
    "remove_scene_object",
    std::bind(
      &PlanningNode::handleRemoveSceneObject,
      this,
      std::placeholders::_1,
      std::placeholders::_2),
    rmw_qos_profile_services_default,
    planning_cb_group_);

  attach_scene_object_service_ = this->create_service<lite_motion_msgs::srv::AttachSceneObject>(
    "attach_scene_object",
    std::bind(
      &PlanningNode::handleAttachSceneObject,
      this,
      std::placeholders::_1,
      std::placeholders::_2),
    rmw_qos_profile_services_default,
    planning_cb_group_);

  check_state_collision_service_ = this->create_service<lite_motion_msgs::srv::CheckStateCollision>(
    "check_state_collision",
    std::bind(
      &PlanningNode::handleCheckStateCollision,
      this,
      std::placeholders::_1,
      std::placeholders::_2),
    rmw_qos_profile_services_default,
    planning_cb_group_);

  set_scene_collision_allowance_service_ = this->create_service<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
    "set_scene_collision_allowance",
    std::bind(
      &PlanningNode::handleSetSceneCollisionAllowance,
      this,
      std::placeholders::_1,
      std::placeholders::_2),
    rmw_qos_profile_services_default,
    planning_cb_group_);

  publishInitialVisualizationState();
}

double PlanningNode::pointTimeSec(
  const trajectory_msgs::msg::JointTrajectoryPoint& point) const {
  return static_cast<double>(point.time_from_start.sec) +
         static_cast<double>(point.time_from_start.nanosec) * 1e-9;
}

void PlanningNode::handleObservedJointState(
  const sensor_msgs::msg::JointState::SharedPtr msg) {
  if (!robot_model_ || !msg || msg->name.empty() || msg->position.empty()) {
    return;
  }

  Eigen::VectorXd q_full =
    current_visual_q_full_.size() == robot_model_->neutralConfiguration().size() ?
    current_visual_q_full_ : robot_model_->neutralConfiguration();

  size_t updated_count = 0;
  const size_t n = std::min(msg->name.size(), msg->position.size());
  for (size_t i = 0; i < n; ++i) {
    try {
      q_full[robot_model_->qIndexOfJoint(msg->name[i])] = msg->position[i];
      ++updated_count;
    } catch (const std::exception&) {
      // Ignore unrelated joints on the shared joint_states topic.
    }
  }

  if (updated_count > 0) {
    current_visual_q_full_ = q_full;
  }
}

sensor_msgs::msg::JointState PlanningNode::makeJointStateMsg(
  const Eigen::VectorXd& q_full,
  const rclcpp::Time& stamp) const {
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = stamp;
  msg.name = robot_model_->activeJointNames();
  msg.position.reserve(msg.name.size());

  for (const auto& joint_name : msg.name) {
    msg.position.push_back(q_full[robot_model_->qIndexOfJoint(joint_name)]);
  }

  return msg;
}

sensor_msgs::msg::JointState PlanningNode::makeJointStateMsgFromGroupPoint(
  const std::string& group_name,
  const Eigen::VectorXd& base_q_full,
  const trajectory_msgs::msg::JointTrajectoryPoint& point,
  const trajectory_msgs::msg::JointTrajectory& traj,
  const rclcpp::Time& stamp) const {
  Eigen::VectorXd q_full = makeFullStateFromGroupPoint(group_name, base_q_full, point);

  sensor_msgs::msg::JointState msg = makeJointStateMsg(q_full, stamp);

  std::unordered_map<std::string, size_t> traj_joint_to_idx;
  traj_joint_to_idx.reserve(traj.joint_names.size());
  for (size_t i = 0; i < traj.joint_names.size(); ++i) {
    traj_joint_to_idx[traj.joint_names[i]] = i;
  }

  if (traj.joint_names.size() == point.velocities.size()) {
    msg.velocity.assign(msg.name.size(), 0.0);
    for (size_t i = 0; i < msg.name.size(); ++i) {
      const auto it = traj_joint_to_idx.find(msg.name[i]);
      if (it != traj_joint_to_idx.end()) {
        msg.velocity[i] = point.velocities[it->second];
      }
    }
  }
  if (traj.joint_names.size() == point.effort.size()) {
    msg.effort.assign(msg.name.size(), 0.0);
    for (size_t i = 0; i < msg.name.size(); ++i) {
      const auto it = traj_joint_to_idx.find(msg.name[i]);
      if (it != traj_joint_to_idx.end()) {
        msg.effort[i] = point.effort[it->second];
      }
    }
  }

  return msg;
}

Eigen::VectorXd PlanningNode::makeFullStateFromGroupPoint(
  const std::string& group_name,
  const Eigen::VectorXd& base_q_full,
  const trajectory_msgs::msg::JointTrajectoryPoint& point) const {
  Eigen::VectorXd q_full = base_q_full;

  const auto& group = robot_model_->getPlanningGroup(group_name);
  const size_t n = std::min(group.joint_names.size(), point.positions.size());
  for (size_t i = 0; i < n; ++i) {
    const auto full_idx = robot_model_->qIndexOfJoint(group.joint_names[i]);
    q_full[full_idx] = point.positions[i];
  }

  return q_full;
}

moveit_msgs::msg::RobotState PlanningNode::makeMoveItRobotStateMsg(
  const Eigen::VectorXd& q_full,
  const rclcpp::Time& stamp,
  bool is_diff) const {
  moveit_msgs::msg::RobotState state;
  state.joint_state = makeJointStateMsg(q_full, stamp);
  state.is_diff = is_diff;
  return state;
}

moveit_msgs::msg::RobotTrajectory PlanningNode::makeMoveItRobotTrajectoryMsg(
  const trajectory_msgs::msg::JointTrajectory& traj) const {
  moveit_msgs::msg::RobotTrajectory out;
  out.joint_trajectory = traj;
  return out;
}

moveit_msgs::msg::CollisionObject PlanningNode::makeCollisionObjectMsg(
  const std::string& object_id,
  const shape_msgs::msg::SolidPrimitive& primitive,
  const geometry_msgs::msg::PoseStamped& pose_stamped) const {
  moveit_msgs::msg::CollisionObject obj;
  obj.id = object_id;
  obj.header = pose_stamped.header;
  if (obj.header.frame_id.empty()) {
    obj.header.frame_id = this->get_parameter("planning_frame").as_string();
  }
  obj.primitives.push_back(primitive);
  obj.primitive_poses.push_back(pose_stamped.pose);
  obj.operation = moveit_msgs::msg::CollisionObject::ADD;
  return obj;
}

shape_msgs::msg::Mesh PlanningNode::loadMeshMsg(
  const std::string& mesh_resource,
  const std::array<double, 3>& mesh_scale) const {
  const std::string uri = normalizeMeshResource(mesh_resource, robot_model_->meshPackageDirs());

  const Eigen::Vector3d scale(mesh_scale[0], mesh_scale[1], mesh_scale[2]);
  std::unique_ptr<shapes::Mesh> mesh(shapes::createMeshFromResource(uri, scale));
  if (!mesh) {
    throw std::runtime_error("failed to load mesh resource: " + mesh_resource);
  }

  shapes::ShapeMsg shape_msg;
  if (!shapes::constructMsgFromShape(mesh.get(), shape_msg)) {
    throw std::runtime_error("failed to convert mesh resource to shape msg: " + mesh_resource);
  }

  const auto * as_mesh = boost::get<shape_msgs::msg::Mesh>(&shape_msg);
  if (!as_mesh) {
    throw std::runtime_error("mesh conversion did not produce shape_msgs/Mesh: " + mesh_resource);
  }
  return *as_mesh;
}

moveit_msgs::msg::CollisionObject PlanningNode::makeCollisionObjectMeshMsg(
  const std::string& object_id,
  const std::string& mesh_resource,
  const std::array<double, 3>& mesh_scale,
  const geometry_msgs::msg::PoseStamped& pose_stamped) const {
  moveit_msgs::msg::CollisionObject obj;
  obj.id = object_id;
  obj.header = pose_stamped.header;
  if (obj.header.frame_id.empty()) {
    obj.header.frame_id = this->get_parameter("planning_frame").as_string();
  }
  obj.meshes.push_back(loadMeshMsg(mesh_resource, mesh_scale));
  obj.mesh_poses.push_back(pose_stamped.pose);
  obj.operation = moveit_msgs::msg::CollisionObject::ADD;
  return obj;
}

moveit_msgs::msg::PlanningScene PlanningNode::makePlanningSceneMsg(
  const Eigen::VectorXd& q_full,
  const std::vector<std::string>& obstacle_ids,
  const std::vector<shape_msgs::msg::SolidPrimitive>& obstacle_shapes,
  const std::vector<geometry_msgs::msg::PoseStamped>& obstacle_poses,
  bool is_diff) const {
  collision_scene_->updateGeometry(q_full);

  moveit_msgs::msg::PlanningScene scene;
  scene.name = "lite_motion_scene";
  scene.robot_model_name = robot_model_->robotName();
  scene.robot_state = makeMoveItRobotStateMsg(q_full, this->now(), is_diff);
  scene.allowed_collision_matrix = buildAllowedCollisionMatrixMsg(*robot_model_);
  scene.is_diff = is_diff;

  const auto persistent_objects = collision_scene_->sceneObjectInfos();
  for (const auto& info : persistent_objects) {
    if (info.attached) {
      // Remove a stale world copy once when the object becomes attached.
      // Publishing this REMOVE diff on every playback frame floods RViz/MoveIt
      // with "Tried to remove world object ... but it does not exist" warnings.
      if (is_diff && world_remove_published_for_attached_ids_.insert(info.id).second) {
        moveit_msgs::msg::CollisionObject remove_world;
        remove_world.id = info.id;
        remove_world.header.frame_id = this->get_parameter("planning_frame").as_string();
        remove_world.operation = moveit_msgs::msg::CollisionObject::REMOVE;
        scene.world.collision_objects.push_back(std::move(remove_world));
      }

      if (info.uses_mesh) {
        moveit_msgs::msg::AttachedCollisionObject out;
        out.link_name = info.frame_id;
        out.object.id = info.id;
        out.object.header.frame_id = info.frame_id;
        out.object.operation = moveit_msgs::msg::CollisionObject::ADD;
        out.object.meshes.push_back(loadMeshMsg(info.mesh_resource, info.mesh_scale));
        out.object.mesh_poses.push_back(info.relative_pose);
        out.touch_links = info.touch_links;
        scene.robot_state.attached_collision_objects.push_back(std::move(out));
        continue;
      }
      scene.robot_state.attached_collision_objects.push_back(makeAttachedCollisionObjectMsg(info));
    } else {
      world_remove_published_for_attached_ids_.erase(info.id);
      if (info.uses_mesh) {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = this->get_parameter("planning_frame").as_string();
        pose_stamped.pose = info.world_pose;
        scene.world.collision_objects.push_back(
          makeCollisionObjectMeshMsg(info.id, info.mesh_resource, info.mesh_scale, pose_stamped));
        continue;
      }
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.frame_id = this->get_parameter("planning_frame").as_string();
      pose_stamped.pose = info.world_pose;
      scene.world.collision_objects.push_back(makeCollisionObjectMsg(info.id, info.primitive, pose_stamped));
    }
  }

  const size_t n_obs = std::min(
    obstacle_ids.size(),
    std::min(obstacle_shapes.size(), obstacle_poses.size()));
  scene.world.collision_objects.reserve(scene.world.collision_objects.size() + n_obs);

  for (size_t i = 0; i < n_obs; ++i) {
    scene.world.collision_objects.push_back(
      makeCollisionObjectMsg(obstacle_ids[i], obstacle_shapes[i], obstacle_poses[i]));
  }

  return scene;
}

moveit_msgs::msg::AttachedCollisionObject PlanningNode::makeAttachedCollisionObjectMsg(
  const CollisionScene::SceneObjectInfo& info) const {
  moveit_msgs::msg::AttachedCollisionObject out;
  out.link_name = info.frame_id;
  out.object.id = info.id;
  out.object.header.frame_id = info.frame_id;
  out.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  out.object.primitives.push_back(info.primitive);
  out.object.primitive_poses.push_back(info.relative_pose);
  out.touch_links = info.touch_links;
  return out;
}

void PlanningNode::publishCurrentPlanningScene(bool is_diff) {
  const Eigen::VectorXd q_full =
    current_visual_q_full_.size() == robot_model_->neutralConfiguration().size() ?
    current_visual_q_full_ : robot_model_->neutralConfiguration();
  auto scene = makePlanningSceneMsg(q_full, {}, {}, {}, is_diff);
  if (planning_scene_pub_) {
    planning_scene_pub_->publish(scene);
  }
  if (monitored_planning_scene_pub_) {
    monitored_planning_scene_pub_->publish(scene);
  }
  publishCollisionMarkers(q_full);
  publishE3Marker();
}

void PlanningNode::logSceneObjectAttachmentState(const std::string& object_id) const {
  const auto infos = collision_scene_->sceneObjectInfos();
  const auto it = std::find_if(
    infos.begin(),
    infos.end(),
    [&object_id](const CollisionScene::SceneObjectInfo& info) {
      return info.id == object_id;
    });
  if (it == infos.end()) {
    RCLCPP_WARN(this->get_logger(), "Scene object [%s] not found when verifying attachment state", object_id.c_str());
    return;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Scene object truth state: id=[%s] attached=%s frame=[%s] touch_links=%zu relative_pose=(%.4f, %.4f, %.4f) world_pose=(%.4f, %.4f, %.4f)",
    it->id.c_str(),
    it->attached ? "true" : "false",
    it->frame_id.c_str(),
    it->touch_links.size(),
    it->relative_pose.position.x,
    it->relative_pose.position.y,
    it->relative_pose.position.z,
    it->world_pose.position.x,
    it->world_pose.position.y,
    it->world_pose.position.z);
}

void PlanningNode::publishCollisionMarkers(const Eigen::VectorXd& q_full) {
  if (!collision_marker_pub_) {
    return;
  }

  current_visual_q_full_ = q_full;
  collision_scene_->updateGeometry(q_full);

  visualization_msgs::msg::MarkerArray marker_array;
  visualization_msgs::msg::Marker clear_marker;
  clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(clear_marker);

  const auto stamp = this->now();
  const auto planning_frame = this->get_parameter("planning_frame").as_string();
  int32_t marker_id = 0;

  const auto scene_infos = collision_scene_->sceneObjectInfos();
  std::unordered_set<std::string> scene_object_ids;
  scene_object_ids.reserve(scene_infos.size());
  for (const auto& info : scene_infos) {
    scene_object_ids.insert(info.id);
  }

  if (this->get_parameter("publish_robot_collision_markers").as_bool()) {
    const auto& collision_model = collision_scene_->collisionModel();
    const auto& collision_data = collision_scene_->collisionData();
    for (size_t i = 0; i < collision_model.geometryObjects.size(); ++i) {
      const auto& geom = collision_model.geometryObjects[i];
      if (scene_object_ids.count(geom.name) > 0) {
        continue;
      }
      visualization_msgs::msg::Marker marker;
      marker.header.stamp = stamp;
      marker.header.frame_id = planning_frame;
      marker.ns = "robot_collision";
      marker.id = marker_id++;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = se3ToPoseMsg(collision_data.oMg[i]);
      marker.color.r = 1.0f;
      marker.color.g = 0.45f;
      marker.color.b = 0.15f;
      marker.color.a = 0.35f;
      marker.frame_locked = false;

      if (!geom.meshPath.empty()) {
        marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        marker.mesh_resource = normalizeMeshResource(geom.meshPath, robot_model_->meshPackageDirs());
        marker.mesh_use_embedded_materials = false;
        marker.scale.x = geom.meshScale.x();
        marker.scale.y = geom.meshScale.y();
        marker.scale.z = geom.meshScale.z();
        marker_array.markers.push_back(std::move(marker));
      }
    }
  }

  const bool publish_collision_labels =
    this->get_parameter("publish_collision_labels").as_bool();

  for (const auto& info : scene_infos) {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = info.attached ? info.frame_id : planning_frame;
    marker.ns = info.attached ? "attached_collision" : "scene_collision";
    marker.id = marker_id++;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = info.attached ? info.relative_pose : info.world_pose;
    marker.frame_locked = info.attached;
    marker.color.r = info.attached ? 0.20f : 0.10f;
    marker.color.g = info.attached ? 0.85f : 0.75f;
    marker.color.b = info.attached ? 1.00f : 0.25f;
    marker.color.a = 0.85f;

    if (publish_collision_labels) {
      visualization_msgs::msg::Marker origin_marker;
      origin_marker.header = marker.header;
      origin_marker.ns = info.attached ? "attached_collision_origin" : "scene_collision_origin";
      origin_marker.id = marker_id++;
      origin_marker.type = visualization_msgs::msg::Marker::SPHERE;
      origin_marker.action = visualization_msgs::msg::Marker::ADD;
      origin_marker.pose = marker.pose;
      origin_marker.frame_locked = marker.frame_locked;
      origin_marker.scale.x = 0.025;
      origin_marker.scale.y = 0.025;
      origin_marker.scale.z = 0.025;
      origin_marker.color.r = 1.0f;
      origin_marker.color.g = 0.0f;
      origin_marker.color.b = 0.0f;
      origin_marker.color.a = 1.0f;
      marker_array.markers.push_back(origin_marker);

      visualization_msgs::msg::Marker text_marker;
      text_marker.header = marker.header;
      text_marker.ns = info.attached ? "attached_collision_label" : "scene_collision_label";
      text_marker.id = marker_id++;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose = marker.pose;
      text_marker.pose.position.z += 0.05;
      text_marker.frame_locked = marker.frame_locked;
      text_marker.scale.z = 0.04;
      text_marker.color.r = 1.0f;
      text_marker.color.g = 1.0f;
      text_marker.color.b = 1.0f;
      text_marker.color.a = 1.0f;
      text_marker.text = info.attached ?
        (info.id + "@@" + info.frame_id + " rel=(" +
          std::to_string(info.relative_pose.position.x).substr(0, 6) + "," +
          std::to_string(info.relative_pose.position.y).substr(0, 6) + "," +
          std::to_string(info.relative_pose.position.z).substr(0, 6) + ")") :
        info.id;
      marker_array.markers.push_back(text_marker);
    }

    if (info.uses_mesh && !info.mesh_resource.empty()) {
      marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
      marker.mesh_resource = normalizeMeshResource(info.mesh_resource, robot_model_->meshPackageDirs());
      marker.mesh_use_embedded_materials = false;
      marker.scale.x = info.mesh_scale[0];
      marker.scale.y = info.mesh_scale[1];
      marker.scale.z = info.mesh_scale[2];
      marker_array.markers.push_back(std::move(marker));
      continue;
    }

    switch (info.primitive.type) {
      case shape_msgs::msg::SolidPrimitive::BOX:
        if (info.primitive.dimensions.size() >= 3) {
          marker.type = visualization_msgs::msg::Marker::CUBE;
          marker.scale.x = info.primitive.dimensions[0];
          marker.scale.y = info.primitive.dimensions[1];
          marker.scale.z = info.primitive.dimensions[2];
          marker_array.markers.push_back(std::move(marker));
        }
        break;
      case shape_msgs::msg::SolidPrimitive::SPHERE:
        if (!info.primitive.dimensions.empty()) {
          marker.type = visualization_msgs::msg::Marker::SPHERE;
          const double d = 2.0 * info.primitive.dimensions[0];
          marker.scale.x = d;
          marker.scale.y = d;
          marker.scale.z = d;
          marker_array.markers.push_back(std::move(marker));
        }
        break;
      case shape_msgs::msg::SolidPrimitive::CYLINDER:
        if (info.primitive.dimensions.size() >= 2) {
          marker.type = visualization_msgs::msg::Marker::CYLINDER;
          const double d = 2.0 * info.primitive.dimensions[1];
          marker.scale.x = d;
          marker.scale.y = d;
          marker.scale.z = info.primitive.dimensions[0];
          marker_array.markers.push_back(std::move(marker));
        }
        break;
      default:
        break;
    }
  }

  collision_marker_pub_->publish(marker_array);
  RCLCPP_DEBUG_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    5000,
    "Published %zu collision markers on %s",
    marker_array.markers.size(),
    this->get_parameter("collision_marker_topic").as_string().c_str());
}

void PlanningNode::collisionMarkerTimerTick() {
  if (!collision_marker_pub_) {
    return;
  }
  const Eigen::VectorXd q_full =
    current_visual_q_full_.size() == robot_model_->neutralConfiguration().size() ?
    current_visual_q_full_ : robot_model_->neutralConfiguration();
  publishCollisionMarkers(q_full);
}

void PlanningNode::publishE3Marker() {
  if (!e3_marker_pub_) {
    return;
  }

  visualization_msgs::msg::Marker marker;
  marker.header.stamp = this->now();
  marker.header.frame_id = this->get_parameter("planning_frame").as_string();
  marker.ns = "tzb_catch";
  marker.id = 0;
  marker.action = visualization_msgs::msg::Marker::DELETE;

  const auto scene_infos = collision_scene_->sceneObjectInfos();
  const auto it = std::find_if(
    scene_infos.begin(),
    scene_infos.end(),
    [this](const CollisionScene::SceneObjectInfo& info) {
      return info.id == this->get_parameter("e3_marker_object_id").as_string();
    });

  if (it != scene_infos.end() && it->uses_mesh && !it->mesh_resource.empty()) {
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
    marker.mesh_resource = normalizeMeshResource(it->mesh_resource, robot_model_->meshPackageDirs());
    marker.mesh_use_embedded_materials = true;
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;
    marker.scale.x = it->mesh_scale[0];
    marker.scale.y = it->mesh_scale[1];
    marker.scale.z = it->mesh_scale[2];

    if (it->attached) {
      marker.header.frame_id = it->frame_id;
      marker.pose = it->relative_pose;
      marker.frame_locked = true;
    } else {
      marker.pose = it->world_pose;
      marker.frame_locked = false;
    }
  }

  e3_marker_pub_->publish(marker);
}

void PlanningNode::e3MarkerTimerTick() {
  publishE3Marker();
}

std::vector<std::string> PlanningNode::inferTouchLinks(const std::string& frame_id) const {
  if (frame_id == "tcp_link" || frame_id == "J6_link") {
    return {"tcp_link", "J6_link", "claw1_link", "claw2_link"};
  }
  if (frame_id == "right_tcp" || frame_id == "right_ccb" || frame_id == "arm_right_J7_link") {
    return {"right_tcp", "right_ccb", "arm_right_J7_link", "zhua_right_1_link", "zhua_right_2_link"};
  }
  if (frame_id == "left_tcp" || frame_id == "left_ccb" || frame_id == "arm_left_J7_link") {
    return {"left_tcp", "left_ccb", "arm_left_J7_link", "zhua_left_1_link", "zhua_left_2_link"};
  }
  return {frame_id};
}

void PlanningNode::handleAddSceneObject(
  const std::shared_ptr<lite_motion_msgs::srv::AddSceneObject::Request> request,
  std::shared_ptr<lite_motion_msgs::srv::AddSceneObject::Response> response) {
  try {
    const bool uses_mesh = !request->mesh_resource.empty();
    if (!collision_scene_->upsertSceneObject(
          request->object_id,
          request->mesh_resource,
          request->mesh_scale,
          request->collision_primitive,
          request->collision_pose)) {
      response->success = false;
      response->message = "unsupported collision primitive";
      return;
    }
    response->success = true;
    response->message = request->mesh_resource.empty() ?
      "scene object added with primitive collision" :
      ("scene object added with mesh collision: " + request->mesh_resource);
    RCLCPP_INFO(
      this->get_logger(),
      "Scene object [%s] inserted into Pinocchio GeometryModel via addGeometryObject(), collision_geometry=%s, resource=[%s]",
      request->object_id.c_str(),
      uses_mesh ? "mesh" : "primitive",
      request->mesh_resource.c_str());
    publishCurrentPlanningScene(true);
  } catch (const std::exception& e) {
    response->success = false;
    response->message = e.what();
  }
}

void PlanningNode::handleRemoveSceneObject(
  const std::shared_ptr<lite_motion_msgs::srv::RemoveSceneObject::Request> request,
  std::shared_ptr<lite_motion_msgs::srv::RemoveSceneObject::Response> response) {
  response->success = collision_scene_->removeSceneObject(request->object_id);
  response->message = response->success ? "scene object removed" : "scene object not found";
  publishCurrentPlanningScene(true);
}

void PlanningNode::handleAttachSceneObject(
  const std::shared_ptr<lite_motion_msgs::srv::AttachSceneObject::Request> request,
  std::shared_ptr<lite_motion_msgs::srv::AttachSceneObject::Response> response) {
  try {
    const auto q_full =
      current_visual_q_full_.size() == robot_model_->neutralConfiguration().size() ?
      current_visual_q_full_ : robot_model_->neutralConfiguration();
    collision_scene_->updateGeometry(q_full);

    const auto touch_links = inferTouchLinks(request->frame_id);
    response->success = collision_scene_->setObjectAttached(
      request->object_id,
      request->attach,
      request->frame_id,
      request->relative_pose,
      touch_links);
    response->message = response->success ? (request->attach ? "scene object attached" : "scene object detached") : "scene object not found";
    if (response->success) {
      logSceneObjectAttachmentState(request->object_id);
    }
    publishCurrentPlanningScene(true);
  } catch (const std::exception& e) {
    response->success = false;
    response->message = e.what();
  }
}

void PlanningNode::handleCheckStateCollision(
  const std::shared_ptr<lite_motion_msgs::srv::CheckStateCollision::Request> request,
  std::shared_ptr<lite_motion_msgs::srv::CheckStateCollision::Response> response) {
  try {
    const auto& group = robot_model_->getPlanningGroup(request->group_name);

    if (request->joint_positions.size() != group.joint_names.size()) {
      response->success = false;
      response->in_collision = false;
      response->message =
        "joint_positions size mismatch for group " + request->group_name;
      return;
    }

    Eigen::VectorXd q_group(static_cast<Eigen::Index>(group.joint_names.size()));
    for (size_t i = 0; i < group.joint_names.size(); ++i) {
      q_group[static_cast<Eigen::Index>(i)] = request->joint_positions[i];
    }

    const Eigen::VectorXd seed_full = robot_model_->neutralConfiguration();
    const Eigen::VectorXd q_full =
      robot_model_->groupToFull(request->group_name, q_group, seed_full);

    const auto report = collision_scene_->getCollisionReport(q_full);

    response->success = true;
    response->in_collision = report.in_collision;
    response->category = report.category;
    response->object_a = report.object_a;
    response->object_b = report.object_b;
    response->detail = report.detail;
    response->message =
      report.in_collision ? "state in collision" : "state collision-free";
  } catch (const std::exception& e) {
    response->success = false;
    response->in_collision = false;
    response->message = e.what();
  }
}


void PlanningNode::handleSetSceneCollisionAllowance(
  const std::shared_ptr<lite_motion_msgs::srv::SetSceneCollisionAllowance::Request> request,
  std::shared_ptr<lite_motion_msgs::srv::SetSceneCollisionAllowance::Response> response) {
  try {
    if (request->object_ids.empty()) {
      response->success = false;
      response->message = "object_ids is empty";
      return;
    }
    if (request->link_names.empty()) {
      response->success = false;
      response->message = "link_names is empty";
      return;
    }

    for (const auto& object_id : request->object_ids) {
      if (object_id.empty()) {
        continue;
      }
      if (object_id == kGlobalRobotSelfCollisionIgnoreObjectId) {
        const bool ignore = request->allowed_penetration < 0.0;
        collision_scene_->setGlobalIgnoredRobotCollisionLinks(request->link_names, ignore);
        RCLCPP_INFO(
          this->get_logger(),
          "Runtime global robot-link collision ignore %s: links=%zu",
          ignore ? "enabled" : "cleared",
          request->link_names.size());
        continue;
      }
      if (object_id == kGlobalRobotSelfCollisionAllowanceObjectId) {
        collision_scene_->setSelfCollisionAllowedPenetration(request->allowed_penetration);
        RCLCPP_INFO(
          this->get_logger(),
          "Runtime robot self-collision allowance set: allowed_penetration=%.4f m",
          request->allowed_penetration);
        continue;
      }
      if (object_id == kBaseGroupCollisionAllowanceObjectId) {
        collision_scene_->setBaseGroupCollisionAllowance(
          request->link_names,
          request->allowed_penetration);
        RCLCPP_INFO(
          this->get_logger(),
          "Runtime base-group collision allowance set: links=%zu allowed_penetration=%.4f m",
          request->link_names.size(),
          request->allowed_penetration);
        continue;
      }
      collision_scene_->setSceneObjectLinkCollisionAllowance(
        object_id,
        request->link_names,
        request->allowed_penetration);
      if (request->allowed_penetration < 0.0) {
        RCLCPP_INFO(
          this->get_logger(),
          "Runtime scene collision ignore enabled: object=[%s], links=%zu",
          object_id.c_str(),
          request->link_names.size());
      } else {
        RCLCPP_INFO(
          this->get_logger(),
          "Runtime relaxed scene collision allowance: object=[%s], links=%zu, allowed_penetration=%.4f m",
          object_id.c_str(),
          request->link_names.size(),
          request->allowed_penetration);
      }
    }

    response->success = true;
    response->message = "scene collision allowance updated";
  } catch (const std::exception& e) {
    response->success = false;
    response->message = e.what();
  }
}
void PlanningNode::publishPlaybackPlanningSceneDiff(const Eigen::VectorXd& q_full) {
  current_visual_q_full_ = q_full;
  auto scene_diff = makePlanningSceneMsg(q_full, {}, {}, {}, true);
  if (planning_scene_pub_) {
    planning_scene_pub_->publish(scene_diff);
  }
  if (monitored_planning_scene_pub_) {
    monitored_planning_scene_pub_->publish(scene_diff);
  }
  publishCollisionMarkers(q_full);
  publishE3Marker();
}

void PlanningNode::publishInitialVisualizationState() {
  const bool enable_visualization =
    this->get_parameter("enable_visualization").as_bool();
  if (!enable_visualization) {
    return;
  }

  const Eigen::VectorXd q_full = robot_model_->neutralConfiguration();
  current_visual_q_full_ = q_full;
  const auto stamp = this->now();

  if (joint_state_pub_) {
    joint_state_pub_->publish(makeJointStateMsg(q_full, stamp));
  }
  publishCurrentPlanningScene(false);
  publishE3Marker();
}

void PlanningNode::stopJointStatePlayback() {
  playback_active_ = false;
  playback_last_index_ = 0;
  playback_traj_ = trajectory_msgs::msg::JointTrajectory();
  playback_group_name_.clear();
  playback_base_q_full_.resize(0);
  if (playback_timer_) {
    playback_timer_->cancel();
  }
}

void PlanningNode::startJointStatePlayback(
  const std::string& group_name,
  const Eigen::VectorXd& q_start_full,
  const trajectory_msgs::msg::JointTrajectory& traj) {
  if (!joint_state_pub_ || !playback_timer_ || traj.points.empty()) {
    return;
  }

  playback_group_name_ = group_name;
  playback_base_q_full_ = q_start_full;
  playback_traj_ = traj;
  playback_started_at_ = this->now();
  playback_last_index_ = 0;
  playback_active_ = true;
  playback_timer_->reset();

  joint_state_pub_->publish(
    makeJointStateMsgFromGroupPoint(
      playback_group_name_,
      playback_base_q_full_,
      playback_traj_.points.front(),
      playback_traj_,
      this->now()));

  publishPlaybackPlanningSceneDiff(
    makeFullStateFromGroupPoint(
      playback_group_name_,
      playback_base_q_full_,
      playback_traj_.points.front()));
}

void PlanningNode::playbackTimerTick() {
  if (!playback_active_ || !joint_state_pub_ || playback_traj_.points.empty()) {
    return;
  }

  const double elapsed =
    (this->now() - playback_started_at_).seconds() * playback_speed_scale_;

  size_t idx = playback_last_index_;
  while (idx + 1 < playback_traj_.points.size() &&
         pointTimeSec(playback_traj_.points[idx + 1]) <= elapsed) {
    ++idx;
  }

  if (idx != playback_last_index_) {
    playback_last_index_ = idx;
    const auto q_full = makeFullStateFromGroupPoint(
      playback_group_name_,
      playback_base_q_full_,
      playback_traj_.points[idx]);

    joint_state_pub_->publish(
      makeJointStateMsgFromGroupPoint(
        playback_group_name_,
        playback_base_q_full_,
        playback_traj_.points[idx],
        playback_traj_,
        this->now()));

    publishPlaybackPlanningSceneDiff(q_full);
  }

  const double total_duration = pointTimeSec(playback_traj_.points.back());
  if (elapsed >= total_duration) {
    const auto final_msg = makeJointStateMsgFromGroupPoint(
      playback_group_name_,
      playback_base_q_full_,
      playback_traj_.points.back(),
      playback_traj_,
      this->now());

    joint_state_pub_->publish(final_msg);
    publishPlaybackPlanningSceneDiff(
      makeFullStateFromGroupPoint(
        playback_group_name_,
        playback_base_q_full_,
        playback_traj_.points.back()));

    RCLCPP_INFO(
      this->get_logger(),
      "Playback finished at final joint state for group [%s]",
      playback_group_name_.c_str());

    if (playback_loop_) {
      playback_started_at_ = this->now();
      playback_last_index_ = 0;
    } else {
      stopJointStatePlayback();
    }
  }

  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    1000,
    "Playback active: idx=%zu / %zu",
    playback_last_index_,
    playback_traj_.points.size());
}

void PlanningNode::publishVisualizationState(
  const std::string& group_name,
  const Eigen::VectorXd& q_start,
  const Eigen::VectorXd& q_goal,
  const trajectory_msgs::msg::JointTrajectory& traj,
  const std::vector<std::string>& obstacle_ids,
  const std::vector<shape_msgs::msg::SolidPrimitive>& obstacle_shapes,
  const std::vector<geometry_msgs::msg::PoseStamped>& obstacle_poses) {
  const bool enable_visualization =
    this->get_parameter("enable_visualization").as_bool();
  if (!enable_visualization) {
    return;
  }

  const Eigen::VectorXd base_q_full =
    current_visual_q_full_.size() == robot_model_->neutralConfiguration().size() ?
    current_visual_q_full_ : robot_model_->neutralConfiguration();
  const Eigen::VectorXd q_start_full =
    robot_model_->groupToFull(group_name, q_start, base_q_full);
  const Eigen::VectorXd q_goal_full =
    robot_model_->groupToFull(group_name, q_goal, q_start_full);

  if (planning_scene_pub_) {
    planning_scene_pub_->publish(
      makePlanningSceneMsg(q_start_full, obstacle_ids, obstacle_shapes, obstacle_poses, false));
  }
  if (monitored_planning_scene_pub_) {
    monitored_planning_scene_pub_->publish(
      makePlanningSceneMsg(q_start_full, obstacle_ids, obstacle_shapes, obstacle_poses, false));
  }

  if (display_traj_pub_) {
    moveit_msgs::msg::DisplayTrajectory display;
    display.model_id = robot_model_->robotName();
    display.trajectory_start = makeMoveItRobotStateMsg(q_start_full, this->now());
    display.trajectory.push_back(makeMoveItRobotTrajectoryMsg(traj));
    display_traj_pub_->publish(display);
  }

  if (joint_state_pub_) {
    if (this->get_parameter("enable_joint_state_playback").as_bool() && !traj.points.empty()) {
      startJointStatePlayback(group_name, q_start_full, traj);
    } else {
      current_visual_q_full_ = q_goal_full;
      joint_state_pub_->publish(makeJointStateMsg(q_goal_full, this->now()));
      publishCurrentPlanningScene(true);
    }
  }
}

void PlanningNode::handlePlanRequest(
  const std::shared_ptr<lite_motion_msgs::srv::PlanArmMotion::Request> request,
  std::shared_ptr<lite_motion_msgs::srv::PlanArmMotion::Response> response) {
  try {
    const std::string group_name = request->group_name;
    const std::string planning_frame =
      this->get_parameter("planning_frame").as_string();

    std::string ee_frame = request->ee_frame;
    if (ee_frame.empty()) {
      if (group_name == "left_arm") {
        ee_frame = this->get_parameter("default_left_ee_frame").as_string();
      } else if (group_name == "right_arm") {
        ee_frame = this->get_parameter("default_right_ee_frame").as_string();
      } else {
        ee_frame = this->get_parameter("default_ee_frame").as_string();
      }
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Using end-effector frame [%s] for group [%s]",
      ee_frame.c_str(),
      group_name.c_str());

    collision_scene_->clearTransientObstacles();

    const size_t n_obs = std::min(
      request->obstacle_ids.size(),
      std::min(request->obstacle_shapes.size(), request->obstacle_poses.size()));

    for (size_t i = 0; i < n_obs; ++i) {
      const auto& pose_stamped = request->obstacle_poses[i];
      if (!pose_stamped.header.frame_id.empty() &&
          pose_stamped.header.frame_id != planning_frame) {
        RCLCPP_WARN(
          this->get_logger(),
          "Obstacle [%s] is in frame [%s], but no TF transform is applied yet. "
          "Pose is assumed already in [%s].",
          request->obstacle_ids[i].c_str(),
          pose_stamped.header.frame_id.c_str(),
          planning_frame.c_str());
      }

      collision_scene_->addTransientObstacleFromPrimitive(
        request->obstacle_ids[i],
        request->obstacle_shapes[i],
        pose_stamped.pose);
    }

    Eigen::VectorXd q_start = Eigen::Map<const Eigen::VectorXd>(
      request->start_joint_positions.data(),
      static_cast<Eigen::Index>(request->start_joint_positions.size()));

    const auto& group = robot_model_->getPlanningGroup(group_name);
    const auto expected_dim = static_cast<Eigen::Index>(group.joint_names.size());

    if (q_start.size() != expected_dim) {
      std::ostringstream oss;
      oss << "start_joint_positions dim mismatch, expected "
          << expected_dim << ", got " << q_start.size();
      response->success = false;
      response->message = oss.str();
      return;
    }

    const bool use_request_solver_settings = request->use_request_solver_settings;
    const bool plan_only = request->plan_only;
    const int default_ik_max_iterations = this->get_parameter("ik_max_iterations").as_int();
    const double default_ik_pos_tolerance = this->get_parameter("ik_pos_tolerance").as_double();
    const double default_ik_rot_tolerance = this->get_parameter("ik_rot_tolerance").as_double();
    const double default_ik_damping = this->get_parameter("ik_damping").as_double();
    const double default_ik_alpha = this->get_parameter("ik_alpha").as_double();
    const double default_ik_max_step_norm = this->get_parameter("ik_max_step_norm").as_double();
    const int default_ik_max_seed_count = this->get_parameter("ik_max_seed_count").as_int();
    const double default_timeout = this->get_parameter("planning_timeout").as_double();
    const double default_path_check_step = this->get_parameter("trajectory_collision_check_step").as_double();
    const int default_ompl_max_plan_attempts = this->get_parameter("ompl_max_plan_attempts").as_int();

    const int ik_max_iterations =
      use_request_solver_settings && request->request_ik_max_iterations > 0 ?
      request->request_ik_max_iterations : default_ik_max_iterations;
    const double ik_pos_tolerance =
      use_request_solver_settings && request->request_ik_pos_tolerance > 0.0 ?
      request->request_ik_pos_tolerance : default_ik_pos_tolerance;
    const double ik_rot_tolerance =
      use_request_solver_settings && request->request_ik_rot_tolerance > 0.0 ?
      request->request_ik_rot_tolerance : default_ik_rot_tolerance;
    const double ik_damping =
      use_request_solver_settings && request->request_ik_damping > 0.0 ?
      request->request_ik_damping : default_ik_damping;
    const double ik_alpha =
      use_request_solver_settings && request->request_ik_alpha > 0.0 ?
      request->request_ik_alpha : default_ik_alpha;
    const double ik_max_step_norm =
      use_request_solver_settings && request->request_ik_max_step_norm > 0.0 ?
      request->request_ik_max_step_norm : default_ik_max_step_norm;
    const int ik_max_seed_count =
      use_request_solver_settings && request->request_ik_max_seed_count > 0 ?
      request->request_ik_max_seed_count : default_ik_max_seed_count;
    const double timeout =
      use_request_solver_settings && request->request_planning_timeout > 0.0 ?
      request->request_planning_timeout : default_timeout;
    const double path_check_step =
      use_request_solver_settings && request->request_path_check_step > 0.0 ?
      request->request_path_check_step : default_path_check_step;
    const bool use_cartesian_nullspace_recovery =
      use_request_solver_settings && request->request_use_cartesian_nullspace_recovery;
    const int cartesian_nullspace_recovery_max_attempts =
      request->request_cartesian_nullspace_recovery_max_attempts > 0 ?
      request->request_cartesian_nullspace_recovery_max_attempts : 8;
    const int cartesian_nullspace_recovery_max_iterations =
      request->request_cartesian_nullspace_recovery_max_iterations > 0 ?
      request->request_cartesian_nullspace_recovery_max_iterations : 45;
    const double cartesian_nullspace_recovery_pos_tolerance =
      request->request_cartesian_nullspace_recovery_pos_tolerance > 0.0 ?
      request->request_cartesian_nullspace_recovery_pos_tolerance : 0.010;
    const double cartesian_nullspace_recovery_roll_pitch_tolerance =
      request->request_cartesian_nullspace_recovery_roll_pitch_tolerance > 0.0 ?
      request->request_cartesian_nullspace_recovery_roll_pitch_tolerance : 0.100;
    const double cartesian_nullspace_recovery_yaw_min =
      wrapToPi(request->request_cartesian_nullspace_recovery_yaw_reference_min_rad);
    const double cartesian_nullspace_recovery_yaw_max =
      wrapToPi(request->request_cartesian_nullspace_recovery_yaw_reference_max_rad);
    const Eigen::Matrix3d cartesian_nullspace_recovery_reference_rotation =
      quaternionMsgToRotation(request->request_cartesian_nullspace_recovery_reference_orientation);
    const double cartesian_nullspace_recovery_step_norm =
      request->request_cartesian_nullspace_recovery_step_norm > 0.0 ?
      request->request_cartesian_nullspace_recovery_step_norm : 0.12;
    const double cartesian_nullspace_recovery_joint_gain =
      request->request_cartesian_nullspace_recovery_joint_gain > 0.0 ?
      request->request_cartesian_nullspace_recovery_joint_gain : 0.45;
    const double cartesian_nullspace_recovery_min_joint_change =
      request->request_cartesian_nullspace_recovery_min_joint_change > 0.0 ?
      request->request_cartesian_nullspace_recovery_min_joint_change : 0.08;
    const bool use_cartesian_nullspace_reference =
      use_request_solver_settings &&
      request->request_use_cartesian_nullspace_reference &&
      request->request_cartesian_nullspace_reference_joints.size() ==
        static_cast<size_t>(expected_dim);
    Eigen::VectorXd cartesian_nullspace_reference_q;
    if (use_cartesian_nullspace_reference) {
      cartesian_nullspace_reference_q = Eigen::Map<const Eigen::VectorXd>(
        request->request_cartesian_nullspace_reference_joints.data(),
        expected_dim);
    }
    const double cartesian_nullspace_reference_gain =
      use_cartesian_nullspace_reference &&
      request->request_cartesian_nullspace_reference_gain > 0.0 ?
      request->request_cartesian_nullspace_reference_gain : 0.0;
    const bool cartesian_nullspace_settle_only =
      use_request_solver_settings &&
      request->request_cartesian_nullspace_settle_only &&
      use_cartesian_nullspace_recovery &&
      use_cartesian_nullspace_reference;
    const double cartesian_max_segment_path_length =
      request->request_cartesian_max_segment_path_length > 0.0 ?
      request->request_cartesian_max_segment_path_length : 0.0;
    const double cartesian_nullspace_gain =
      request->request_cartesian_nullspace_gain > 0.0 ?
      request->request_cartesian_nullspace_gain : 0.20;
    const double cartesian_singular_threshold =
      request->request_cartesian_singular_threshold > 0.0 ?
      request->request_cartesian_singular_threshold : 0.04;
    const double cartesian_branch_jump_max_rad =
      request->request_cartesian_branch_jump_max_rad > 0.0 ?
      request->request_cartesian_branch_jump_max_rad : 0.75;
    const double cartesian_branch_jump_norm_max =
      request->request_cartesian_branch_jump_norm_max > 0.0 ?
      request->request_cartesian_branch_jump_norm_max : 1.20;
    const double cartesian_waypoint_pos_tolerance =
      request->request_cartesian_waypoint_pos_tolerance > 0.0 ?
      request->request_cartesian_waypoint_pos_tolerance :
      std::min(ik_pos_tolerance, 0.0015);
    const double cartesian_waypoint_rot_tolerance =
      request->request_cartesian_waypoint_rot_tolerance > 0.0 ?
      request->request_cartesian_waypoint_rot_tolerance :
      std::min(ik_rot_tolerance, 0.020);
    const double request_global_collision_allowed_penetration =
      use_request_solver_settings && request->request_global_collision_allowed_penetration > 0.0 ?
      request->request_global_collision_allowed_penetration : 0.0;
    const int ompl_max_plan_attempts =
      use_request_solver_settings && request->request_ompl_max_plan_attempts > 0 ?
      request->request_ompl_max_plan_attempts : default_ompl_max_plan_attempts;
    const int request_max_ik_goal_candidates =
      use_request_solver_settings && request->request_max_ik_goal_candidates > 0 ?
      request->request_max_ik_goal_candidates : 0;
    const bool use_sdls =
      use_request_solver_settings && request->request_use_sdls;

    if (use_request_solver_settings) {
      RCLCPP_INFO(
        this->get_logger(),
        "Per-request solver settings: ik_iter=%d pos_tol=%.4f rot_tol=%.4f cart_wp_tol=(pos=%.4f,rot=%.4f) damping=%.2e alpha=%.3f step=%.3f seeds=%d adaptive_sdls=%s nullspace=%.3f singular_thresh=%.4f branch_jump=(max=%.3f,norm=%.3f) timeout=%.2f path_step=%.3f cartesian_segment_limit=%.4f global_collision_allowance=%.4f ompl_attempts=%d max_ik_goals=%d nullspace_recovery=%s(ns_attempts=%d,ns_iter=%d,ns_pos_tol=%.4f,ns_rp_tol=%.4f,ns_yaw=[%.4f,%.4f],ns_ref=%s,ns_ref_gain=%.3f,settle_only=%s)",
        ik_max_iterations,
        ik_pos_tolerance,
        ik_rot_tolerance,
        cartesian_waypoint_pos_tolerance,
        cartesian_waypoint_rot_tolerance,
        ik_damping,
        ik_alpha,
        ik_max_step_norm,
        ik_max_seed_count,
        use_sdls ? "on" : "off",
        cartesian_nullspace_gain,
        cartesian_singular_threshold,
        cartesian_branch_jump_max_rad,
        cartesian_branch_jump_norm_max,
        timeout,
        path_check_step,
        cartesian_max_segment_path_length,
        request_global_collision_allowed_penetration,
        ompl_max_plan_attempts,
        request_max_ik_goal_candidates,
        use_cartesian_nullspace_recovery ? "on" : "off",
        cartesian_nullspace_recovery_max_attempts,
        cartesian_nullspace_recovery_max_iterations,
        cartesian_nullspace_recovery_pos_tolerance,
        cartesian_nullspace_recovery_roll_pitch_tolerance,
        cartesian_nullspace_recovery_yaw_min,
        cartesian_nullspace_recovery_yaw_max,
        use_cartesian_nullspace_reference ? "on" : "off",
        cartesian_nullspace_reference_gain,
        cartesian_nullspace_settle_only ? "on" : "off");
    }
    ScopedGlobalCollisionAllowance scoped_global_collision_allowance(
      *collision_scene_,
      request_global_collision_allowed_penetration);
    if (request_global_collision_allowed_penetration > 0.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Per-request global collision penetration allowance active: %.4f m",
        request_global_collision_allowed_penetration);
    }

    const Eigen::VectorXd base_visual_q_full =
      current_visual_q_full_.size() == robot_model_->neutralConfiguration().size() ?
      current_visual_q_full_ : robot_model_->neutralConfiguration();
    const Eigen::VectorXd q_start_full =
      robot_model_->groupToFull(group_name, q_start, base_visual_q_full);

    Eigen::VectorXd q_goal;
    std::vector<Eigen::VectorXd> collision_free_ik_goals;

    if (request->use_pose_goal) {
      if (!request->target_pose.header.frame_id.empty() &&
          request->target_pose.header.frame_id != planning_frame) {
        RCLCPP_WARN(
          this->get_logger(),
          "Target pose is in frame [%s], but no TF transform is applied yet. "
          "Pose is assumed already in [%s].",
          request->target_pose.header.frame_id.c_str(),
          planning_frame.c_str());
      }

      Eigen::Matrix<double, 6, 1> task_weights = Eigen::Matrix<double, 6, 1>::Zero();
      if (request->use_weighted_orientation_goal) {
        task_weights <<
          1.0,
          1.0,
          1.0,
          std::max(0.0, request->orientation_weight_x),
          std::max(0.0, request->orientation_weight_y),
          std::max(0.0, request->orientation_weight_z);
        RCLCPP_INFO(
          this->get_logger(),
          "Weighted IK pose goal enabled: position weights=[1,1,1], orientation weights=[%.4f, %.4f, %.4f]",
          task_weights[3],
          task_weights[4],
          task_weights[5]);
      }

      const bool use_pose_yaw_tolerance =
        request->use_cartesian_yaw_tolerance &&
        request->cartesian_yaw_tolerance_rad > 1e-6 &&
        request->cartesian_yaw_sample_count > 1;
      Eigen::Vector3d pose_yaw_axis(
        request->cartesian_yaw_axis.x,
        request->cartesian_yaw_axis.y,
        request->cartesian_yaw_axis.z);
      if (pose_yaw_axis.norm() < 1e-12) {
        pose_yaw_axis = Eigen::Vector3d::UnitZ();
      } else {
        pose_yaw_axis.normalize();
      }
      const bool use_pose_yaw_target =
        use_pose_yaw_tolerance && request->use_cartesian_yaw_target;
      const bool use_pose_yaw_reference_range =
        use_pose_yaw_tolerance && request->use_cartesian_yaw_reference_range;
      const bool use_pose_yaw_reference_eval =
        use_pose_yaw_target || use_pose_yaw_reference_range;
      const Eigen::Matrix3d pose_yaw_reference_rotation = use_pose_yaw_reference_eval ?
        quaternionMsgToRotation(request->cartesian_yaw_reference_orientation) :
        Eigen::Matrix3d::Identity();
      const double pose_yaw_target_rad = wrapToPi(request->cartesian_yaw_target_rad);
      const double pose_yaw_reference_min_rad =
        wrapToPi(request->cartesian_yaw_reference_min_rad);
      const double pose_yaw_reference_max_rad =
        wrapToPi(request->cartesian_yaw_reference_max_rad);
      const std::string pose_yaw_target_frame =
        request->cartesian_yaw_target_frame.empty() ?
        ee_frame : request->cartesian_yaw_target_frame;

      if (!request->use_cartesian_path) {
        const pinocchio::SE3 goal_pose = poseMsgToSE3(request->target_pose.pose);
        const Eigen::VectorXd base_q_full =
          robot_model_->groupToFull(group_name, q_start, base_visual_q_full);

        if (use_pose_yaw_tolerance) {
          auto yaw_offsets = use_pose_yaw_reference_range ?
            generateYawOffsetsForReferenceRange(
              goal_pose,
              pose_yaw_axis,
              pose_yaw_reference_rotation,
              pose_yaw_reference_min_rad,
              pose_yaw_reference_max_rad,
              request->cartesian_yaw_sample_count) :
            generateYawOffsets(
              request->cartesian_yaw_tolerance_rad,
              request->cartesian_yaw_sample_count);
          if (request->skip_cartesian_yaw_exact) {
            yaw_offsets.erase(
              std::remove_if(
                yaw_offsets.begin(),
                yaw_offsets.end(),
                [](double yaw_offset) {
                  return std::abs(wrapToPi(yaw_offset)) <= 1e-9;
                }),
              yaw_offsets.end());
          }
          if (yaw_offsets.empty()) {
            response->success = false;
            response->message = "IK failed. Yaw tolerance requested but all yaw samples were skipped";
            return;
          }
          std::atomic_bool yaw_cancel{false};
          std::vector<std::future<PoseGoalCandidate>> futures;
          futures.reserve(yaw_offsets.size());
          for (double yaw_offset : yaw_offsets) {
            futures.push_back(std::async(
              std::launch::async,
              [this,
               goal_pose,
               group_name,
               q_start,
               base_q_full,
               ee_frame,
               yaw_offset,
               pose_yaw_axis,
               request,
               task_weights,
               ik_max_iterations,
               ik_pos_tolerance,
               ik_rot_tolerance,
               ik_damping,
               ik_alpha,
               ik_max_step_norm,
               ik_max_seed_count,
               use_sdls,
               &yaw_cancel]() {
                return solvePoseGoalCandidate(
                  *kinematics_solver_,
                  *collision_scene_,
                  *robot_model_,
                  goal_pose,
                  group_name,
                  q_start,
                  base_q_full,
                  ee_frame,
                  yaw_offset,
                  pose_yaw_axis,
                  request->use_weighted_orientation_goal,
                  task_weights,
                  ik_max_iterations,
                  ik_pos_tolerance,
                  ik_rot_tolerance,
                  ik_damping,
                  ik_alpha,
                  ik_max_step_norm,
                  ik_max_seed_count,
                  &yaw_cancel,
                  use_sdls);
              }));
          }

          PoseGoalCandidate selected;
          double selected_relative_yaw = 0.0;
          double selected_yaw_target_error = std::numeric_limits<double>::infinity();
          std::string reject_reason;
          auto consider_pose_yaw_candidate =
            [&](PoseGoalCandidate candidate) -> bool {
            if (!candidate.reject_reason.empty()) {
              reject_reason = candidate.reject_reason;
            }
            if (!candidate.found) {
              return false;
            }
            if (!use_pose_yaw_target && !use_pose_yaw_reference_range) {
              selected = std::move(candidate);
              return true;
            }
            double candidate_relative_yaw = 0.0;
            double candidate_yaw_target_error = 0.0;
            if (use_pose_yaw_reference_eval) {
              candidate_relative_yaw = frameRelativeYawForGroupState(
                *robot_model_,
                group_name,
                candidate.q,
                base_q_full,
                pose_yaw_target_frame,
                pose_yaw_reference_rotation);
              if (use_pose_yaw_reference_range &&
                  !yawWithinReferenceRange(
                    candidate_relative_yaw,
                    pose_yaw_reference_min_rad,
                    pose_yaw_reference_max_rad)) {
                std::ostringstream oss;
                oss << "target frame [" << pose_yaw_target_frame
                    << "] relative yaw " << candidate_relative_yaw
                    << " outside requested range ["
                    << pose_yaw_reference_min_rad << ", "
                    << pose_yaw_reference_max_rad << "]";
                reject_reason = oss.str();
                return false;
              }
            }
            if (use_pose_yaw_target) {
              candidate_yaw_target_error =
                yawTargetError(candidate_relative_yaw, pose_yaw_target_rad);
            }
            const bool better =
              !selected.found ||
              (use_pose_yaw_target &&
               (candidate_yaw_target_error < selected_yaw_target_error - 1e-9 ||
                (std::abs(candidate_yaw_target_error - selected_yaw_target_error) <= 1e-9 &&
                 candidate.distance < selected.distance)));
            if (better) {
              if (use_pose_yaw_target) {
                selected_relative_yaw = candidate_relative_yaw;
                selected_yaw_target_error = candidate_yaw_target_error;
              }
              selected = std::move(candidate);
              return !use_pose_yaw_target;
            }
            return false;
          };
          if (use_pose_yaw_target) {
            for (auto& future : futures) {
              if (!future.valid()) {
                continue;
              }
              (void)consider_pose_yaw_candidate(future.get());
            }
          } else {
            std::vector<bool> consumed(futures.size(), false);
            size_t remaining = futures.size();
            while (remaining > 0 && !selected.found) {
              bool progressed = false;
              for (size_t i = 0; i < futures.size(); ++i) {
                if (consumed[i] || !futures[i].valid()) {
                  continue;
                }
                if (futures[i].wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                  continue;
                }
                consumed[i] = true;
                --remaining;
                progressed = true;
                if (consider_pose_yaw_candidate(futures[i].get())) {
                  yaw_cancel.store(true);
                  break;
                }
              }
              if (!selected.found && !progressed) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
              }
            }
            if (selected.found) {
              yaw_cancel.store(true);
            }
          }

          if (!selected.found) {
            std::ostringstream oss;
            oss << "IK failed. No collision-free solution found"
                << " (last_reject=" << reject_reason << ")";
            response->success = false;
            response->message = oss.str();
            return;
          }

          q_goal = selected.q;
          response->used_cartesian_yaw_tolerance = std::abs(wrapToPi(selected.yaw_offset)) > 1e-9;
          response->cartesian_yaw_offset_rad = selected.yaw_offset;
          if (use_pose_yaw_target) {
            RCLCPP_INFO(
              this->get_logger(),
              "Pose goal selected yaw tolerance offset %.4f rad by target-frame relative yaw: frame=%s relative_yaw=%.4f target=%.4f error=%.4f samples=%zu.",
              selected.yaw_offset,
              pose_yaw_target_frame.c_str(),
              selected_relative_yaw,
              pose_yaw_target_rad,
              selected_yaw_target_error,
              yaw_offsets.size());
          }
          if (std::abs(wrapToPi(selected.yaw_offset)) > 1e-9) {
            RCLCPP_INFO(
              this->get_logger(),
              "Pose goal used yaw tolerance offset %.4f rad about axis_in_%s=(%.4f, %.4f, %.4f).",
              selected.yaw_offset,
              planning_frame.c_str(),
              pose_yaw_axis.x(),
              pose_yaw_axis.y(),
              pose_yaw_axis.z());
          }
        } else {
          auto seeds = buildIkSeeds(q_start);
          if (ik_max_seed_count > 0 &&
              static_cast<int>(seeds.size()) > ik_max_seed_count) {
            seeds.resize(static_cast<size_t>(ik_max_seed_count));
          }

          std::vector<KinematicsSolver::IKCandidate> candidates;
          if (request->use_weighted_orientation_goal) {
            candidates = kinematics_solver_->solveIKWeightedMultiSeed(
              goal_pose,
              group_name,
              seeds,
              task_weights,
              ee_frame,
              ik_max_iterations,
              ik_pos_tolerance,
              ik_rot_tolerance,
              ik_damping,
              ik_alpha,
              ik_max_step_norm,
              nullptr,
              use_sdls);
          } else {
            candidates = kinematics_solver_->solveIKMultiSeed(
              goal_pose,
              group_name,
              seeds,
              ee_frame,
              ik_max_iterations,
              ik_pos_tolerance,
              ik_rot_tolerance,
              ik_damping,
              ik_alpha,
              ik_max_step_norm,
              nullptr,
              use_sdls);
          }

          bool found_collision_free = false;
          double best_dist = std::numeric_limits<double>::infinity();
          size_t collision_free_count = 0;
          std::string last_collision_reject;

          for (const auto& cand : candidates) {
            const Eigen::VectorXd q_full =
              robot_model_->groupToFull(
                group_name,
                cand.q_group,
                base_q_full);

            if (!collision_scene_->isStateValid(q_full)) {
              const auto report = collision_scene_->getCollisionReport(q_full);
              RCLCPP_WARN(
                this->get_logger(),
                "IK candidate rejected by collision: %s [%s vs %s] %s",
                report.category.c_str(),
                report.object_a.c_str(),
                report.object_b.c_str(),
                report.detail.c_str());
              std::ostringstream reject_oss;
              reject_oss << report.category << " [" << report.object_a
                         << " vs " << report.object_b << "] " << report.detail;
              last_collision_reject = reject_oss.str();
              continue;
            }

            ++collision_free_count;
            collision_free_ik_goals.push_back(cand.q_group);
            const double d = (cand.q_group - q_start).norm();
            if (d < best_dist) {
              best_dist = d;
              q_goal = cand.q_group;
              found_collision_free = true;
            }
          }

          if (!found_collision_free) {
            std::ostringstream oss;
            oss << "IK failed. No collision-free solution found"
                << " (candidates=" << candidates.size()
                << ", collision_free=" << collision_free_count;
            if (!last_collision_reject.empty()) {
              oss << ", last_collision=" << last_collision_reject;
            } else if (candidates.empty()) {
              oss << ", reason=no IK candidate reached pose tolerance";
            }
            oss << ")";
            response->success = false;
            response->message = oss.str();
            return;
          }

          RCLCPP_INFO(
            this->get_logger(),
            "IK selected collision-free candidate, distance to start = %.4f among %zu candidate(s)",
            best_dist,
            candidates.size());
        }
      } else {
        q_goal = q_start;
      }
    } else {
      q_goal = Eigen::Map<const Eigen::VectorXd>(
        request->goal_joint_positions.data(),
        static_cast<Eigen::Index>(request->goal_joint_positions.size()));

      if (q_goal.size() != expected_dim) {
        std::ostringstream oss;
        oss << "goal_joint_positions dim mismatch, expected "
            << expected_dim << ", got " << q_goal.size();
        response->success = false;
        response->message = oss.str();
        return;
      }
    }

    const auto start_report = collision_scene_->getCollisionReport(q_start_full);
    if (start_report.in_collision) {
      RCLCPP_WARN(
        this->get_logger(),
        "Plan request start state already in collision: %s [%s vs %s] %s",
        start_report.category.c_str(),
        start_report.object_a.c_str(),
        start_report.object_b.c_str(),
        start_report.detail.c_str());
    }

    if (!(request->use_pose_goal && request->use_cartesian_path)) {
      const Eigen::VectorXd q_goal_full_for_log =
        robot_model_->groupToFull(group_name, q_goal, q_start_full);
      const auto goal_report = collision_scene_->getCollisionReport(q_goal_full_for_log);
      if (goal_report.in_collision) {
        RCLCPP_WARN(
          this->get_logger(),
          "Plan request goal state in collision before OMPL: %s [%s vs %s] %s",
          goal_report.category.c_str(),
          goal_report.object_a.c_str(),
          goal_report.object_b.c_str(),
          goal_report.detail.c_str());
      }
    }

    if (request->use_pose_goal && request->use_cartesian_path && cartesian_nullspace_settle_only) {
      CartesianWaypointCandidate settle_context;
      settle_context.reject_reason = "requested null-space settle only";
      const int focus_index =
        chooseLargestReferenceDeltaJoint(group, q_start, &cartesian_nullspace_reference_q);
      const auto recovery = solveNullspaceRecovery(
        *robot_model_,
        *collision_scene_,
        *motion_planner_,
        group_name,
        ee_frame,
        q_start,
        q_start_full,
        settle_context,
        cartesian_nullspace_recovery_max_attempts,
        cartesian_nullspace_recovery_max_iterations,
        cartesian_nullspace_recovery_pos_tolerance,
        cartesian_nullspace_recovery_roll_pitch_tolerance,
        cartesian_nullspace_recovery_yaw_min,
        cartesian_nullspace_recovery_yaw_max,
        cartesian_nullspace_recovery_reference_rotation,
        cartesian_nullspace_recovery_step_norm,
        cartesian_nullspace_recovery_joint_gain,
        cartesian_nullspace_recovery_min_joint_change,
        &cartesian_nullspace_reference_q,
        cartesian_nullspace_reference_gain,
        ik_damping,
        path_check_step,
        focus_index);

      if (!recovery.found) {
        const bool usable_partial =
          recovery.q.size() == q_start.size() &&
          recovery.pos_error <= cartesian_nullspace_recovery_pos_tolerance &&
          recovery.roll_error <= cartesian_nullspace_recovery_roll_pitch_tolerance &&
          recovery.pitch_error <= cartesian_nullspace_recovery_roll_pitch_tolerance &&
          yawWithinReferenceRange(
            recovery.relative_yaw,
            cartesian_nullspace_recovery_yaw_min,
            cartesian_nullspace_recovery_yaw_max);
        if (!usable_partial) {
          response->success = false;
          response->message =
            "null-space settle failed: " +
            (recovery.reason.empty() ? std::string("no detailed reason") : recovery.reason);
          RCLCPP_WARN(
            this->get_logger(),
            "Null-space settle-only request failed. focus_joint=%s reason=%s",
            recovery.focus_joint_name.c_str(),
            recovery.reason.empty() ? "no detailed reason" : recovery.reason.c_str());
          return;
        }
        RCLCPP_WARN(
          this->get_logger(),
          "Null-space settle-only accepted best partial candidate toward reference. focus_joint=%s delta=%.6f ref_max_err=%.6f pos_err=%.6f roll_err=%.6f pitch_err=%.6f rel_yaw=%.6f reason=%s",
          recovery.focus_joint_name.c_str(),
          recovery.focus_joint_delta,
          recovery.reference_max_error,
          recovery.pos_error,
          recovery.roll_error,
          recovery.pitch_error,
          recovery.relative_yaw,
          recovery.reason.empty() ? "no detailed reason" : recovery.reason.c_str());
      }

      RCLCPP_WARN(
        this->get_logger(),
        "Inserted requested null-space settle. focus_joint=%s delta=%.6f ref_max_err=%.6f pos_err=%.6f roll_err=%.6f pitch_err=%.6f rel_yaw=%.6f iterations=%d reference=%s.",
        recovery.focus_joint_name.c_str(),
        recovery.focus_joint_delta,
        recovery.reference_max_error,
        recovery.pos_error,
        recovery.roll_error,
        recovery.pitch_error,
        recovery.relative_yaw,
        recovery.iterations,
        recovery.used_reference ? "on" : "off");
      logNullspaceRecoveryJoints(this->get_logger(), group, recovery.q);

      std::vector<Eigen::VectorXd> settle_path{q_start, recovery.q};
      q_goal = recovery.q;
      auto traj = trajectory_generator_->generate(group_name, settle_path);
      traj.header.stamp = this->now();

      response->trajectory = traj;
      response->success = true;
      response->message = "Null-space settle succeeded.";
      response->used_cartesian_yaw_tolerance = false;
      response->cartesian_yaw_offset_rad = 0.0;

      if (plan_only) {
        RCLCPP_INFO(
          this->get_logger(),
          "Plan-only null-space settle request for group [%s] succeeded; skipping trajectory publish/playback.",
          group_name.c_str());
        return;
      }

      trajectory_pub_->publish(traj);

      std::vector<std::string> vis_ids(
        request->obstacle_ids.begin(),
        request->obstacle_ids.begin() + static_cast<std::ptrdiff_t>(n_obs));
      std::vector<shape_msgs::msg::SolidPrimitive> vis_shapes(
        request->obstacle_shapes.begin(),
        request->obstacle_shapes.begin() + static_cast<std::ptrdiff_t>(n_obs));
      std::vector<geometry_msgs::msg::PoseStamped> vis_poses(
        request->obstacle_poses.begin(),
        request->obstacle_poses.begin() + static_cast<std::ptrdiff_t>(n_obs));

      publishVisualizationState(
        group_name,
        q_start,
        q_goal,
        traj,
        vis_ids,
        vis_shapes,
        vis_poses);
      return;
    }

    if (request->use_pose_goal && request->use_cartesian_path) {
      const pinocchio::SE3 start_pose = kinematics_solver_->solveFK(group_name, q_start, ee_frame);
      const pinocchio::SE3 goal_pose = poseMsgToSE3(request->target_pose.pose);
      const double cartesian_max_step =
        request->cartesian_max_step > 0.0 ? request->cartesian_max_step : 0.03;
      const double cartesian_max_rotation_step =
        this->get_parameter("cartesian_max_rotation_step_rad").as_double();
      const int default_cartesian_min_steps_param =
        this->get_parameter("cartesian_min_interpolation_steps").as_int();
      const int cartesian_min_steps_param =
        request->request_cartesian_min_interpolation_steps > 0 ?
        request->request_cartesian_min_interpolation_steps :
        default_cartesian_min_steps_param;
      const int cartesian_max_steps_param =
        this->get_parameter("cartesian_max_interpolation_steps").as_int();
      const int default_cartesian_refinement_max_steps_param =
        this->get_parameter("cartesian_segment_length_refinement_max_steps").as_int();
      const int cartesian_refinement_max_steps_param =
        request->request_cartesian_refinement_max_steps > 0 ?
        request->request_cartesian_refinement_max_steps :
        default_cartesian_refinement_max_steps_param;
      const size_t configured_cartesian_max_steps =
        cartesian_max_steps_param > 0 ? static_cast<size_t>(cartesian_max_steps_param) : 0;
      const size_t initial_cartesian_steps = computeCartesianInterpolationSteps(
        start_pose,
        goal_pose,
        cartesian_max_step,
        cartesian_max_rotation_step,
        static_cast<size_t>(std::max(1, cartesian_min_steps_param)),
        configured_cartesian_max_steps);
      const size_t refinement_max_steps =
        configured_cartesian_max_steps > 0 ?
        configured_cartesian_max_steps :
        static_cast<size_t>(std::max(
          static_cast<int>(initial_cartesian_steps),
          std::max(1, cartesian_refinement_max_steps_param)));
      const bool log_each_cartesian_waypoint =
        this->get_parameter("cartesian_log_each_waypoint").as_bool();
      const bool cartesian_diagnostics_enabled =
        this->get_parameter("cartesian_diagnostics_enabled").as_bool();
      const double cartesian_diagnostics_suspicious_path_m =
        std::max(0.0, this->get_parameter("cartesian_diagnostics_suspicious_path_m").as_double());
      const double cartesian_diagnostics_suspicious_ratio =
        std::max(1.0, this->get_parameter("cartesian_diagnostics_suspicious_ratio").as_double());
      const double cartesian_request_distance =
        (goal_pose.translation() - start_pose.translation()).norm();
      const double min_step_nominal_distance =
        cartesian_request_distance / static_cast<double>(std::max(1, cartesian_min_steps_param));
      const double cartesian_request_rotation =
        Eigen::Quaterniond(start_pose.rotation()).angularDistance(Eigen::Quaterniond(goal_pose.rotation()));

      RCLCPP_INFO(
        this->get_logger(),
        "Cartesian planning enabled: distance=%.4f m, rotation=%.4f rad, min_steps=%d, distance/min_steps=%.4f m, max_translation_step=%.4f m, max_rotation_step=%.4f rad, initial_steps=%zu, segment_path_limit=%.4f m frame=%s refinement_max_steps=%zu diagnostics=%s",
        cartesian_request_distance,
        cartesian_request_rotation,
        std::max(1, cartesian_min_steps_param),
        min_step_nominal_distance,
        cartesian_max_step,
        cartesian_max_rotation_step,
        initial_cartesian_steps,
        cartesian_max_segment_path_length,
        ee_frame.c_str(),
        refinement_max_steps,
        cartesian_diagnostics_enabled ? "on" : "off");

      const Eigen::Vector3d yaw_axis = Eigen::Vector3d::UnitZ();

      auto solve_cartesian_waypoint =
        [&](const pinocchio::SE3& interp_pose,
            const Eigen::VectorXd& waypoint_seed,
            const Eigen::VectorXd& previous_q) {
          CartesianWaypointCandidate selected;
          std::atomic_bool waypoint_cancel{false};
          std::atomic_bool waypoint_timed_out{false};
          std::mutex waypoint_watchdog_mutex;
          std::condition_variable waypoint_watchdog_cv;
          bool waypoint_solve_done = false;
          std::thread waypoint_watchdog;
          const double waypoint_timeout_sec =
            use_request_solver_settings && timeout > 0.0 ? std::max(0.05, timeout) : 0.0;
          if (waypoint_timeout_sec > 0.0) {
            waypoint_watchdog = std::thread(
              [&waypoint_cancel,
               &waypoint_timed_out,
               &waypoint_watchdog_mutex,
               &waypoint_watchdog_cv,
               &waypoint_solve_done,
               waypoint_timeout_sec]() {
                std::unique_lock<std::mutex> lock(waypoint_watchdog_mutex);
                const bool finished = waypoint_watchdog_cv.wait_for(
                  lock,
                  std::chrono::duration<double>(waypoint_timeout_sec),
                  [&waypoint_solve_done]() { return waypoint_solve_done; });
                if (!finished) {
                  waypoint_timed_out.store(true);
                  waypoint_cancel.store(true);
                }
              });
          }

          selected = solveCartesianWaypointCandidate(
            *kinematics_solver_,
            *motion_planner_,
            *robot_model_,
            *collision_scene_,
            interp_pose,
            group_name,
            ee_frame,
            waypoint_seed,
            previous_q,
            q_start_full,
            0.0,
            yaw_axis,
            ik_max_iterations,
            cartesian_waypoint_pos_tolerance,
            cartesian_waypoint_rot_tolerance,
            ik_damping,
            ik_alpha,
            ik_max_step_norm,
            ik_max_seed_count,
            path_check_step,
            use_sdls,
            cartesian_nullspace_gain,
            cartesian_singular_threshold,
            cartesian_branch_jump_max_rad,
            cartesian_branch_jump_norm_max,
            &waypoint_cancel);
          {
            std::lock_guard<std::mutex> lock(waypoint_watchdog_mutex);
            waypoint_solve_done = true;
          }
          waypoint_watchdog_cv.notify_all();
          if (waypoint_watchdog.joinable()) {
            waypoint_watchdog.join();
          }
          if (waypoint_timed_out.load() && !selected.found) {
            std::ostringstream oss;
            oss << "cartesian waypoint IK solve timed out after "
                << waypoint_timeout_sec << " s";
            if (!selected.reject_reason.empty()) {
              oss << "; last_reject=" << selected.reject_reason;
            }
            selected.reject_reason = oss.str();
          }
          return selected;
        };

      std::vector<Eigen::VectorXd> cartesian_path;
      cartesian_path.reserve(refinement_max_steps + 1);
      cartesian_path.push_back(q_start);

      Eigen::VectorXd q_seed = q_start;
      pinocchio::SE3 remaining_start_pose = start_pose;
      pinocchio::SE3 remaining_goal_pose = goal_pose;
      size_t remaining_steps = initial_cartesian_steps;
      size_t accepted_steps = 0;
      int consecutive_nullspace_recoveries = 0;
      double max_measured_segment_path_length = 0.0;
      size_t max_measured_segment_step_idx = 0;
      size_t max_measured_segment_total_steps = 0;
      double max_measured_target_chord = 0.0;
      double max_measured_target_rotation_step = 0.0;
      double max_measured_endpoint_chord = 0.0;
      double max_measured_start_position_error = 0.0;
      double max_measured_start_rotation_error = 0.0;
      double max_measured_target_position_error = 0.0;
      double max_measured_target_rotation_error = 0.0;
      double max_measured_nominal_rotation_error = 0.0;
      double max_measured_joint_delta_norm = 0.0;
      double max_measured_joint_delta_max_abs = 0.0;
      int max_measured_joint_delta_index = -1;
      double max_measured_yaw_offset = 0.0;
      double max_measured_min_singular_value = 0.0;
      double max_measured_condition_number = 0.0;
      bool max_measured_near_singular = false;
      bool max_measured_fallback_multiseed = false;
          int max_measured_ik_candidate_count = 0;
          int max_measured_endpoint_collision_reject_count = 0;
          int max_measured_path_collision_reject_count = 0;
      std::string max_measured_endpoint_collision_reject_reason;
      std::string max_measured_path_collision_reject_reason;

      while (remaining_steps > 0) {
        const size_t total_steps = accepted_steps + remaining_steps;
        const size_t step_idx = accepted_steps + 1;
        const double alpha = 1.0 / static_cast<double>(remaining_steps);
        const pinocchio::SE3 interp_pose =
          interpolatePose(remaining_start_pose, remaining_goal_pose, alpha);
        const double target_chord_length =
          (interp_pose.translation() - remaining_start_pose.translation()).norm();
        const double target_rotation_step =
          Eigen::Quaterniond(remaining_start_pose.rotation()).angularDistance(
            Eigen::Quaterniond(interp_pose.rotation()));
        const auto actual_start_pose = framePoseForGroupState(
          *robot_model_,
          group_name,
          cartesian_path.back(),
          q_start_full,
          ee_frame);
        const double start_position_error =
          (actual_start_pose.translation() - remaining_start_pose.translation()).norm();
        const double start_rotation_error =
          Eigen::Quaterniond(actual_start_pose.rotation()).angularDistance(
            Eigen::Quaterniond(remaining_start_pose.rotation()));

        if (cartesian_diagnostics_enabled) {
          RCLCPP_INFO(
            this->get_logger(),
            "Cartesian waypoint %zu/%zu split diagnostic: accepted=%zu remaining=%zu alpha=%.6f target_chord=%.6f m target_rot_step=%.6f rad actual_start_vs_nominal=(pos=%.6f m rot=%.6f rad) start_xyz=(%.4f, %.4f, %.4f) target_xyz=(%.4f, %.4f, %.4f).",
            step_idx,
            total_steps,
            accepted_steps,
            remaining_steps,
            alpha,
            target_chord_length,
            target_rotation_step,
            start_position_error,
            start_rotation_error,
            actual_start_pose.translation().x(),
            actual_start_pose.translation().y(),
            actual_start_pose.translation().z(),
            interp_pose.translation().x(),
            interp_pose.translation().y(),
            interp_pose.translation().z());
        }

        const bool arc_limit_guard_enabled =
          use_sdls &&
          use_cartesian_nullspace_recovery &&
          consecutive_nullspace_recoveries < 2 &&
          group.joint_names.size() > 6;
        if (arc_limit_guard_enabled) {
          const auto upper = robot_model_->upperBoundsForGroup(group_name);
          double guard_margin = std::numeric_limits<double>::infinity();
          const int guard_focus_index = chooseArcUpperLimitGuardJoint(
            group,
            cartesian_path.back(),
            upper,
            0.22,
            &guard_margin);
          if (guard_focus_index >= 0) {
            CartesianWaypointCandidate guard_context;
            guard_context.reject_reason =
              "preemptive arc upper-limit guard before Cartesian waypoint";
            guard_context.ik_diagnostics.joint_limit_clamped = true;
            guard_context.ik_diagnostics.joint_limit_index = guard_focus_index;
            guard_context.ik_diagnostics.joint_limit_value =
              cartesian_path.back()[guard_focus_index];
            const auto lower = robot_model_->lowerBoundsForGroup(group_name);
            guard_context.ik_diagnostics.joint_limit_lower = lower[guard_focus_index];
            guard_context.ik_diagnostics.joint_limit_upper = upper[guard_focus_index];

            const auto recovery = solveNullspaceRecovery(
              *robot_model_,
              *collision_scene_,
              *motion_planner_,
              group_name,
              ee_frame,
              cartesian_path.back(),
              q_start_full,
              guard_context,
              std::max(1, cartesian_nullspace_recovery_max_attempts / 2),
              cartesian_nullspace_recovery_max_iterations,
              cartesian_nullspace_recovery_pos_tolerance,
              cartesian_nullspace_recovery_roll_pitch_tolerance,
              cartesian_nullspace_recovery_yaw_min,
              cartesian_nullspace_recovery_yaw_max,
              cartesian_nullspace_recovery_reference_rotation,
              cartesian_nullspace_recovery_step_norm,
              std::max(cartesian_nullspace_recovery_joint_gain, 0.80),
              std::min(cartesian_nullspace_recovery_min_joint_change, 0.04),
              nullptr,
              0.0,
              ik_damping,
              path_check_step,
              guard_focus_index);
            if (recovery.found) {
              RCLCPP_WARN(
                this->get_logger(),
                "Cartesian waypoint %zu/%zu inserted preemptive arc upper-limit null-space guard. focus_joint=%s upper_margin=%.6f delta=%.6f pos_err=%.6f roll_err=%.6f pitch_err=%.6f rel_yaw=%.6f iterations=%d.",
                step_idx,
                total_steps,
                recovery.focus_joint_name.c_str(),
                guard_margin,
                recovery.focus_joint_delta,
                recovery.pos_error,
                recovery.roll_error,
                recovery.pitch_error,
                recovery.relative_yaw,
                recovery.iterations);
              logNullspaceRecoveryJoints(this->get_logger(), group, recovery.q);

              cartesian_path.push_back(recovery.q);
              q_seed = recovery.q;
              remaining_start_pose = framePoseForGroupState(
                *robot_model_,
                group_name,
                recovery.q,
                q_start_full,
                ee_frame);
              ++accepted_steps;
              ++consecutive_nullspace_recoveries;
              continue;
            }
            RCLCPP_WARN(
              this->get_logger(),
              "Cartesian waypoint %zu/%zu preemptive arc upper-limit guard failed. focus_joint=%s upper_margin=%.6f reason=%s",
              step_idx,
              total_steps,
              recovery.focus_joint_name.c_str(),
              guard_margin,
              recovery.reason.empty() ? "no detailed reject reason" : recovery.reason.c_str());
          }
        }

        auto selected = solve_cartesian_waypoint(
          interp_pose,
          q_seed,
          cartesian_path.back());

        if (!selected.found) {
          const bool pure_singularity_failure =
            selected.ik_diagnostics.near_singular &&
            !selected.ik_diagnostics.joint_limit_clamped &&
            selected.endpoint_collision_reject_count == 0 &&
            selected.path_collision_reject_count == 0;
          if (pure_singularity_failure) {
            RCLCPP_WARN(
              this->get_logger(),
              "Cartesian waypoint %zu/%zu exact solve failed near a singularity without joint-limit/collision evidence; skipping null-space recovery so singular handling stays inside DLS/SDLS IK. min_sigma=%.6e cond=%.3e use_sdls=%s reason=%s",
              step_idx,
              total_steps,
              selected.ik_diagnostics.min_singular_value,
              selected.ik_diagnostics.condition_number,
              use_sdls ? "true" : "false",
              selected.reject_reason.empty() ? "no detailed reject reason" : selected.reject_reason.c_str());
          }
          if (!pure_singularity_failure &&
              use_cartesian_nullspace_recovery &&
              consecutive_nullspace_recoveries < 2) {
            const int reference_focus_index =
              use_cartesian_nullspace_reference ?
              chooseLargestReferenceDeltaJoint(group, cartesian_path.back(), &cartesian_nullspace_reference_q) :
              -1;
            const auto recovery = solveNullspaceRecovery(
              *robot_model_,
              *collision_scene_,
              *motion_planner_,
              group_name,
              ee_frame,
              cartesian_path.back(),
              q_start_full,
              selected,
              cartesian_nullspace_recovery_max_attempts,
              cartesian_nullspace_recovery_max_iterations,
              cartesian_nullspace_recovery_pos_tolerance,
              cartesian_nullspace_recovery_roll_pitch_tolerance,
              cartesian_nullspace_recovery_yaw_min,
              cartesian_nullspace_recovery_yaw_max,
              cartesian_nullspace_recovery_reference_rotation,
              cartesian_nullspace_recovery_step_norm,
              cartesian_nullspace_recovery_joint_gain,
              cartesian_nullspace_recovery_min_joint_change,
              use_cartesian_nullspace_reference ? &cartesian_nullspace_reference_q : nullptr,
              cartesian_nullspace_reference_gain,
              ik_damping,
              path_check_step,
              reference_focus_index);
            if (recovery.found) {
              RCLCPP_WARN(
                this->get_logger(),
                "Cartesian waypoint %zu/%zu exact solve failed (%s); inserted null-space recovery before retry. focus_joint=%s delta=%.6f pos_err=%.6f roll_err=%.6f pitch_err=%.6f rel_yaw=%.6f iterations=%d reference=%s.",
                step_idx,
                total_steps,
                selected.reject_reason.empty() ? "no detailed reject reason" : selected.reject_reason.c_str(),
                recovery.focus_joint_name.c_str(),
                recovery.focus_joint_delta,
                recovery.pos_error,
                recovery.roll_error,
                recovery.pitch_error,
                recovery.relative_yaw,
                recovery.iterations,
                recovery.used_reference ? "on" : "off");
              logNullspaceRecoveryJoints(this->get_logger(), group, recovery.q);

              cartesian_path.push_back(recovery.q);
              q_seed = recovery.q;
              remaining_start_pose = framePoseForGroupState(
                *robot_model_,
                group_name,
                recovery.q,
                q_start_full,
                ee_frame);
              ++accepted_steps;
              ++consecutive_nullspace_recoveries;
              continue;
            }
            RCLCPP_WARN(
              this->get_logger(),
              "Cartesian waypoint %zu/%zu null-space recovery failed after exact solve failure. focus_joint=%s reason=%s",
              step_idx,
              total_steps,
              recovery.focus_joint_name.c_str(),
              recovery.reason.empty() ? "no detailed reject reason" : recovery.reason.c_str());
          }
          const size_t refined_total_steps = accepted_steps + remaining_steps + 2;
          if (refined_total_steps <= refinement_max_steps) {
            RCLCPP_WARN(
              this->get_logger(),
              "Cartesian waypoint %zu/%zu has no collision-free IK waypoint (%s). Re-splitting remaining trajectory: remaining_steps=%zu -> %zu before declaring unreachable/limited.",
              step_idx,
              total_steps,
              selected.reject_reason.empty() ? "no detailed reject reason" : selected.reject_reason.c_str(),
              remaining_steps,
              remaining_steps + 1);
            remaining_steps += 1;
            continue;
          }
          std::ostringstream oss;
          oss << "cartesian step " << step_idx << "/" << total_steps
              << " failed: no collision-free IK waypoint";
          if (!selected.reject_reason.empty()) {
            oss << " (last_reject=" << selected.reject_reason << ")";
          }
          oss << " (refinement exhausted: requested_total_steps=" << refined_total_steps
              << ", max=" << refinement_max_steps << ")";
          response->success = false;
          response->message = oss.str();
          return;
        }

        double measured_segment_path_length = 0.0;
        std::string segment_length_reject_reason;
        const bool segment_length_ok = measureCartesianSegmentPathLength(
          *robot_model_,
          group_name,
          cartesian_path.back(),
          selected.q,
          q_start_full,
          ee_frame,
          path_check_step,
          cartesian_max_segment_path_length,
          &measured_segment_path_length,
          &segment_length_reject_reason);
        const auto actual_end_pose = framePoseForGroupState(
          *robot_model_,
          group_name,
          selected.q,
          q_start_full,
          ee_frame);
        const double endpoint_chord_length =
          (actual_end_pose.translation() - actual_start_pose.translation()).norm();
        const double nominal_target_position_error =
          (actual_end_pose.translation() - interp_pose.translation()).norm();
        const double nominal_target_rotation_error =
          Eigen::Quaterniond(actual_end_pose.rotation()).angularDistance(
            Eigen::Quaterniond(interp_pose.rotation()));
        const double path_to_target_chord_ratio =
          target_chord_length > 1e-9 ?
          measured_segment_path_length / target_chord_length :
          std::numeric_limits<double>::infinity();
        const bool suspicious_cartesian_segment =
          measured_segment_path_length > cartesian_diagnostics_suspicious_path_m ||
          (target_chord_length > 1e-9 &&
           measured_segment_path_length >
           cartesian_diagnostics_suspicious_ratio * target_chord_length);
        const double endpoint_progress_ratio =
          target_chord_length > 1e-9 ?
          endpoint_chord_length / target_chord_length :
          std::numeric_limits<double>::infinity();
        const bool stalled_cartesian_progress =
          use_sdls &&
          ((target_chord_length > 0.004 && endpoint_progress_ratio < 0.35) ||
           (target_chord_length <= 0.004 &&
            target_rotation_step > 0.030 &&
            nominal_target_rotation_error > 0.65 * target_rotation_step));
        const bool new_max_segment =
          measured_segment_path_length > max_measured_segment_path_length;
        if (new_max_segment) {
          max_measured_segment_path_length = measured_segment_path_length;
          max_measured_segment_step_idx = step_idx;
          max_measured_segment_total_steps = total_steps;
          max_measured_target_chord = target_chord_length;
          max_measured_target_rotation_step = target_rotation_step;
          max_measured_endpoint_chord = endpoint_chord_length;
          max_measured_start_position_error = start_position_error;
          max_measured_start_rotation_error = start_rotation_error;
          max_measured_target_position_error = selected.target_position_error;
          max_measured_target_rotation_error = selected.target_rotation_error;
          max_measured_nominal_rotation_error = nominal_target_rotation_error;
          max_measured_joint_delta_norm = selected.distance;
          max_measured_joint_delta_max_abs = selected.joint_delta_max_abs;
          max_measured_joint_delta_index = selected.joint_delta_max_index;
          max_measured_yaw_offset = selected.yaw_offset;
          max_measured_min_singular_value = selected.ik_diagnostics.min_singular_value;
          max_measured_condition_number = selected.ik_diagnostics.condition_number;
          max_measured_near_singular = selected.ik_diagnostics.near_singular;
          max_measured_fallback_multiseed = selected.fallback_multiseed_used;
          max_measured_ik_candidate_count = selected.ik_candidate_count;
          max_measured_endpoint_collision_reject_count =
            selected.endpoint_collision_reject_count;
          max_measured_path_collision_reject_count = selected.path_collision_reject_count;
          max_measured_endpoint_collision_reject_reason =
            selected.endpoint_collision_reject_reason;
          max_measured_path_collision_reject_reason =
            selected.path_collision_reject_reason;
        }

        if (cartesian_diagnostics_enabled || suspicious_cartesian_segment || !segment_length_ok) {
          const char* level_hint =
            (!segment_length_ok || suspicious_cartesian_segment) ? "SUSPICIOUS" : "normal";
          const std::string max_joint_name =
            selected.joint_delta_max_index >= 0 &&
            static_cast<size_t>(selected.joint_delta_max_index) < group.joint_names.size() ?
            group.joint_names[static_cast<size_t>(selected.joint_delta_max_index)] :
            std::string("n/a");
          const std::string limit_joint_name =
            selected.ik_diagnostics.joint_limit_index >= 0 &&
            static_cast<size_t>(selected.ik_diagnostics.joint_limit_index) < group.joint_names.size() ?
            group.joint_names[static_cast<size_t>(selected.ik_diagnostics.joint_limit_index)] :
            std::string("n/a");
          RCLCPP_WARN(
            this->get_logger(),
            "Cartesian waypoint %zu/%zu frame-path diagnostic [%s]: target_chord=%.6f m target_rot_step=%.6f rad measured_ccb_path=%.6f m endpoint_chord=%.6f m path/target_chord=%.3f hard_limit=%.6f m start_actual_vs_nominal=(pos=%.6f m rot=%.6f rad) end_vs_nominal=(pos=%.6f m rot=%.6f rad) end_vs_yaw_target=(pos=%.6f m rot=%.6f rad) yaw_offset=%.6f rad joint_delta_norm=%.6f max_joint_delta=%s:%.6f ik_candidates=%d endpoint_collision_rejects=%d endpoint_collision_last=[%s] path_collision_rejects=%d path_collision_last=[%s] min_sigma=%.6e cond=%.3e near_singular=%s joint_limit_clamped=%s limit_joint=%s limit_value=%.6f limit_range=[%.6f, %.6f] fallback_multiseed=%s.",
            step_idx,
            total_steps,
            level_hint,
            target_chord_length,
            target_rotation_step,
            measured_segment_path_length,
            endpoint_chord_length,
            path_to_target_chord_ratio,
            cartesian_max_segment_path_length,
            start_position_error,
            start_rotation_error,
            nominal_target_position_error,
            nominal_target_rotation_error,
            selected.target_position_error,
            selected.target_rotation_error,
            selected.yaw_offset,
            selected.distance,
            max_joint_name.c_str(),
            selected.joint_delta_max_abs,
            selected.ik_candidate_count,
            selected.endpoint_collision_reject_count,
            selected.endpoint_collision_reject_reason.empty() ?
              "none" : selected.endpoint_collision_reject_reason.c_str(),
            selected.path_collision_reject_count,
            selected.path_collision_reject_reason.empty() ?
              "none" : selected.path_collision_reject_reason.c_str(),
            selected.ik_diagnostics.min_singular_value,
            selected.ik_diagnostics.condition_number,
            selected.ik_diagnostics.near_singular ? "true" : "false",
            selected.ik_diagnostics.joint_limit_clamped ? "true" : "false",
            limit_joint_name.c_str(),
            selected.ik_diagnostics.joint_limit_value,
            selected.ik_diagnostics.joint_limit_lower,
            selected.ik_diagnostics.joint_limit_upper,
            selected.fallback_multiseed_used ? "true" : "false");
        }

        if (!segment_length_ok) {
          const size_t refined_total_steps = accepted_steps + remaining_steps + 2;
          if (refined_total_steps > refinement_max_steps) {
            std::ostringstream oss;
            oss << "cartesian step " << step_idx << "/" << total_steps
                << " failed hard frame-path check";
            if (!segment_length_reject_reason.empty()) {
              oss << ": " << segment_length_reject_reason;
            }
            oss << " (refinement would require " << refined_total_steps
                << " steps, max=" << refinement_max_steps << ")";
            response->success = false;
            response->message = oss.str();
            return;
          }
          RCLCPP_WARN(
            this->get_logger(),
            "Cartesian waypoint %zu/%zu failed frame-path check (%s). Re-splitting remaining trajectory: remaining_steps=%zu -> %zu. Diagnostic at reject: target_chord=%.6f m measured_ccb_path=%.6f m endpoint_chord=%.6f m path/target_chord=%.3f yaw_offset=%.6f rad joint_delta_norm=%.6f.",
            step_idx,
            total_steps,
            segment_length_reject_reason.empty() ? "no detailed reject reason" :
              segment_length_reject_reason.c_str(),
            remaining_steps,
            remaining_steps + 1,
            target_chord_length,
            measured_segment_path_length,
            endpoint_chord_length,
            path_to_target_chord_ratio,
            selected.yaw_offset,
            selected.distance);
          remaining_steps += 1;
          continue;
        }

        if (stalled_cartesian_progress) {
          const size_t refined_total_steps = accepted_steps + remaining_steps + 2;
          if (refined_total_steps > refinement_max_steps) {
            std::ostringstream oss;
            oss << "cartesian step " << step_idx << "/" << total_steps
                << " failed stalled frame progress"
                << " (target_chord=" << target_chord_length
                << ", endpoint_chord=" << endpoint_chord_length
                << ", progress_ratio=" << endpoint_progress_ratio
                << ", target_rot_step=" << target_rotation_step
                << ", end_vs_nominal_rot=" << nominal_target_rotation_error
                << ", max=" << refinement_max_steps << ")";
            response->success = false;
            response->message = oss.str();
            return;
          }
          RCLCPP_WARN(
            this->get_logger(),
            "Cartesian waypoint %zu/%zu rejected stalled frame progress before acceptance. target_chord=%.6f endpoint_chord=%.6f progress_ratio=%.3f target_rot_step=%.6f end_vs_nominal_rot=%.6f. Re-splitting remaining trajectory: remaining_steps=%zu -> %zu.",
            step_idx,
            total_steps,
            target_chord_length,
            endpoint_chord_length,
            endpoint_progress_ratio,
            target_rotation_step,
            nominal_target_rotation_error,
            remaining_steps,
            remaining_steps + 1);
          remaining_steps += 1;
          continue;
        }

        if (log_each_cartesian_waypoint) {
          RCLCPP_INFO(
            this->get_logger(),
            "Cartesian waypoint %zu/%zu accepted, joint delta to previous = %.4f, measured frame path = %.4f m",
            step_idx,
            total_steps,
            selected.distance,
            measured_segment_path_length);
        }

        cartesian_path.push_back(selected.q);
        q_seed = selected.q;
        remaining_start_pose = selected.target_pose;
        consecutive_nullspace_recoveries = 0;
        --remaining_steps;
        ++accepted_steps;
      }

      if (cartesian_max_segment_path_length > 0.0) {
        const std::string max_joint_name =
          max_measured_joint_delta_index >= 0 &&
          static_cast<size_t>(max_measured_joint_delta_index) < group.joint_names.size() ?
          group.joint_names[static_cast<size_t>(max_measured_joint_delta_index)] :
          std::string("n/a");
        const double max_path_to_target_chord_ratio =
          max_measured_target_chord > 1e-9 ?
          max_measured_segment_path_length / max_measured_target_chord :
          std::numeric_limits<double>::infinity();
        RCLCPP_INFO(
          this->get_logger(),
          "Cartesian path frame-path check passed during sequential planning: max_segment_path=%.4f m limit=%.4f m steps=%zu frame=%s.",
          max_measured_segment_path_length,
          cartesian_max_segment_path_length,
          cartesian_path.size() - 1,
          ee_frame.c_str());
        RCLCPP_WARN(
          this->get_logger(),
          "Cartesian path max-segment diagnostic: max_step=%zu/%zu measured_ccb_path=%.6f m target_chord=%.6f m target_rot_step=%.6f rad endpoint_chord=%.6f m path/target_chord=%.3f start_actual_vs_nominal=(pos=%.6f m rot=%.6f rad) end_vs_yaw_target=(pos=%.6f m rot=%.6f rad) end_vs_nominal_rot=%.6f rad yaw_offset=%.6f rad joint_delta_norm=%.6f max_joint_delta=%s:%.6f ik_candidates=%d endpoint_collision_rejects=%d path_collision_rejects=%d min_sigma=%.6e cond=%.3e near_singular=%s fallback_multiseed=%s. If this value is large while the request still passes, the active segment hard limit is larger than the measured path.",
          max_measured_segment_step_idx,
          max_measured_segment_total_steps,
          max_measured_segment_path_length,
          max_measured_target_chord,
          max_measured_target_rotation_step,
          max_measured_endpoint_chord,
          max_path_to_target_chord_ratio,
          max_measured_start_position_error,
          max_measured_start_rotation_error,
          max_measured_target_position_error,
          max_measured_target_rotation_error,
          max_measured_nominal_rotation_error,
          max_measured_yaw_offset,
          max_measured_joint_delta_norm,
          max_joint_name.c_str(),
          max_measured_joint_delta_max_abs,
          max_measured_ik_candidate_count,
          max_measured_endpoint_collision_reject_count,
          max_measured_path_collision_reject_count,
          max_measured_min_singular_value,
          max_measured_condition_number,
          max_measured_near_singular ? "true" : "false",
          max_measured_fallback_multiseed ? "true" : "false");
      }

      RCLCPP_INFO(
        this->get_logger(),
        "Cartesian path accepted with %zu IK waypoint(s); passing sparse waypoints to Ruckig time parameterization",
        cartesian_path.size() - 1);

      q_goal = cartesian_path.back();
      auto traj = trajectory_generator_->generate(group_name, cartesian_path);
      traj.header.stamp = this->now();

      response->trajectory = traj;
      response->success = true;
      response->message = "Cartesian planning succeeded.";
      response->used_cartesian_yaw_tolerance = false;
      response->cartesian_yaw_offset_rad = 0.0;

      if (plan_only) {
        RCLCPP_INFO(
          this->get_logger(),
          "Plan-only Cartesian request for group [%s] succeeded; skipping trajectory publish/playback.",
          group_name.c_str());
        return;
      }

      trajectory_pub_->publish(traj);

      std::vector<std::string> vis_ids(
        request->obstacle_ids.begin(),
        request->obstacle_ids.begin() + static_cast<std::ptrdiff_t>(n_obs));
      std::vector<shape_msgs::msg::SolidPrimitive> vis_shapes(
        request->obstacle_shapes.begin(),
        request->obstacle_shapes.begin() + static_cast<std::ptrdiff_t>(n_obs));
      std::vector<geometry_msgs::msg::PoseStamped> vis_poses(
        request->obstacle_poses.begin(),
        request->obstacle_poses.begin() + static_cast<std::ptrdiff_t>(n_obs));

      publishVisualizationState(
        group_name,
        q_start,
        q_goal,
        traj,
        vis_ids,
        vis_shapes,
        vis_poses);
      return;
    }

    if (request->use_pose_goal && !request->use_cartesian_path &&
        request->use_weighted_orientation_goal && collision_free_ik_goals.size() > 1) {
      std::string last_reason;
      std::sort(
        collision_free_ik_goals.begin(),
        collision_free_ik_goals.end(),
        [&q_start](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) {
          return (lhs - q_start).squaredNorm() < (rhs - q_start).squaredNorm();
        });
      const size_t weighted_goal_try_count = request_max_ik_goal_candidates > 0 ?
        std::min(collision_free_ik_goals.size(), static_cast<size_t>(request_max_ik_goal_candidates)) :
        collision_free_ik_goals.size();
      for (size_t goal_idx = 0; goal_idx < weighted_goal_try_count; ++goal_idx) {
        const Eigen::VectorXd& candidate_goal = collision_free_ik_goals[goal_idx];
        const auto candidate_plan = motion_planner_->plan(
          group_name, q_start, candidate_goal, q_start_full, timeout, path_check_step, ompl_max_plan_attempts);
        if (!candidate_plan.success) {
          last_reason = candidate_plan.reason;
          RCLCPP_WARN(
            this->get_logger(),
            "Weighted IK candidate %zu/%zu failed during OMPL/path validation: %s",
            goal_idx + 1,
            weighted_goal_try_count,
            candidate_plan.reason.c_str());
          continue;
        }

        q_goal = candidate_goal;
        auto traj = trajectory_generator_->generate(group_name, candidate_plan.path);
        traj.header.stamp = this->now();

        response->trajectory = traj;
        response->success = true;
        response->message = "Weighted pose-goal planning succeeded.";

        if (plan_only) {
          RCLCPP_INFO(
            this->get_logger(),
            "Plan-only weighted pose-goal request for group [%s] succeeded; skipping trajectory publish/playback.",
            group_name.c_str());
          return;
        }

        trajectory_pub_->publish(traj);

        std::vector<std::string> vis_ids(
          request->obstacle_ids.begin(),
          request->obstacle_ids.begin() + static_cast<std::ptrdiff_t>(n_obs));
        std::vector<shape_msgs::msg::SolidPrimitive> vis_shapes(
          request->obstacle_shapes.begin(),
          request->obstacle_shapes.begin() + static_cast<std::ptrdiff_t>(n_obs));
        std::vector<geometry_msgs::msg::PoseStamped> vis_poses(
          request->obstacle_poses.begin(),
          request->obstacle_poses.begin() + static_cast<std::ptrdiff_t>(n_obs));

        publishVisualizationState(
          group_name,
          q_start,
          q_goal,
          traj,
          vis_ids,
          vis_shapes,
          vis_poses);
        return;
      }

      response->success = false;
      response->message = last_reason.empty() ?
        "weighted_pose_goal_all_collision_free_ik_candidates_failed" : last_reason;
      return;
    }

    const auto plan_result =
      motion_planner_->plan(group_name, q_start, q_goal, q_start_full, timeout, path_check_step, ompl_max_plan_attempts);

    if (!plan_result.success) {
      response->success = false;
      response->message = plan_result.reason;

      std::ostringstream log_oss;
      log_oss << "Motion planner failed for group [" << group_name << "]: "
              << plan_result.reason << "\n";
      log_oss << "Start state: " << q_start.transpose() << "\n";
      log_oss << "Goal state:  " << q_goal.transpose();
      RCLCPP_WARN(this->get_logger(), "%s", log_oss.str().c_str());
      return;
    }

    auto traj = trajectory_generator_->generate(group_name, plan_result.path);
    traj.header.stamp = this->now();

    response->trajectory = traj;
    response->success = true;
    response->message = "Planning succeeded.";

    if (plan_only) {
      RCLCPP_INFO(
        this->get_logger(),
        "Plan-only request for group [%s] succeeded; skipping trajectory publish/playback.",
        group_name.c_str());
      return;
    }

    trajectory_pub_->publish(traj);

    std::vector<std::string> vis_ids(
      request->obstacle_ids.begin(),
      request->obstacle_ids.begin() + static_cast<std::ptrdiff_t>(n_obs));
    std::vector<shape_msgs::msg::SolidPrimitive> vis_shapes(
      request->obstacle_shapes.begin(),
      request->obstacle_shapes.begin() + static_cast<std::ptrdiff_t>(n_obs));
    std::vector<geometry_msgs::msg::PoseStamped> vis_poses(
      request->obstacle_poses.begin(),
      request->obstacle_poses.begin() + static_cast<std::ptrdiff_t>(n_obs));

    publishVisualizationState(
      group_name,
      q_start,
      q_goal,
      traj,
      vis_ids,
      vis_shapes,
      vis_poses);
  } catch (const std::exception& e) {
    response->success = false;
    response->message = e.what();
    RCLCPP_ERROR(this->get_logger(), "Planning request failed: %s", e.what());
  }
}

}  // namespace lite_motion_planner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<lite_motion_planner::PlanningNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
