#include "lite_motion_planner/collision_scene.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <hpp/fcl/BVH/BVH_model.h>
#include <hpp/fcl/mesh_loader/loader.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <Eigen/Geometry>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace lite_motion_planner {
namespace {

constexpr const char* kFilePrefix = "file://";
constexpr const char* kPackagePrefix = "package://";

std::string normalizePathSeparators(const std::string& in) {
  std::string out = in;
  std::replace(out.begin(), out.end(), '\\', '/');
  return out;
}

std::string trimLeadingSlash(std::string s) {
  while (!s.empty() && s.front() == '/') {
    s.erase(s.begin());
  }
  return s;
}

bool fileExists(const std::string& p) {
  std::error_code ec;
  return std::filesystem::exists(std::filesystem::path(p), ec);
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
    // Fall through to mesh_package_dirs lookup.
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

std::string resolveMeshResourceToPath(
  const std::string& mesh_resource,
  const std::vector<std::string>& mesh_package_dirs)
{
  if (mesh_resource.empty()) {
    return std::string();
  }

  if (mesh_resource.rfind(kFilePrefix, 0) == 0) {
    return mesh_resource.substr(std::char_traits<char>::length(kFilePrefix));
  }

  if (mesh_resource.rfind(kPackagePrefix, 0) == 0) {
    return resolvePackageMeshPath(mesh_resource, mesh_package_dirs);
  }

  const auto normalized = normalizePathSeparators(mesh_resource);
  if (fileExists(normalized)) {
    return normalized;
  }

  for (const auto& base_dir_raw : mesh_package_dirs) {
    const auto candidate = (std::filesystem::path(base_dir_raw) / normalized).string();
    if (fileExists(candidate)) {
      return candidate;
    }
  }

  return std::string();
}

hpp::fcl::Transform3f se3ToFclTransform(const pinocchio::SE3& pose) {
  hpp::fcl::Transform3f tf;
  const auto t = pose.translation();
  tf.setTranslation(hpp::fcl::Vec3f(
    static_cast<float>(t.x()),
    static_cast<float>(t.y()),
    static_cast<float>(t.z())));

  const Eigen::Quaterniond q(pose.rotation());
  tf.setQuatRotation(hpp::fcl::Quaternion3f(
    static_cast<float>(q.w()),
    static_cast<float>(q.x()),
    static_cast<float>(q.y()),
    static_cast<float>(q.z())));
  return tf;
}

std::string geometryObjectLinkName(const pinocchio::Model& model, const pinocchio::GeometryObject& geom) {
  if (geom.parentFrame < model.frames.size()) {
    return model.frames[geom.parentFrame].name;
  }
  return geom.name;
}

pinocchio::SE3 poseMsgToSE3(const geometry_msgs::msg::Pose& pose) {
  const Eigen::Quaterniond q(
    pose.orientation.w,
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z);
  return pinocchio::SE3(
    q.toRotationMatrix(),
    Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z));
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

CollisionScene::CollisionScene(const RobotModel& robot_model)
: robot_model_(robot_model),
  collision_data_(collision_model_),
  last_q_full_(robot_model.neutralConfiguration())
{
  pinocchio::urdf::buildGeom(
    robot_model_.model(),
    robot_model_.urdfPath(),
    pinocchio::COLLISION,
    collision_model_,
    robot_model_.meshPackageDirs());

  // Pinocchio only loads the GeometryObjects from URDF here. Collision checks are
  // actually executed only for pairs present in GeometryModel::collisionPairs.
  // Without adding these pairs, robot-vs-robot and robot-vs-base self collision
  // can be silently skipped, which matches the observed RViz base_link
  // interference during playback. SRDF-disabled pairs are still filtered later
  // by isAllowedCollisionPair(), so adjacent/known-safe pairs remain ignored.
  collision_model_.addAllCollisionPairs();

  collision_data_ = pinocchio::GeometryData(collision_model_);
  updateRobotGeometry(last_q_full_);
}

hpp::fcl::Transform3f CollisionScene::poseMsgToFclTransform(const geometry_msgs::msg::Pose& pose) {
  hpp::fcl::Transform3f tf;
  tf.setTranslation(hpp::fcl::Vec3f(
    static_cast<float>(pose.position.x),
    static_cast<float>(pose.position.y),
    static_cast<float>(pose.position.z)));

  const auto& q = pose.orientation;
  hpp::fcl::Quaternion3f quat(
    static_cast<float>(q.w), static_cast<float>(q.x), static_cast<float>(q.y), static_cast<float>(q.z));
  tf.setQuatRotation(quat);
  return tf;
}

std::shared_ptr<hpp::fcl::CollisionGeometry> CollisionScene::primitiveToGeometry(
  const shape_msgs::msg::SolidPrimitive& primitive)
{
  switch (primitive.type) {
    case shape_msgs::msg::SolidPrimitive::BOX:
      if (primitive.dimensions.size() >= 3) {
        return std::make_shared<hpp::fcl::Box>(
          primitive.dimensions[0], primitive.dimensions[1], primitive.dimensions[2]);
      }
      break;
    case shape_msgs::msg::SolidPrimitive::SPHERE:
      if (!primitive.dimensions.empty()) {
        return std::make_shared<hpp::fcl::Sphere>(primitive.dimensions[0]);
      }
      break;
    case shape_msgs::msg::SolidPrimitive::CYLINDER:
      if (primitive.dimensions.size() >= 2) {
        return std::make_shared<hpp::fcl::Cylinder>(primitive.dimensions[1], primitive.dimensions[0]);
      }
      break;
    default:
      break;
  }
  return nullptr;
}

bool CollisionScene::isAllowedCollisionPair(const std::string& a, const std::string& b) const {
  return robot_model_.allowedCollisionMatrix().count(makeOrderedLinkPair(a, b)) > 0;
}

double CollisionScene::allowedPenetrationForPair(
  const SceneObject* scene_a,
  const SceneObject* scene_b,
  const std::string& link_a,
  const std::string& link_b) const
{
  double allowed_penetration = global_collision_allowed_penetration_;
  if (base_group_collision_allowed_penetration_ > allowed_penetration &&
      (base_group_collision_allowance_links_.count(link_a) > 0 ||
       base_group_collision_allowance_links_.count(link_b) > 0)) {
    allowed_penetration = base_group_collision_allowed_penetration_;
  }
  for (const auto& relaxed_pair : relaxed_scene_collision_pairs_) {
    if (scene_a != nullptr && scene_a->id == relaxed_pair.object_id && link_b == relaxed_pair.link_name) {
      if (relaxed_pair.allowed_penetration < 0.0) {
        return relaxed_pair.allowed_penetration;
      }
      allowed_penetration = std::max(allowed_penetration, relaxed_pair.allowed_penetration);
    }
    if (scene_b != nullptr && scene_b->id == relaxed_pair.object_id && link_a == relaxed_pair.link_name) {
      if (relaxed_pair.allowed_penetration < 0.0) {
        return relaxed_pair.allowed_penetration;
      }
      allowed_penetration = std::max(allowed_penetration, relaxed_pair.allowed_penetration);
    }
  }
  for (const auto& persistent_pair : persistent_scene_collision_pairs_) {
    if (scene_a != nullptr && scene_a->id == persistent_pair.object_id && link_b == persistent_pair.link_name) {
      if (persistent_pair.allowed_penetration < 0.0) {
        return persistent_pair.allowed_penetration;
      }
      allowed_penetration = std::max(allowed_penetration, persistent_pair.allowed_penetration);
    }
    if (scene_b != nullptr && scene_b->id == persistent_pair.object_id && link_a == persistent_pair.link_name) {
      if (persistent_pair.allowed_penetration < 0.0) {
        return persistent_pair.allowed_penetration;
      }
      allowed_penetration = std::max(allowed_penetration, persistent_pair.allowed_penetration);
    }
  }
  return allowed_penetration;
}

double CollisionScene::selfCollisionAllowedPenetrationForPair(
  const std::string& link_a,
  const std::string& link_b) const
{
  double allowed_penetration =
    std::max(self_collision_allowed_penetration_, global_collision_allowed_penetration_);
  if (robot_link_collision_allowed_penetration_ > allowed_penetration &&
      (robot_link_collision_allowance_links_.count(link_a) > 0 ||
       robot_link_collision_allowance_links_.count(link_b) > 0)) {
    allowed_penetration = robot_link_collision_allowed_penetration_;
  }
  if (base_group_collision_allowed_penetration_ > allowed_penetration &&
      (base_group_collision_allowance_links_.count(link_a) > 0 ||
       base_group_collision_allowance_links_.count(link_b) > 0)) {
    allowed_penetration = base_group_collision_allowed_penetration_;
  }
  return allowed_penetration;
}

void CollisionScene::setSceneObjectLinkCollisionAllowance(
  const std::string& object_id,
  const std::vector<std::string>& link_names,
  double allowed_penetration)
{
  // Update only the requested object-link pairs.  The haomu flow uses one
  // long-lived E3 allowance for TCP/J7/finger touch near grasp, and a
  // temporary E3-vs-cun_* allowance only for the final zc_* move.  Removing
  // every allowance for the object here would accidentally wipe the grasp
  // allowances when the final zc_* relaxation is enabled or cleared.
  relaxed_scene_collision_pairs_.erase(
    std::remove_if(
      relaxed_scene_collision_pairs_.begin(),
      relaxed_scene_collision_pairs_.end(),
      [&object_id, &link_names](const RelaxedSceneCollisionPair& pair) {
        if (pair.object_id != object_id) {
          return false;
        }
        return std::find(link_names.begin(), link_names.end(), pair.link_name) != link_names.end();
      }),
    relaxed_scene_collision_pairs_.end());

  if (allowed_penetration == 0.0) {
    return;
  }

  for (const auto& link_name : link_names) {
    if (link_name.empty()) {
      continue;
    }
    RelaxedSceneCollisionPair pair;
    pair.object_id = object_id;
    pair.link_name = link_name;
    pair.allowed_penetration = allowed_penetration;
    relaxed_scene_collision_pairs_.push_back(std::move(pair));
  }
}

void CollisionScene::setPersistentSceneObjectLinkCollisionAllowance(
  const std::string& object_id,
  const std::vector<std::string>& link_names,
  double allowed_penetration)
{
  persistent_scene_collision_pairs_.erase(
    std::remove_if(
      persistent_scene_collision_pairs_.begin(),
      persistent_scene_collision_pairs_.end(),
      [&object_id, &link_names](const RelaxedSceneCollisionPair& pair) {
        if (pair.object_id != object_id) {
          return false;
        }
        return std::find(link_names.begin(), link_names.end(), pair.link_name) != link_names.end();
      }),
    persistent_scene_collision_pairs_.end());

  if (allowed_penetration == 0.0) {
    return;
  }

  for (const auto& link_name : link_names) {
    if (link_name.empty()) {
      continue;
    }
    RelaxedSceneCollisionPair pair;
    pair.object_id = object_id;
    pair.link_name = link_name;
    pair.allowed_penetration = allowed_penetration;
    persistent_scene_collision_pairs_.push_back(std::move(pair));
  }
}

void CollisionScene::setSelfCollisionAllowedPenetration(double allowed_penetration)
{
  self_collision_allowed_penetration_ = std::max(0.0, allowed_penetration);
}

void CollisionScene::setGlobalCollisionAllowedPenetration(double allowed_penetration)
{
  global_collision_allowed_penetration_ = std::max(0.0, allowed_penetration);
}

void CollisionScene::setRobotLinkCollisionAllowance(
  const std::vector<std::string>& link_names,
  double allowed_penetration)
{
  robot_link_collision_allowance_links_.clear();
  robot_link_collision_allowed_penetration_ = std::max(0.0, allowed_penetration);
  if (robot_link_collision_allowed_penetration_ == 0.0) {
    return;
  }

  for (const auto& link_name : link_names) {
    if (!link_name.empty()) {
      robot_link_collision_allowance_links_.insert(link_name);
    }
  }
}

void CollisionScene::setBaseGroupCollisionAllowance(
  const std::vector<std::string>& link_names,
  double allowed_penetration)
{
  base_group_collision_allowance_links_.clear();
  base_group_collision_allowed_penetration_ = std::max(0.0, allowed_penetration);
  if (base_group_collision_allowed_penetration_ == 0.0) {
    return;
  }

  for (const auto& link_name : link_names) {
    if (!link_name.empty()) {
      base_group_collision_allowance_links_.insert(link_name);
    }
  }
}

void CollisionScene::setGlobalIgnoredRobotCollisionLinks(
  const std::vector<std::string>& link_names,
  bool ignore)
{
  for (const auto& link_name : link_names) {
    if (link_name.empty()) {
      continue;
    }
    if (ignore) {
      globally_ignored_robot_collision_links_.insert(link_name);
    } else {
      globally_ignored_robot_collision_links_.erase(link_name);
    }
  }
}

void CollisionScene::updateRobotGeometry(const Eigen::VectorXd& q_full) const {
  last_q_full_ = q_full;
  pinocchio::forwardKinematics(
    robot_model_.model(),
    const_cast<RobotModel&>(robot_model_).data(),
    q_full);
  pinocchio::updateFramePlacements(
    robot_model_.model(),
    const_cast<RobotModel&>(robot_model_).data());
  pinocchio::updateGeometryPlacements(
    robot_model_.model(),
    const_cast<RobotModel&>(robot_model_).data(),
    collision_model_,
    const_cast<pinocchio::GeometryData&>(collision_data_),
    q_full);
}

void CollisionScene::refreshGeometryData() {
  collision_data_ = pinocchio::GeometryData(collision_model_);
  if (last_q_full_.size() == robot_model_.neutralConfiguration().size()) {
    updateRobotGeometry(last_q_full_);
  } else {
    last_q_full_ = robot_model_.neutralConfiguration();
    updateRobotGeometry(last_q_full_);
  }
}

void CollisionScene::activateSceneObject(SceneObject& object) {
  if (!object.geometry) {
    throw std::runtime_error("scene object has no collision geometry: " + object.id);
  }

  if (collision_model_.existGeometryName(object.id)) {
    collision_model_.removeGeometryObject(object.id);
    object.active_in_collision_model = false;
    refreshGeometryData();
  }

  const Eigen::Vector3d mesh_scale(
    object.mesh_scale[0], object.mesh_scale[1], object.mesh_scale[2]);

  pinocchio::GeometryObject geom_object(
    object.id,
    object.parent_joint,
    object.parent_frame,
    object.geometry_placement,
    object.geometry,
    object.mesh_resource,
    mesh_scale);

  const auto geom_index = collision_model_.addGeometryObject(geom_object, robot_model_.model());

  for (pinocchio::GeomIndex other_index = 0;
       other_index < static_cast<pinocchio::GeomIndex>(collision_model_.geometryObjects.size());
       ++other_index) {
    if (other_index == geom_index) {
      continue;
    }

    const auto& other_geom = collision_model_.geometryObjects[other_index];
    const auto other_scene = findSceneObjectByGeometryName(other_geom.name);
    if (other_scene != nullptr) {
      continue;
    }

    const auto other_link = geometryObjectLinkName(robot_model_.model(), other_geom);
    if (object.touch_links.count(other_link) > 0) {
      continue;
    }

    collision_model_.addCollisionPair(pinocchio::CollisionPair(geom_index, other_index));
  }

  object.active_in_collision_model = true;
  refreshGeometryData();
}

void CollisionScene::deactivateSceneObject(SceneObject& object) {
  if (!object.active_in_collision_model) {
    return;
  }

  if (collision_model_.existGeometryName(object.id)) {
    collision_model_.removeGeometryObject(object.id);
  }

  object.active_in_collision_model = false;
  refreshGeometryData();
}

void CollisionScene::updateGeometry(const Eigen::VectorXd& q_full) const {
  updateRobotGeometry(q_full);
}

geometry_msgs::msg::Pose CollisionScene::computeAttachedWorldPose(const SceneObject& object) const {
  const auto frame_id = robot_model_.getFrameIdChecked(object.frame_id);
  const auto& frame_pose = robot_model_.data().oMf[frame_id];
  return se3ToPoseMsg(frame_pose * poseMsgToSE3(object.relative_pose));
}

const CollisionScene::SceneObject* CollisionScene::findSceneObjectByGeometryName(
  const std::string& geometry_name) const {
  auto it = std::find_if(
    scene_objects_.begin(),
    scene_objects_.end(),
    [&geometry_name](const SceneObject& obj) {
      return obj.id == geometry_name && obj.active_in_collision_model;
    });
  return it == scene_objects_.end() ? nullptr : &(*it);
}

bool CollisionScene::checkSelfCollision(const Eigen::VectorXd& q_full) const {
  const auto report = getCollisionReport(q_full);
  return report.in_collision && report.category == "self";
}

bool CollisionScene::checkEnvironmentCollision(const Eigen::VectorXd& q_full) const {
  const auto report = getCollisionReport(q_full);
  return report.in_collision && report.category == "environment";
}

bool CollisionScene::isStateValid(const Eigen::VectorXd& q_full) const {
  return !getCollisionReport(q_full).in_collision;
}

CollisionScene::CollisionReport CollisionScene::getCollisionReport(const Eigen::VectorXd& q_full) const {
  updateRobotGeometry(q_full);
  return getCollisionReportWithData(
    q_full,
    const_cast<RobotModel&>(robot_model_).data(),
    const_cast<pinocchio::GeometryData&>(collision_data_));
}

CollisionScene::CollisionReport CollisionScene::getCollisionReportThreadSafe(const Eigen::VectorXd& q_full) const {
  pinocchio::Data model_data(robot_model_.model());
  pinocchio::GeometryData geometry_data(collision_model_);
  return getCollisionReportWithData(q_full, model_data, geometry_data);
}

CollisionScene::CollisionReport CollisionScene::getCollisionReportWithData(
  const Eigen::VectorXd& q_full,
  pinocchio::Data& model_data,
  pinocchio::GeometryData& geometry_data) const {
  pinocchio::forwardKinematics(robot_model_.model(), model_data, q_full);
  pinocchio::updateFramePlacements(robot_model_.model(), model_data);
  pinocchio::updateGeometryPlacements(
    robot_model_.model(),
    model_data,
    collision_model_,
    geometry_data,
    q_full);

  for (size_t i = 0; i < collision_model_.collisionPairs.size(); ++i) {
    const auto& pair = collision_model_.collisionPairs[i];
    const auto& geom_a = collision_model_.geometryObjects[pair.first];
    const auto& geom_b = collision_model_.geometryObjects[pair.second];

    const auto* scene_a = findSceneObjectByGeometryName(geom_a.name);
    const auto* scene_b = findSceneObjectByGeometryName(geom_b.name);

    if (scene_a != nullptr && scene_b != nullptr) {
      continue;
    }

    const auto link_a = geometryObjectLinkName(robot_model_.model(), geom_a);
    const auto link_b = geometryObjectLinkName(robot_model_.model(), geom_b);

    if (globally_ignored_robot_collision_links_.count(link_a) > 0 ||
        globally_ignored_robot_collision_links_.count(link_b) > 0) {
      continue;
    }

    if (scene_a == nullptr && scene_b == nullptr && isAllowedCollisionPair(link_a, link_b)) {
      continue;
    }

    hpp::fcl::CollisionObject obj_a(geom_a.geometry, se3ToFclTransform(geometry_data.oMg[pair.first]), false);
    hpp::fcl::CollisionObject obj_b(geom_b.geometry, se3ToFclTransform(geometry_data.oMg[pair.second]), false);

    hpp::fcl::CollisionRequest request;
    request.enable_contact = true;
    request.num_max_contacts = 1;
    const bool robot_self_pair = scene_a == nullptr && scene_b == nullptr;
    const double allowed_penetration = robot_self_pair ?
      selfCollisionAllowedPenetrationForPair(link_a, link_b) :
      allowedPenetrationForPair(scene_a, scene_b, link_a, link_b);
    if (allowed_penetration < 0.0) {
      continue;
    }
    if (allowed_penetration > 0.0) {
      request.security_margin = -allowed_penetration;
    }

    hpp::fcl::CollisionResult result;
    hpp::fcl::collide(&obj_a, &obj_b, request, result);

    if (!result.isCollision()) {
      continue;
    }

    CollisionReport report;
    report.in_collision = true;
    report.object_a = scene_a != nullptr ? scene_a->id : link_a;
    report.object_b = scene_b != nullptr ? scene_b->id : link_b;

    if (scene_a == nullptr && scene_b == nullptr) {
      report.category = "self";
      report.detail = "self collision pair";
      if (allowed_penetration > 0.0) {
        report.detail += " (penetration allowance=" + std::to_string(allowed_penetration) + " m)";
      }
    } else {
      report.category = "environment";
      const SceneObject* scene_obj = scene_a != nullptr ? scene_a : scene_b;
      if (allowed_penetration > 0.0) {
        report.detail = (scene_obj->attached ? "attached geometry collision" : "scene geometry collision") +
          (" (penetration allowance=" + std::to_string(allowed_penetration) + " m)");
      } else {
        report.detail = scene_obj->attached ? "attached geometry collision" : "scene geometry collision";
      }
    }
    return report;
  }

  hpp::fcl::CollisionRequest request;
  if (global_collision_allowed_penetration_ > 0.0) {
    request.security_margin = -global_collision_allowed_penetration_;
  }
  for (size_t gi = 0; gi < collision_model_.geometryObjects.size(); ++gi) {
    const auto& geom = collision_model_.geometryObjects[gi];
    if (findSceneObjectByGeometryName(geom.name) != nullptr) {
      continue;
    }

    const auto robot_link = geometryObjectLinkName(robot_model_.model(), geom);
    if (globally_ignored_robot_collision_links_.count(robot_link) > 0) {
      continue;
    }
    hpp::fcl::CollisionObject robot_obj(geom.geometry, se3ToFclTransform(geometry_data.oMg[gi]), false);

    for (const auto& obs : transient_obstacles_) {
      hpp::fcl::CollisionResult result;
      hpp::fcl::collide(&robot_obj, obs.object.get(), request, result);
      if (result.isCollision()) {
        CollisionReport report;
        report.in_collision = true;
        report.category = "environment";
        report.object_a = robot_link;
        report.object_b = obs.id;
        report.detail = "transient obstacle collision";
        if (global_collision_allowed_penetration_ > 0.0) {
          report.detail += " (penetration allowance=" +
            std::to_string(global_collision_allowed_penetration_) + " m)";
        }
        return report;
      }
    }
  }

  return CollisionReport{};
}

void CollisionScene::clearTransientObstacles() {
  transient_obstacles_.clear();
}

void CollisionScene::addTransientObstacleFromPrimitive(
  const std::string& id,
  const shape_msgs::msg::SolidPrimitive& primitive,
  const geometry_msgs::msg::Pose& pose)
{
  auto geometry = primitiveToGeometry(primitive);
  if (!geometry) {
    return;
  }
  Obstacle obs;
  obs.id = id;
  obs.geometry = geometry;
  obs.object = std::make_shared<hpp::fcl::CollisionObject>(obs.geometry, poseMsgToFclTransform(pose), false);
  transient_obstacles_.push_back(std::move(obs));
}

bool CollisionScene::upsertSceneObject(
  const std::string& id,
  const std::string& mesh_resource,
  const std::vector<double>& mesh_scale,
  const shape_msgs::msg::SolidPrimitive& primitive,
  const geometry_msgs::msg::Pose& pose)
{
  std::shared_ptr<hpp::fcl::CollisionGeometry> geometry;
  bool uses_mesh = false;
  std::array<double, 3> scale{{1.0, 1.0, 1.0}};

  if (!mesh_resource.empty()) {
    const std::string mesh_path = resolveMeshResourceToPath(mesh_resource, robot_model_.meshPackageDirs());
    if (mesh_path.empty() || !fileExists(mesh_path)) {
      throw std::runtime_error("failed to resolve mesh resource to existing file: " + mesh_resource);
    }

    if (!mesh_scale.empty()) {
      if (mesh_scale.size() != 3) {
        throw std::runtime_error("mesh_scale must have 3 values when provided");
      }
      scale = {mesh_scale[0], mesh_scale[1], mesh_scale[2]};
    }

    hpp::fcl::MeshLoader loader;
    auto mesh_geom = loader.load(
      mesh_path,
      hpp::fcl::Vec3f(
        static_cast<hpp::fcl::FCL_REAL>(scale[0]),
        static_cast<hpp::fcl::FCL_REAL>(scale[1]),
        static_cast<hpp::fcl::FCL_REAL>(scale[2])));

    if (!mesh_geom) {
      throw std::runtime_error("failed to load mesh resource: " + mesh_resource);
    }

    geometry = std::static_pointer_cast<hpp::fcl::CollisionGeometry>(mesh_geom);
    uses_mesh = true;
  } else {
    geometry = primitiveToGeometry(primitive);
    if (!geometry) {
      return false;
    }
  }

  auto it = std::find_if(scene_objects_.begin(), scene_objects_.end(), [&id](const SceneObject& obj) {
    return obj.id == id;
  });

  if (it == scene_objects_.end()) {
    SceneObject obj;
    obj.id = id;
    obj.relative_pose.orientation.w = 1.0;
    scene_objects_.push_back(std::move(obj));
    it = std::prev(scene_objects_.end());
  }

  deactivateSceneObject(*it);

  it->uses_mesh = uses_mesh;
  it->mesh_resource = mesh_resource;
  it->mesh_scale = scale;
  it->primitive = primitive;
  it->world_pose = pose;
  it->attached = false;
  it->frame_id.clear();
  it->relative_pose = geometry_msgs::msg::Pose();
  it->relative_pose.orientation.w = 1.0;
  it->touch_links.clear();
  it->geometry = geometry;
  if (robot_model_.model().existFrame("base_link")) {
    const auto base_frame_idx = robot_model_.getFrameIdChecked("base_link");
    const auto& base_frame = robot_model_.model().frames[base_frame_idx];
    it->parent_joint = base_frame.parentJoint;
    it->parent_frame = base_frame_idx;
    it->geometry_placement = base_frame.placement * poseMsgToSE3(pose);
  } else {
    it->parent_joint = 0;
    it->parent_frame = 0;
    it->geometry_placement = poseMsgToSE3(pose);
  }

  activateSceneObject(*it);
  return true;
}

bool CollisionScene::removeSceneObject(const std::string& id) {
  auto it = std::find_if(scene_objects_.begin(), scene_objects_.end(), [&id](const SceneObject& obj) {
    return obj.id == id;
  });
  if (it == scene_objects_.end()) {
    return false;
  }

  deactivateSceneObject(*it);
  scene_objects_.erase(it);
  return true;
}

bool CollisionScene::setObjectAttached(
  const std::string& id,
  bool attach,
  const std::string& frame_id,
  const geometry_msgs::msg::Pose& relative_pose,
  const std::vector<std::string>& touch_links)
{
  auto it = std::find_if(scene_objects_.begin(), scene_objects_.end(), [&id](const SceneObject& obj) {
    return obj.id == id;
  });
  if (it == scene_objects_.end()) {
    return false;
  }

  const Eigen::VectorXd q_ref =
    last_q_full_.size() == robot_model_.neutralConfiguration().size() ?
    last_q_full_ : robot_model_.neutralConfiguration();
  updateRobotGeometry(q_ref);

  geometry_msgs::msg::Pose detached_world_pose = it->attached ? computeAttachedWorldPose(*it) : it->world_pose;
  deactivateSceneObject(*it);

  if (attach) {
    const auto frame_idx = robot_model_.getFrameIdChecked(frame_id);
    const auto& frame = robot_model_.model().frames[frame_idx];

    it->attached = true;
    it->frame_id = frame_id;
    it->relative_pose = relative_pose;
    it->touch_links.clear();
    it->touch_links.insert(touch_links.begin(), touch_links.end());
    it->parent_joint = frame.parentJoint;
    it->parent_frame = frame_idx;
    it->geometry_placement = frame.placement * poseMsgToSE3(relative_pose);
  } else {
    it->attached = false;
    it->frame_id.clear();
    it->relative_pose = geometry_msgs::msg::Pose();
    it->relative_pose.orientation.w = 1.0;
    it->touch_links.clear();
    it->world_pose = detached_world_pose;
    if (robot_model_.model().existFrame("base_link")) {
      const auto base_frame_idx = robot_model_.getFrameIdChecked("base_link");
      const auto& base_frame = robot_model_.model().frames[base_frame_idx];
      it->parent_joint = base_frame.parentJoint;
      it->parent_frame = base_frame_idx;
      it->geometry_placement = base_frame.placement * poseMsgToSE3(detached_world_pose);
    } else {
      it->parent_joint = 0;
      it->parent_frame = 0;
      it->geometry_placement = poseMsgToSE3(detached_world_pose);
    }
  }

  activateSceneObject(*it);
  return true;
}

std::vector<CollisionScene::SceneObjectInfo> CollisionScene::sceneObjectInfos() const {
  std::vector<SceneObjectInfo> out;
  out.reserve(scene_objects_.size());
  for (const auto& obj : scene_objects_) {
    SceneObjectInfo info;
    info.id = obj.id;
    info.uses_mesh = obj.uses_mesh;
    info.mesh_resource = obj.mesh_resource;
    info.mesh_scale = obj.mesh_scale;
    info.primitive = obj.primitive;
    info.world_pose = obj.attached ? computeAttachedWorldPose(obj) : obj.world_pose;
    info.attached = obj.attached;
    info.frame_id = obj.frame_id;
    info.relative_pose = obj.relative_pose;
    info.touch_links.assign(obj.touch_links.begin(), obj.touch_links.end());
    out.push_back(std::move(info));
  }
  return out;
}

}  // namespace lite_motion_planner
