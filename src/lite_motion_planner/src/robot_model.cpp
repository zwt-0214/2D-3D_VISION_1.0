#include "lite_motion_planner/robot_model.hpp"

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <tinyxml2.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace lite_motion_planner {
namespace {

JointLimit parseJointLimit(const tinyxml2::XMLElement* joint_el) {
  JointLimit limit;
  const auto* limit_el = joint_el->FirstChildElement("limit");
  if (!limit_el) {
    return limit;
  }

  limit.lower = limit_el->DoubleAttribute("lower", 0.0);
  limit.upper = limit_el->DoubleAttribute("upper", 0.0);
  limit.velocity = limit_el->DoubleAttribute("velocity", 0.0);

  // 这里的加速度/jerk 先给出保守默认值；后续可扩展为从 SRDF 自定义标签读取。
  limit.acceleration = (limit.velocity > 0.0) ? (2.0 * limit.velocity) : 1.0;
  limit.jerk = (limit.acceleration > 0.0) ? (5.0 * limit.acceleration) : 5.0;
  return limit;
}

}  // namespace

RobotModel::RobotModel(
  const std::string& urdf_path,
  const std::string& srdf_path,
  const std::vector<std::string>& mesh_package_dirs)
: urdf_path_(urdf_path),
  srdf_path_(srdf_path),
  mesh_package_dirs_(mesh_package_dirs)
{
  loadPinocchioModel();
  parseUrdfStructure();
  parseSrdf();
  buildJointIndexMaps();
}

void RobotModel::loadPinocchioModel() {
  pinocchio::urdf::buildModel(urdf_path_, model_);
  data_ = pinocchio::Data(model_);
}

void RobotModel::parseUrdfStructure() {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(urdf_path_.c_str()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("Failed to load URDF: " + urdf_path_);
  }

  auto* robot_el = doc.FirstChildElement("robot");
  if (!robot_el) {
    throw std::runtime_error("URDF missing <robot> root element.");
  }

  for (auto* joint_el = robot_el->FirstChildElement("joint"); joint_el;
       joint_el = joint_el->NextSiblingElement("joint")) {
    UrdfJointInfo info;
    info.name = joint_el->Attribute("name") ? joint_el->Attribute("name") : "";
    info.type = joint_el->Attribute("type") ? joint_el->Attribute("type") : "";

    const auto* parent_el = joint_el->FirstChildElement("parent");
    const auto* child_el = joint_el->FirstChildElement("child");
    if (!parent_el || !child_el) {
      continue;
    }

    info.parent_link = parent_el->Attribute("link") ? parent_el->Attribute("link") : "";
    info.child_link = child_el->Attribute("link") ? child_el->Attribute("link") : "";

    urdf_joints_by_name_[info.name] = info;
    child_link_to_joint_[info.child_link] = info.name;
    joint_limits_[info.name] = parseJointLimit(joint_el);
  }
}

void RobotModel::parseSrdf() {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(srdf_path_.c_str()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("Failed to load SRDF: " + srdf_path_);
  }

  auto* robot_el = doc.FirstChildElement("robot");
  if (!robot_el) {
    throw std::runtime_error("SRDF missing <robot> root element.");
  }

  parseGroups(robot_el);
  parseAllowedCollisions(robot_el);
}

void RobotModel::parseGroups(tinyxml2::XMLElement* robot_el) {
  for (auto* group_el = robot_el->FirstChildElement("group"); group_el;
       group_el = group_el->NextSiblingElement("group")) {
    const char* group_name_cstr = group_el->Attribute("name");
    if (!group_name_cstr) {
      continue;
    }
    const std::string group_name = group_name_cstr;

    if (auto* chain_el = group_el->FirstChildElement("chain")) {
      const auto* base_link = chain_el->Attribute("base_link");
      const auto* tip_link = chain_el->Attribute("tip_link");
      if (!base_link || !tip_link) {
        throw std::runtime_error("Chain group missing base_link or tip_link: " + group_name);
      }
      planning_groups_[group_name] = resolveChainGroup(group_name, base_link, tip_link);
      continue;
    }

    PlanningGroupInfo info;
    info.name = group_name;
    for (auto* joint_el = group_el->FirstChildElement("joint"); joint_el;
         joint_el = joint_el->NextSiblingElement("joint")) {
      const auto* joint_name = joint_el->Attribute("name");
      if (!joint_name) {
        continue;
      }
      info.joint_names.emplace_back(joint_name);
    }
    planning_groups_[group_name] = info;
  }
}

void RobotModel::parseAllowedCollisions(tinyxml2::XMLElement* robot_el) {
  for (auto* dc_el = robot_el->FirstChildElement("disable_collisions"); dc_el;
       dc_el = dc_el->NextSiblingElement("disable_collisions")) {
    const auto* link1 = dc_el->Attribute("link1");
    const auto* link2 = dc_el->Attribute("link2");
    if (!link1 || !link2) {
      continue;
    }

    std::string reason = "adjacent";
    if (const auto* reason_c = dc_el->Attribute("reason")) {
      reason = reason_c;
      std::transform(reason.begin(), reason.end(), reason.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }
    if (reason != "adjacent" && reason != "default" && reason != "never" && reason != "user") {
      continue;
    }

    allowed_collision_matrix_.insert(makeOrderedLinkPair(link1, link2));
  }
}

PlanningGroupInfo RobotModel::resolveChainGroup(
  const std::string& group_name,
  const std::string& base_link,
  const std::string& tip_link) const
{
  PlanningGroupInfo info;
  info.name = group_name;
  info.base_link = base_link;
  info.tip_link = tip_link;

  std::string current_link = tip_link;
  while (current_link != base_link) {
    const auto it = child_link_to_joint_.find(current_link);
    if (it == child_link_to_joint_.end()) {
      throw std::runtime_error(
        "Cannot resolve chain group '" + group_name + "' from tip '" + tip_link + "' to base '" + base_link + "'.");
    }

    const auto& joint_name = it->second;
    const auto& joint_info = urdf_joints_by_name_.at(joint_name);
    info.joint_names.push_back(joint_name);
    info.link_names.push_back(current_link);
    current_link = joint_info.parent_link;
  }

  std::reverse(info.joint_names.begin(), info.joint_names.end());
  std::reverse(info.link_names.begin(), info.link_names.end());
  info.link_names.insert(info.link_names.begin(), base_link);
  return info;
}

void RobotModel::buildJointIndexMaps() {
  for (pinocchio::JointIndex jid = 1; jid < static_cast<pinocchio::JointIndex>(model_.njoints); ++jid) {
    const std::string joint_name = model_.names[jid];
    joint_to_qidx_[joint_name] = model_.joints[jid].idx_q();
    joint_to_vidx_[joint_name] = model_.joints[jid].idx_v();
  }
}

const PlanningGroupInfo& RobotModel::getPlanningGroup(const std::string& group_name) const {
  const auto it = planning_groups_.find(group_name);
  if (it == planning_groups_.end()) {
    throw std::runtime_error("Unknown planning group: " + group_name);
  }
  return it->second;
}

bool RobotModel::hasPlanningGroup(const std::string& group_name) const {
  return planning_groups_.count(group_name) > 0;
}

Eigen::VectorXd RobotModel::neutralConfiguration() const {
  return pinocchio::neutral(model_);
}

Eigen::VectorXd RobotModel::lowerBoundsForGroup(const std::string& group_name) const {
  const auto& group = getPlanningGroup(group_name);
  Eigen::VectorXd out(group.joint_names.size());
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    out[static_cast<Eigen::Index>(i)] = joint_limits_.at(group.joint_names[i]).lower;
  }
  return out;
}

Eigen::VectorXd RobotModel::upperBoundsForGroup(const std::string& group_name) const {
  const auto& group = getPlanningGroup(group_name);
  Eigen::VectorXd out(group.joint_names.size());
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    out[static_cast<Eigen::Index>(i)] = joint_limits_.at(group.joint_names[i]).upper;
  }
  return out;
}

Eigen::VectorXd RobotModel::velocityLimitsForGroup(const std::string& group_name) const {
  const auto& group = getPlanningGroup(group_name);
  Eigen::VectorXd out(group.joint_names.size());
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    out[static_cast<Eigen::Index>(i)] = joint_limits_.at(group.joint_names[i]).velocity;
  }
  return out;
}

Eigen::VectorXd RobotModel::accelerationLimitsForGroup(const std::string& group_name) const {
  const auto& group = getPlanningGroup(group_name);
  Eigen::VectorXd out(group.joint_names.size());
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    out[static_cast<Eigen::Index>(i)] = joint_limits_.at(group.joint_names[i]).acceleration;
  }
  return out;
}

Eigen::VectorXd RobotModel::jerkLimitsForGroup(const std::string& group_name) const {
  const auto& group = getPlanningGroup(group_name);
  Eigen::VectorXd out(group.joint_names.size());
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    out[static_cast<Eigen::Index>(i)] = joint_limits_.at(group.joint_names[i]).jerk;
  }
  return out;
}

Eigen::VectorXd RobotModel::fullToGroup(const std::string& group_name, const Eigen::VectorXd& q_full) const {
  const auto& group = getPlanningGroup(group_name);
  Eigen::VectorXd q_group(group.joint_names.size());
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    q_group[static_cast<Eigen::Index>(i)] = q_full[qIndexOfJoint(group.joint_names[i])];
  }
  return q_group;
}

Eigen::VectorXd RobotModel::groupToFull(
  const std::string& group_name,
  const Eigen::VectorXd& q_group,
  const Eigen::VectorXd& seed_full) const
{
  const auto& group = getPlanningGroup(group_name);
  if (q_group.size() != static_cast<Eigen::Index>(group.joint_names.size())) {
    throw std::runtime_error("groupToFull: joint vector size mismatch for group " + group_name);
  }

  Eigen::VectorXd q_full = seed_full;
  for (size_t i = 0; i < group.joint_names.size(); ++i) {
    q_full[qIndexOfJoint(group.joint_names[i])] = q_group[static_cast<Eigen::Index>(i)];
  }
  return q_full;
}

int RobotModel::qIndexOfJoint(const std::string& joint_name) const {
  const auto it = joint_to_qidx_.find(joint_name);
  if (it == joint_to_qidx_.end()) {
    throw std::runtime_error("Joint not found in Pinocchio model: " + joint_name);
  }
  return it->second;
}

int RobotModel::vIndexOfJoint(const std::string& joint_name) const {
  const auto it = joint_to_vidx_.find(joint_name);
  if (it == joint_to_vidx_.end()) {
    throw std::runtime_error("Joint velocity index not found in Pinocchio model: " + joint_name);
  }
  return it->second;
}

pinocchio::FrameIndex RobotModel::getFrameIdChecked(const std::string& frame_name) const {
  if (!model_.existFrame(frame_name)) {
    throw std::runtime_error("Frame not found: " + frame_name);
  }
  return model_.getFrameId(frame_name);
}

std::vector<std::string> RobotModel::activeJointNames() const {
  std::vector<std::string> names;
  names.reserve(static_cast<size_t>(model_.njoints > 0 ? model_.njoints - 1 : 0));
  for (pinocchio::JointIndex jid = 1; jid < static_cast<pinocchio::JointIndex>(model_.njoints); ++jid) {
    if (model_.joints[jid].nq() > 0) {
      names.push_back(model_.names[jid]);
    }
  }
  return names;
}

std::string RobotModel::robotName() const {
  return model_.name.empty() ? std::string("robot") : model_.name;
}

}  // namespace lite_motion_planner
