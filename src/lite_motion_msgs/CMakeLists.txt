cmake_minimum_required(VERSION 3.16)
project(lite_motion_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PlanArmMotion.srv"
  "srv/AddSceneObject.srv"
  "srv/RemoveSceneObject.srv"
  "srv/AttachSceneObject.srv"
  "srv/CheckStateCollision.srv"
  "srv/SetSceneCollisionAllowance.srv"
  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
    shape_msgs
    trajectory_msgs
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()