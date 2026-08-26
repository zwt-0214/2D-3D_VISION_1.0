cmake_minimum_required(VERSION 3.10)
project(rm_serial_msgs)

add_compile_options(-O3 -Wall -Wpedantic -Werror)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Receive.msg"
  "msg/Target.msg"
  "msg/RobotMode.msg"

  DEPENDENCIES
    geometry_msgs
)

ament_package()