# Deleted:cmake_minimum_required(VERSION 2.8.3)
cmake_minimum_required(VERSION 3.8)

project(engineer_0520_description)

# Deleted:find_package(catkin REQUIRED)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

# Deleted:catkin_package()

# Deleted:find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION share/${PROJECT_NAME}/${dir})
endforeach(dir)

ament_package()