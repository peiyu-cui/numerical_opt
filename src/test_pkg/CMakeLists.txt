cmake_minimum_required(VERSION 3.0.2)
project(test_pkg)

set (CMAKE_BUILD_TYPE "Release")
# set (CMAKE_BUILD_TYPE "Debug")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  opt_solver
)

find_package(Eigen3 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS roscpp rospy std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)


add_executable(test_slover_node src/solver_example.cpp)

target_link_libraries(test_slover_node
  ${catkin_LIBRARIES}
)

