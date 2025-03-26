cmake_minimum_required(VERSION 3.0.2)
project(opt_solver)

set (CMAKE_BUILD_TYPE "Release")
# set (CMAKE_BUILD_TYPE "Debug")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)
find_package(Eigen3 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES opt_solver
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/gradient_descent.cpp
  src/bfgs.cpp
  src/cautious_bfgs.cpp
  src/lbfgs.cpp
  src/newton.cpp
  src/newton_cg.cpp
)

