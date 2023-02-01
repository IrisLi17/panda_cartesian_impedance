cmake_minimum_required(VERSION 3.4)
project(panda_cartesian_impedance)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  controller_interface
  dynamic_reconfigure
  eigen_conversions
  franka_hw
  franka_gripper
  geometry_msgs
  hardware_interface
  tf
  tf_conversions
  message_generation
  pluginlib
  realtime_tools
  roscpp
  rospy
)

find_package(Eigen3 REQUIRED)
find_package(Franka 0.7.0 REQUIRED)
find_package(Torch REQUIRED)


add_message_files(FILES
  DualPoseStamped.msg
)

generate_messages(
  DEPENDENCIES 
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES panda_cartesian_impedance
  CATKIN_DEPENDS
    controller_interface
    dynamic_reconfigure
    eigen_conversions
    franka_hw
    franka_gripper
    geometry_msgs
    hardware_interface
    tf
    tf_conversions
    message_runtime
    pluginlib
    realtime_tools
    roscpp
  DEPENDS Franka
)

add_library(panda_cartesian_impedance
  src/dual_arm_cartesian_impedance_controller.cpp
  # src/neural_commander.cpp
  src/cartesian_ik.cpp
  src/cartesian_pd.cpp
  src/cartesian_osc.cpp
)

target_link_libraries(panda_cartesian_impedance PUBLIC
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)

target_include_directories(panda_cartesian_impedance SYSTEM PUBLIC
  ${Franka_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_include_directories(panda_cartesian_impedance PUBLIC
  include
)

add_executable(inference_model src/inference_model.cpp)
target_link_libraries(inference_model PUBLIC
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)

target_include_directories(inference_model SYSTEM PUBLIC
  ${Franka_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

target_include_directories(inference_model PUBLIC
  include
)

## Installation
install(TARGETS panda_cartesian_impedance
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES panda_cartesian_impedance_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
catkin_install_python(
  PROGRAMS scripts/pose_commander.py scripts/obs_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(panda_cartesian_impedance FILES ${SOURCES} ${HEADERS})
  add_tidy_target(panda_cartesian_impedance
    FILES ${SOURCES}
    DEPENDS panda_cartesian_impedance
  )
endif()
