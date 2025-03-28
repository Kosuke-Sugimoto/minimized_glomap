cmake_minimum_required(VERSION 3.10)
project(minimized_glomap)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(global_positioning global_positioning.cpp)
target_link_libraries(global_positioning Ceres::ceres Eigen3::Eigen ${OpenCV_LIBS})
