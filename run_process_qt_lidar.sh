#!/bin/bash
source /opt/ros/noetic/setup.bash
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/devel/setup.bash"
roslaunch  lslidar_ls_driver  qt_imu_gps_lslidar_process.launch
