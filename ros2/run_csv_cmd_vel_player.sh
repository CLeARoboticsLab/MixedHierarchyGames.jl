#!/usr/bin/env bash

set -e -o pipefail

colcon build
# Disable nounset while sourcing ROS 2 setup to avoid unbound env vars
set +u
source install/setup.bash
set -u

# Optional: headless plotting/backends used elsewhere
export GKSwstype=nul
export JULIA_SSL_LIBRARY=system

ros2 run multi_robot_controller csv_cmd_vel_player


