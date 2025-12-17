#!/usr/bin/env bash
set -eo pipefail

# Use venv Python
export PATH="/opt/venv/bin:${PATH}"

# Use system OpenSSL for Julia
export JULIA_SSL_LIBRARY=system

# Set Cyclone DDS config if not already set
if [ -z "${CYCLONEDDS_URI:-}" ] && [ -f /home/developer/workspace/ros2/config/cyclonedds.xml ]; then
    export CYCLONEDDS_URI="file:///home/developer/workspace/ros2/config/cyclonedds.xml"
fi

# Source ROS 2
set +u
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash 2>/dev/null || true
fi

# Source workspace if built
if [ -f /home/developer/workspace/ros2/install/local_setup.bash ]; then
    source /home/developer/workspace/ros2/install/local_setup.bash 2>/dev/null || true
fi
set -u

# Execute command or start shell
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    exec bash
fi
