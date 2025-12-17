source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash

# Use venv Python explicitly to ensure we have all required packages (rclpy, yaml, etc.)
if [ -f /opt/venv/bin/python3 ]; then
    /opt/venv/bin/python3 src/multi_robot_controller/multi_robot_controller/controller_node.py
else
    python3 src/multi_robot_controller/multi_robot_controller/controller_node.py
fi