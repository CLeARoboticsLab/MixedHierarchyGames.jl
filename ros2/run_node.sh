colcon build
source install/setup.bash

# Use venv Python explicitly to ensure we have all required packages (rclpy, yaml, etc.)
if [ -f /opt/venv/bin/python3 ]; then
    /opt/venv/bin/python3 src/mixed_hierarchy_games/mixed_hierarchy_games/controller_node.py
else
    python3 src/mixed_hierarchy_games/mixed_hierarchy_games/controller_node.py
fi