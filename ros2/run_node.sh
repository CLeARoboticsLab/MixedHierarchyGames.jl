# export PYJULIA_JULIA="/opt/julia-1.10.5/bin/julia"

source /opt/ros/foxy/setup.bash

# cd ~/Tianyu/Robot Experiments/MCP_Isaac_sim_ros2/ros2
colcon build
source install/local_setup.bash
# ros2 run multi_robot_controller controller_node
python3 src/multi_robot_controller/multi_robot_controller/controller_node.py
