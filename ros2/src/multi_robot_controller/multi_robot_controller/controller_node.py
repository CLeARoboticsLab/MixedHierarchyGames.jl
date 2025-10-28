import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math

import time
from pathlib import Path
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

# Only do this once — import and include your Julia code
def julia_init():
    # Go up to the main project root: ros2/src/multi_robot_controller/multi_robot_controller -> main project
    project_root = str(Path(__file__).resolve().parents[4])
    time_start = time.perf_counter()
    Main.eval(
        f"""
        import Pkg
        Pkg.activate(raw"{project_root}")
        try
            Pkg.instantiate()
        catch e
            @warn "Pkg.instantiate() failed" exception=e
        end

        using Logging
        global_logger(NullLogger())

    Base.include(Main, joinpath(raw"{project_root}", "examples", "automatic_solver.jl"))
    Base.include(Main, joinpath(raw"{project_root}", "examples", "test_automatic_solver.jl"))
    Base.include(Main, joinpath(raw"{project_root}", "examples", "hardware_functions.jl"))
        """
    )

    # Build preoptimization once
    time_middle = time.perf_counter()
    pre = Main.HardwareFunctions.build_lq_preoptimization(10, 0.5, silence_logs=True)
    time_end = time.perf_counter()
    print(f"Time taken: {time_end - time_middle} seconds")
    print(f"Preoptimization built successfully")
    print(f"Total time taken: {time_end - time_start} seconds")
    return pre
    
def goal_reached(posittion, goal_position, threshold=0.3):
    """
    Check if the robot has reached the goal position within a threshold.
    """
    distance = math.sqrt((posittion[0] - goal_position[0]) ** 2 + (posittion[1] - goal_position[1]) ** 2)
    return distance < threshold


class MultiRobotController(Node):
    def __init__(self, pre):
        super().__init__('multi_robot_controller')

        # Publishers
        self.cmd_pub_01 = self.create_publisher(Twist, '/cmd_vel_01', 10)
        self.cmd_pub_02 = self.create_publisher(Twist, '/cmd_vel_02', 10)
        self.cmd_pub_03 = self.create_publisher(Twist, '/cmd_vel_03', 10)  # Placeholder for third robot if needed

        # Subscribers
        self.odom_sub_01 = self.create_subscription(
            Odometry, '/odom_01', self.odom_callback_01, 10)
        self.odom_sub_02 = self.create_subscription(
            Odometry, '/odom_02', self.odom_callback_02, 10)
        self.odom_sub_03 = self.create_subscription(
            Odometry, '/odom_03', self.odom_callback_03, 10)  # Placeholder for third robot if needed

        # Odometry buffers
        self.latest_odom_01 = None
        self.latest_odom_02 = None
        self.latest_odom_03 = None  # Placeholder for third robot if needed

        # Timer to run planner at 10 Hz
        self.timer = self.create_timer(0.1, self.run_planner_step)

        self.pre = pre

        self.get_logger().info("MultiRobotController node started.")

    def odom_callback_01(self, msg):
        self.latest_odom_01 = msg

    def odom_callback_02(self, msg):
        self.latest_odom_02 = msg

    def odom_callback_03(self, msg):
        self.latest_odom_03 = msg

    def convert_odom_to_state(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Convert quaternion to yaw (theta)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        v = msg.twist.twist.linear.x

        return [x, y, theta, v]

    def convert_to_cmd_vel(self, vx, vy, pose, target_pose, current_theta):
        v = math.hypot(vx, vy)
        # target_theta = math.atan2(vy, vx)
        target_theta = math.atan2(target_pose[1] - pose[1], target_pose[0] - pose[0])
        print("target_theta:", target_theta)
        print("current_theta:", current_theta)
        # if vy >= 0:
        #     target_theta = math.asin(vy, v)
        # else:
        #     target_theta = math.pi - math.asin(vy, vx)

        omega = (target_theta - current_theta)  # Assuming a time step of 0.1 seconds
        
        return v, omega

    def run_planner_step(self):
        if self.latest_odom_01 is None or self.latest_odom_02 is None:
            # only for placeholder
            # time_start = time.perf_counter()
            # result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(self.pre, [[0, 1], [0, 2], [0, 3]])
            # time_end = time.perf_counter()
            # print(f"Time taken: {time_end - time_start} seconds")
            self.get_logger().warn("Waiting for odometry...")
            return

        try:
            state1 = self.convert_odom_to_state(self.latest_odom_01)
            state2 = self.convert_odom_to_state(self.latest_odom_02)
            state3 = self.convert_odom_to_state(self.latest_odom_03)  # Placeholder for third robot if needed

            # print("state1:", state1)
            # print("state2:", state2)

            # Julia solver expects vector of vectors: [[px; py], [px; py], [px; py]]
            # Create Julia vectors explicitly to ensure correct data structure
            julia_state1 = Main.eval(f"[{state1[0]}; {state1[1]}]")
            julia_state2 = Main.eval(f"[{state2[0]}; {state2[1]}]") 
            julia_state3 = Main.eval(f"[{state3[0]}; {state3[1]}]")
            initial_state = [julia_state1, julia_state2, julia_state3]

            # sol = Main.nplayer_navigation(initial_state, guess)

            # result = Main.nplayer_hierarchy_navigation(initial_state)
            result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(self.pre, initial_state)
            
            # guess should be updated

            # result is (next_state, curr_control)
            # next_state: [[x1_next], [x2_next], [x3_next]]  
            # curr_control: [[u1_curr], [u2_curr], [u3_curr]] where ui_curr = [vx, vy]
            next_states, curr_controls = result
            
            # Debug logging
            self.get_logger().info(f"Robot positions: P1=[{state1[0]:.3f}, {state1[1]:.3f}], P2=[{state2[0]:.3f}, {state2[1]:.3f}], P3=[{state3[0]:.3f}, {state3[1]:.3f}]")
            self.get_logger().info(f"Next states: {next_states}")
            self.get_logger().info(f"Controls: {curr_controls}")
            
            # Extract control commands [vx, vy] for each robot
            u1 = curr_controls[0]  # [vx1, vy1]
            u2 = curr_controls[1]  # [vx2, vy2]
            u3 = curr_controls[2]  # [vx3, vy3]
            
            # Convert [vx, vy] to linear velocity and angular velocity
            v1, omega1 = self.convert_to_cmd_vel(u1[0], u1[1], [state1[0], state1[1]], next_states[0], state1[2])
            v2, omega2 = self.convert_to_cmd_vel(u2[0], u2[1], [state2[0], state2[1]], next_states[1], state2[2])
            v3, omega3 = self.convert_to_cmd_vel(u3[0], u3[1], [state3[0], state3[1]], next_states[2], state3[2])

            # Clip angular velocities for safety
            omega1 = np.clip(omega1, -0.5, 0.5)
            omega2 = np.clip(omega2, -0.5, 0.5)
            omega3 = np.clip(omega3, -0.5, 0.5)

            twist1 = Twist()
            twist1.linear.x = v1
            twist1.angular.z = omega1  # optional: compute omega

            twist2 = Twist()
            twist2.linear.x = v2
            twist2.angular.z = omega2

            twist3 = Twist()  # Placeholder for third robot if needed
            twist3.linear.x = v3
            twist3.angular.z = omega3

            if not goal_reached(state1[:2], [0, 0]):
                self.cmd_pub_01.publish(twist1)
            else:
                self._logger.info("Goal reached for robot 1, stopping.")
                twist1.linear.x = 0.0
                twist1.angular.z = 0.0
                self.cmd_pub_01.publish(twist1)

            if not goal_reached(state2[:2], [0, 0]):
                self.cmd_pub_02.publish(twist2)
            else:
                self._logger.info("Goal reached for robot 2, stopping.")
                twist2.linear.x = 0.0
                twist2.angular.z = 0.0
                self.cmd_pub_02.publish(twist2)

            if not goal_reached(state3[:2], [0, 0]):
                self.cmd_pub_03.publish(twist3)
            else:
                self._logger.info("Goal reached for robot 3, stopping.")
                twist3.linear.x = 0.0
                twist3.angular.z = 0.0
                self.cmd_pub_03.publish(twist3)

            # self.get_logger().info(f"Published: v1={v1:.2f}, v2={v2:.2f}")
            # self.get_logger().info(f"Published: omega1={omega1:.2f}, omega2={omega2:.2f}")

        except Exception as e:
            self.get_logger().error(f"Julia error: {e}")


def main(pre, args=None):
    rclpy.init(args=args)
    # res = preoptimizaton()
    # figure out how to pass res to node

    # if there is a variable conversion issue, try to use Hamzah's code
    
    node = MultiRobotController(pre)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    pre = julia_init()
    main(pre)
