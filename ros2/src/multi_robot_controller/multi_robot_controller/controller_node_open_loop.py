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
    # return distance < threshold
    return False


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
        self.timer = self.create_timer(0.5, self.run_planner_step) # 0.5

        self.pre = pre
        self.z_guess = None  # optional warm-start guess for internal solver variables

        # Publishers grouped for convenience
        self.cmd_publishers = [
            self.cmd_pub_01,
            self.cmd_pub_02,
            self.cmd_pub_03,
        ]

        # Open-loop execution state
        self.open_loop_plan = None
        self.plan_ready = False
        self.plan_step = 0
        self.plan_steps = 0
        self.plan_stop_sent = False

        # Flag to avoid spamming the odometry wait log
        self._waiting_for_initial_odometry = True

        # Plan execution timing
        self.control_period = 0.5
        self._last_plan_time = 0.0

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
        if not self._initial_odometry_ready():
            self._wait_for_initial_odometry()
            return

        if not self.plan_ready:
            self._attempt_build_open_loop_plan()
            return

        if self.plan_step >= self.plan_steps:
            if not self.plan_stop_sent:
                self._publish_stop_twists()
                self.plan_stop_sent = True
                self.get_logger().info("Open-loop sequence finished; robots stopped.")
            return

        now = time.perf_counter()
        if now - self._last_plan_time < self.control_period:
            return

        self._last_plan_time = now
        self._publish_plan_step(self.plan_step)
        self.plan_step += 1

    def _attempt_build_open_loop_plan(self):
        if (self.latest_odom_01 is None or self.latest_odom_02 is None
                or self.latest_odom_03 is None):
            self.get_logger().warn("Waiting for initial odometry to compute open-loop plan...")
            return

        state1 = self.convert_odom_to_state(self.latest_odom_01)
        state2 = self.convert_odom_to_state(self.latest_odom_02)
        state3 = self.convert_odom_to_state(self.latest_odom_03)

        initial_state = [
            Main.eval(f"[{state1[0]}; {state1[1]}]"),
            Main.eval(f"[{state2[0]}; {state2[1]}]"),
            Main.eval(f"[{state3[0]}; {state3[1]}]"),
        ]

        try:
            result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(
                self.pre, initial_state, self.z_guess, silence_logs=True)
        except Exception as e:
            self.get_logger().error(f"Julia solver failed during open-loop planning: {e}")
            return

        self.z_guess = self._get_field(result, "z_sol")

        plan = self._build_open_loop_plan(result)
        if plan is None or plan["num_steps"] == 0:
            self.get_logger().warn("Open-loop plan is empty; retrying on next tick.")
            return

        self.open_loop_plan = plan
        self.plan_steps = plan["num_steps"]
        self.plan_step = 0
        self.plan_ready = True
        self.plan_stop_sent = False

        self._last_plan_time = time.perf_counter() - self.control_period

        robot_positions = [
            [state1[0], state1[1]],
            [state2[0], state2[1]],
            [state3[0], state3[1]],
        ]
        self.get_logger().info(f"Computed open-loop sequence ({self.plan_steps} steps) starting from {robot_positions}.")

    def _publish_plan_step(self, step_idx):
        if self.open_loop_plan is None:
            return

        for idx, publisher in enumerate(self.cmd_publishers):
            robot_plan = self.open_loop_plan["players"][idx]
            if step_idx >= robot_plan["num_steps"]:
                publisher.publish(Twist())
                continue

            controls = robot_plan["controls"][step_idx]
            states = robot_plan["states"]
            pose = states[step_idx][:2]
            target_pose = states[min(step_idx + 1, len(states) - 1)][:2]
            current_theta = robot_plan["thetas"][step_idx]

            v, omega = self.convert_to_cmd_vel(controls[0], controls[1], pose, target_pose, current_theta)
            omega = np.clip(omega, -0.5, 0.5)

            twist = Twist()
            twist.linear.x = v
            twist.angular.z = omega
            publisher.publish(twist)

    def _publish_stop_twists(self):
        for publisher in self.cmd_publishers:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            publisher.publish(twist)

    def _build_open_loop_plan(self, result):
        z_sol = self._get_field(result, "z_sol")
        if z_sol is None:
            return None

        z_list = self._normalize_value(z_sol)
        if not isinstance(z_list, list):
            return None

        try:
            n_players = int(self._get_field(self.pre, "N"))
            T = int(self._get_field(self.pre, "T"))
            problem_dims = self._get_field(self.pre, "problem_dims")
            state_dim = int(self._get_field(problem_dims, "state_dimension"))
            control_dim = int(self._get_field(problem_dims, "control_dimension"))
        except Exception as exc:
            self.get_logger().error(f"Failed to read problem dimensions from preoptimization: {exc}")
            return None

        x_dim = state_dim * (T + 1)
        u_dim = control_dim * (T + 1)
        primal_dim = x_dim + u_dim

        player_plans = []
        for i in range(n_players):
            start = i * primal_dim
            segment = z_list[start:start + primal_dim]
            if len(segment) != primal_dim:
                self.get_logger().warn("Unexpected z_sol size for player %d", i + 1)
                return None

            states = [
                segment[t * state_dim:(t + 1) * state_dim]
                for t in range(T + 1)
            ]
            controls = [
                segment[x_dim + t * control_dim:x_dim + (t + 1) * control_dim]
                for t in range(T + 1)
            ]
            player_plans.append({
                "states": states,
                "controls": controls,
            })

        step_counts = []
        for robot_plan in player_plans:
            states = robot_plan["states"]
            controls = robot_plan["controls"]
            steps = max(0, min(len(states) - 1, len(controls)))
            robot_plan["num_steps"] = steps
            step_counts.append(steps)

        if not step_counts:
            return None

        overall_steps = min(step_counts)
        if overall_steps == 0:
            return None

        for robot_plan in player_plans:
            steps = robot_plan["num_steps"]
            thetas = []
            for t in range(steps):
                current = robot_plan["states"][t]
                future = robot_plan["states"][min(t + 1, len(robot_plan["states"]) - 1)]
                dx = future[0] - current[0]
                dy = future[1] - current[1]
                if dx == 0 and dy == 0:
                    angle = thetas[-1] if thetas else 0.0
                else:
                    angle = math.atan2(dy, dx)
                thetas.append(angle)
            robot_plan["thetas"] = thetas

        return {"players": player_plans, "num_steps": overall_steps}

    def _normalize_value(self, value):
        try:
            tl = value.tolist()
        except Exception:
            tl = None
        if tl is not None:
            return self._normalize_value(tl)
        if isinstance(value, (list, tuple)):
            return [self._normalize_value(v) for v in value]
        try:
            return float(value)
        except Exception:
            return value

    def _get_field(self, obj, name):
        try:
            return obj[name]
        except Exception:
            return getattr(obj, name)

    def _initial_odometry_ready(self):
        return (
            self.latest_odom_01 is not None
            and self.latest_odom_02 is not None
            and self.latest_odom_03 is not None
        )

    def _wait_for_initial_odometry(self):
        if not self._waiting_for_initial_odometry:
            return
        self.get_logger().warn("Waiting for initial odometry before running open-loop solver...")
        self._waiting_for_initial_odometry = False


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
