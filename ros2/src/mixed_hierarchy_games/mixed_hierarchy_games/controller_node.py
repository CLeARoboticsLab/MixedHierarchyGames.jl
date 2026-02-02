"""
ROS2 controller node for multi-robot pursuit–evasion with a Julia backend.

This node:
- Subscribes to odometry for three agents
- Calls a Julia-based planner to compute next-step controls
- Publishes `geometry_msgs/Twist` commands
- Logs trajectories to CSV and (optionally) saves a GIF/PNG of the paths

Parameters (ROS2):
- dt (double, default: 0.1): Node timer step (s)
- max_linear_speed (double, default: 0.2): Speed clamp for published linear.x (m/s)
- max_angular_speed (double, default: 0.3): Speed clamp for published angular.z (rad/s)
- k_omega (double, default: 1.0): Proportional gain for heading control
- omega_lpf_alpha (double, default: 0.5): Low-pass filter alpha for angular velocity
- max_omega_local (double, default: 1.0): Local clamp inside heading controller
- goal_position (double[2], default: [0.0, 0.0]): Evader goal position [x, y]
- enable_plotting (bool, default: True): Save GIF/PNG on completion
- solver_verbose (bool, default: False): Verbosity for Julia solver at each step
- preopt_horizon (int, default: 10): Pre-optimization horizon (forward steps) for Julia

Notes:
- The Julia project root is auto-discovered by searching upward for `Project.toml`.
- As a fallback, a fixed ancestor is used to preserve backward compatibility.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import csv
import logging
from typing import Any, List, Optional, Sequence, Tuple
import sys
import os

import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

def find_project_root() -> str:
    """
    Find the Julia project root by searching up from this file for `Project.toml`.
    Falls back to a fixed ancestor to preserve prior behavior.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / "Project.toml"
        if candidate.exists():
            return str(parent)
    return str(here.parents[4])


def julia_init(preopt_horizon: int = 10, dt: float = 0.1, project_root: Optional[str] = None) -> Any:
    """
    Initialize Julia, activate the project, include necessary files, and build preoptimization.
    """
    logger = logging.getLogger("julia_init")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    root = project_root or find_project_root()
    logger.info(f"Using Julia project root: {root}")

    time_start = time.perf_counter()
    try:
        Main.eval(
            f"""
            import Pkg
            Pkg.activate(raw"{root}")
            Pkg.instantiate()
            Base.include(Main, joinpath(raw"{root}", "examples", "automatic_solver.jl"))
            Base.include(Main, joinpath(raw"{root}", "examples", "test_automatic_solver.jl"))
            Base.include(Main, joinpath(raw"{root}", "examples", "hardware_functions.jl"))
            """
        )
    except Exception as exc:
        logger.exception("Failed to initialize Julia project and include files.")
        raise

    # Build preoptimization once
    time_middle = time.perf_counter()
    try:
        pre = Main.HardwareFunctions.build_lq_preoptimization(int(preopt_horizon), float(dt), silence_logs=True)
    except Exception as exc:
        logger.exception("Failed to build Julia preoptimization.")
        raise
    time_end = time.perf_counter()
    logger.info(f"Preoptimization built successfully in {time_end - time_middle:.3f}s "
                f"(total init {time_end - time_start:.3f}s)")
    return pre
    
def goal_reached(position: Sequence[float], goal_position: Sequence[float], threshold: float = 0.3) -> bool:
    """
    Check if the robot has reached the goal position within a threshold.
    """
    distance = math.sqrt((position[0] - goal_position[0]) ** 2 + (position[1] - goal_position[1]) ** 2)
    return distance < threshold


class MultiRobotController(Node):
    def __init__(self, pre: Any):
        super().__init__('mixed_hierarchy_games_controller')

        # Parameters
        self.dt: float = float(self.declare_parameter('dt', 0.1).value)
        self.max_linear_speed: float = float(self.declare_parameter('max_linear_speed', 0.2).value)
        self.max_angular_speed: float = float(self.declare_parameter('max_angular_speed', 0.3).value)
        self.k_omega: float = float(self.declare_parameter('k_omega', 1.0).value)
        self.omega_lpf_alpha: float = float(self.declare_parameter('omega_lpf_alpha', 0.5).value)
        self.max_omega_local: float = float(self.declare_parameter('max_omega_local', 1.0).value)
        self.goal_position: List[float] = list(self.declare_parameter('goal_position', [0.0, 0.0]).value)
        self.enable_plotting: bool = bool(self.declare_parameter('enable_plotting', True).value)
        self.solver_verbose: bool = bool(self.declare_parameter('solver_verbose', False).value)

        # Publishers
        self.cmd_pub_01 = self.create_publisher(Twist, '/cmd_vel_01', 10)
        self.cmd_pub_02 = self.create_publisher(Twist, '/cmd_vel_02', 10)
        self.cmd_pub_03 = self.create_publisher(Twist, '/cmd_vel_03', 10)

        # Subscribers
        self.odom_sub_01 = self.create_subscription(
            Odometry, '/odom_01', self.odom_callback_01, 10)
        self.odom_sub_02 = self.create_subscription(
            Odometry, '/odom_02', self.odom_callback_02, 10)
        self.odom_sub_03 = self.create_subscription(
            Odometry, '/odom_03', self.odom_callback_03, 10)

        # Odometry buffers
        self.latest_odom_01 = None
        self.latest_odom_02 = None
        self.latest_odom_03 = None

        # Timer based on parameterized dt
        self.timer = self.create_timer(self.dt, self.run_planner_step)

        self.pre = pre
        self.z_guess = None  
        self.trajectory = []
        self.project_root = str(Path(__file__).resolve().parents[4])
        self.gif_output_path = Path(self.project_root) / "ros2" / "trajectory.gif"
        self.png_output_path = Path(self.project_root) / "ros2" / "trajectory.png"
        self._shutdown_initiated = False
        self._shutdown_timer = None
        # CSV logging of trajectory and cmd_vel at fixed timestep
        self.step_index = 0
        self._init_csv_logger()

        self.get_logger().info("MultiRobotController node started.")

    def odom_callback_01(self, msg):
        self.latest_odom_01 = msg

    def odom_callback_02(self, msg):
        self.latest_odom_02 = msg

    def odom_callback_03(self, msg):
        self.latest_odom_03 = msg

    def convert_odom_to_state(self, msg: Odometry) -> List[float]:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Convert quaternion to yaw (theta)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        v = msg.twist.twist.linear.x

        return [x, y, theta, v]

    def convert_to_cmd_vel(
        self,
        vx: float,
        vy: float,
        pose: Sequence[float],
        target_pose: Sequence[float],
        current_theta: float,
    ) -> Tuple[float, float]:
        v = math.hypot(vx, vy)
        target_theta = math.atan2(target_pose[1] - pose[1], target_pose[0] - pose[0])

        def _wrap_to_pi(angle):
            return (angle + math.pi) % (2.0 * math.pi) - math.pi

        angle_error = _wrap_to_pi(target_theta - current_theta)

        k_omega = getattr(self, "k_omega", 1.0)
        omega_cmd = k_omega * angle_error

        alpha = getattr(self, "omega_lpf_alpha", 0.5)   
        prev_omega = getattr(self, "prev_omega", 0.0)
        omega = alpha * omega_cmd + (1.0 - alpha) * prev_omega
        self.prev_omega = omega

        max_omega_local = getattr(self, "max_omega_local", 1.0)
        omega = max(-max_omega_local, min(max_omega_local, omega))
        
        return v, omega
    
    def _publish_stop(self):
        twist_zero = Twist()
        twist_zero.linear.x = 0.0
        twist_zero.angular.z = 0.0
        self.cmd_pub_01.publish(twist_zero)
        self.cmd_pub_02.publish(twist_zero)
        self.cmd_pub_03.publish(twist_zero)

    def _schedule_shutdown(self):
        if self._shutdown_initiated:
            return
        self._shutdown_initiated = True
        def _do_shutdown():
            try:
                if self.timer is not None:
                    self.timer.cancel()
            except Exception:
                pass
            self._publish_stop()
            self._save_trajectory_gif()
            try:
                self.destroy_node()
            finally:
                rclpy.shutdown()
            try:
                if self._shutdown_timer is not None:
                    self._shutdown_timer.cancel()
            except Exception:
                pass
        self._shutdown_timer = self.create_timer(0.01, _do_shutdown)

    def _save_trajectory_gif(self):
        if not self.enable_plotting:
            return
        if len(self.trajectory) < 2:
            self.get_logger().warn("Not enough trajectory points to generate GIF.")
            return

        x1 = [p[0][0] for p in self.trajectory]
        y1 = [p[0][1] for p in self.trajectory]
        x2 = [p[1][0] for p in self.trajectory]
        y2 = [p[1][1] for p in self.trajectory]
        x3 = [p[2][0] for p in self.trajectory]
        y3 = [p[2][1] for p in self.trajectory]

        all_x = x1 + x2 + x3
        all_y = y1 + y2 + y3
        gx, gy = float(self.goal_position[0]), float(self.goal_position[1])
        all_x.append(gx)
        all_y.append(gy)
        xmin, xmax = min(all_x) - 0.5, max(all_x) + 0.5
        ymin, ymax = min(all_y) - 0.5, max(all_y) + 0.5

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title("Agent Trajectories")

        line1, = ax.plot([], [], 'r-', lw=2, label='Pursuer')
        line2, = ax.plot([], [], 'g-', lw=2, label='Guard')
        line3, = ax.plot([], [], 'b-', lw=2, label='Evader')
        pt1, = ax.plot([], [], 'ro', ms=5)
        pt2, = ax.plot([], [], 'go', ms=5)
        pt3, = ax.plot([], [], 'bo', ms=5)
        goal_label = f'Evader Goal'
        goal_pt, = ax.plot([gx], [gy], 'k*', ms=10, label=goal_label)
        ax.legend(loc='upper right')

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            pt1.set_data([], [])
            pt2.set_data([], [])
            pt3.set_data([], [])
            return line1, line2, line3, pt1, pt2, pt3, goal_pt

        def animate(i):
            line1.set_data(x1[:i+1], y1[:i+1])
            line2.set_data(x2[:i+1], y2[:i+1])
            line3.set_data(x3[:i+1], y3[:i+1])
            pt1.set_data([x1[i]], [y1[i]])
            pt2.set_data([x2[i]], [y2[i]])
            pt3.set_data([x3[i]], [y3[i]])
            return line1, line2, line3, pt1, pt2, pt3, goal_pt

        ani = animation.FuncAnimation(
            fig, animate, init_func=init, frames=len(self.trajectory),
            interval=100, blit=True
        )

        self.gif_output_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(self.gif_output_path), writer='pillow', fps=10)
        self.get_logger().info(f"Saved trajectory GIF to: {self.gif_output_path}")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.set_title("Final Pursuit-Evasion Trajectories")
        ax2.plot(x1, y1, 'r-', lw=2, label='Pursuer')
        ax2.plot(x2, y2, 'g-', lw=2, label='Guard')
        ax2.plot(x3, y3, 'b-', lw=2, label='Evader')
        ax2.plot([x1[-1]], [y1[-1]], 'ro', ms=5)
        ax2.plot([x2[-1]], [y2[-1]], 'go', ms=5)
        ax2.plot([x3[-1]], [y3[-1]], 'bo', ms=5)
        goal_label2 = f'Evader Goal'
        ax2.plot([gx], [gy], 'k*', ms=10, label=goal_label2)
        ax2.legend(loc='upper right')
        fig2.savefig(str(self.png_output_path), dpi=150, bbox_inches='tight')
        self.get_logger().info(f"Saved final trajectory figure to: {self.png_output_path}")
        plt.close(fig2)

    def _init_csv_logger(self):
        self.csv_output_path = Path(self.project_root) / "ros2" / "trajectory_log.csv"
        self.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(str(self.csv_output_path), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time_s",
            "x1", "y1",
            "x2", "y2",
            "x3", "y3",
            "v1", "omega1",
            "v2", "omega2",
            "v3", "omega3",
            "plan_time_ms",
        ])
        self.csv_file.flush()

    def _close_csv(self):
        if hasattr(self, "csv_file") and self.csv_file and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f"Saved trajectory CSV to: {self.csv_output_path}")

    def run_planner_step(self) -> None:
        if self.latest_odom_01 is None or self.latest_odom_02 is None or self.latest_odom_03 is None:
            self.get_logger().warn("Waiting for odometry...")
            return

        try:
            state1 = self.convert_odom_to_state(self.latest_odom_01)
            state2 = self.convert_odom_to_state(self.latest_odom_02)
            state3 = self.convert_odom_to_state(self.latest_odom_03)
        except Exception:
            self.get_logger().error("Failed to parse odometry messages.")
            return

        # Julia solver expects vector of vectors: [[px; py], [px; py], [px; py]]
        try:
            julia_state1 = Main.eval(f"[{state1[0]}; {state1[1]}]")
            julia_state2 = Main.eval(f"[{state2[0]}; {state2[1]}]")
            julia_state3 = Main.eval(f"[{state3[0]}; {state3[1]}]")
            initial_state = [julia_state1, julia_state2, julia_state3]
        except Exception:
            self.get_logger().error("Failed to construct Julia initial state.")
            return
        
        try:
            time_plan_start = time.perf_counter()
            result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(
                self.pre, initial_state, self.z_guess, silence_logs=not self.solver_verbose
            )
            time_plan_end = time.perf_counter()
            plan_time_ms = (time_plan_end - time_plan_start) * 1000.0
        except Exception:
            self.get_logger().error("Julia planner call failed; publishing stop for safety.")
            self._publish_stop()
            return

        try:
            next_states = result.x_next
            curr_controls = result.u_curr
            self.z_guess = result.z_sol
        except Exception:
            self.get_logger().error("Unexpected planner result structure; publishing stop.")
            self._publish_stop()
            return
        
        try:
            u1 = curr_controls[0]  # [vx1, vy1]
            u2 = curr_controls[1]  # [vx2, vy2]
            u3 = curr_controls[2]  # [vx3, vy3]
        except Exception:
            self.get_logger().error("Invalid control output from planner; publishing stop.")
            self._publish_stop()
            return
        
        v1, omega1 = self.convert_to_cmd_vel(u1[0], u1[1], [state1[0], state1[1]], next_states[0], state1[2])
        v2, omega2 = self.convert_to_cmd_vel(u2[0], u2[1], [state2[0], state2[1]], next_states[1], state2[2])
        v3, omega3 = self.convert_to_cmd_vel(u3[0], u3[1], [state3[0], state3[1]], next_states[2], state3[2])

        self.trajectory.append(((state1[0], state1[1]), (state2[0], state2[1]), (state3[0], state3[1])))

        omega1 = float(np.clip(omega1, -self.max_angular_speed, self.max_angular_speed))
        omega2 = float(np.clip(omega2, -self.max_angular_speed, self.max_angular_speed))
        omega3 = float(np.clip(omega3, -self.max_angular_speed, self.max_angular_speed))
        
        v1 = float(np.clip(v1, -self.max_linear_speed, self.max_linear_speed))
        v2 = float(np.clip(v2, -self.max_linear_speed, self.max_linear_speed))
        v3 = float(np.clip(v3, -self.max_linear_speed, self.max_linear_speed))
        
        twist1 = Twist()
        twist1.linear.x = v1
        twist1.angular.z = omega1

        twist2 = Twist()
        twist2.linear.x = v2
        twist2.angular.z = omega2

        twist3 = Twist()
        twist3.linear.x = v3
        twist3.angular.z = omega3

        t_now = self.step_index * self.dt
        if hasattr(self, "csv_writer"):
            self.csv_writer.writerow([
                f"{t_now:.1f}",
                state1[0], state1[1],
                state2[0], state2[1],
                state3[0], state3[1],
                v1, omega1,
                v2, omega2,
                v3, omega3,
                f"{plan_time_ms:.1f}",
            ])
        if hasattr(self, "csv_file"):
            self.csv_file.flush()
        self.step_index += 1

        if not goal_reached(state3[:2], self.goal_position):
            self.cmd_pub_01.publish(twist1)
            self.cmd_pub_02.publish(twist2)
            self.cmd_pub_03.publish(twist3)
        else:
            if not self._shutdown_initiated:
                self.get_logger().info("Goal reached for evader, stopping. Initiating shutdown...")
                self._schedule_shutdown()

    def destroy_node(self):
        self._close_csv()
        super().destroy_node()

def main(pre: Any, args=None):
    rclpy.init(args=args)
    node = MultiRobotController(pre)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    pre = julia_init()
    main(pre)
