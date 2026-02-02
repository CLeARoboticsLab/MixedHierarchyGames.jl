# ros2 topic pub /cmd_vel_01 geometry_msgs/msg/Twist "{'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}}"

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import csv

import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

GOAL_POSITION = [0.0, 0.0]

# Only do this once — import and include your Julia code
def julia_init():
    # Go up to the main project root: ros2/src/mixed_hierarchy_games/mixed_hierarchy_games -> main project
    project_root = str(Path(__file__).resolve().parents[4])
    time_start = time.perf_counter()
    
    # Pre-load OpenSSL_jll to use system OpenSSL libraries
    # Set environment variable first, then load OpenSSL_jll before any other packages
    Main.eval("""
        ENV["JULIA_SSL_LIBRARY"] = "system"
        # Clear any incompatible artifacts that might have been downloaded
        try
            import Pkg
            Pkg.Artifacts.clear_artifacts!()
        catch e
            @warn "Artifact clear failed" exception=e
        end
        # Load OpenSSL_jll with system libraries
        try
            using OpenSSL_jll
        catch e
            @warn "OpenSSL_jll preload failed, continuing anyway" exception=e
        end
    """)
    
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
    pre = Main.HardwareFunctions.build_lq_preoptimization(10, 0.1, silence_logs=True)
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
        super().__init__('mixed_hierarchy_games_controller')

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
        self.timer = self.create_timer(0.1, self.run_planner_step) # 0.5

        self.pre = pre
        self.z_guess = None  # optional warm-start guess for internal solver variables
        # Trajectory logging (from odometry)
        self.trajectory = []  # list of ((x1,y1), (x2,y2), (x3,y3))
        self.project_root = str(Path(__file__).resolve().parents[4])
        self.gif_output_path = Path(self.project_root) / "ros2" / "trajectory.gif"
        self.png_output_path = Path(self.project_root) / "ros2" / "trajectory_final.png"
        self._shutdown_initiated = False
        # CSV logging of trajectory and cmd_vel at fixed 0.1s timestep
        self.dt = 0.1
        self.step_index = 0
        self._init_csv_logger()

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
        target_theta = math.atan2(target_pose[1] - pose[1], target_pose[0] - pose[0])
        # print("target_theta:", target_theta)
        # print("current_theta:", current_theta)

        # Wrap angle error to [-pi, pi] to avoid discontinuities around the boundary
        def _wrap_to_pi(angle):
            return (angle + math.pi) % (2.0 * math.pi) - math.pi

        angle_error = _wrap_to_pi(target_theta - current_theta)

        # Proportional heading control with optional smoothing and clamping
        k_omega = getattr(self, "k_omega", 1.0)
        omega_cmd = k_omega * angle_error

        # Low-pass filter to avoid sudden sign flips (keep smallest inertia)
        alpha = getattr(self, "omega_lpf_alpha", 0.5)  # 0<alpha<=1, closer to 1 = less smoothing
        prev_omega = getattr(self, "prev_omega", 0.0)
        omega = alpha * omega_cmd + (1.0 - alpha) * prev_omega
        self.prev_omega = omega

        # Optional safety clamp here is light; final clamp happens before publish
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

    def _save_trajectory_gif(self):
        if len(self.trajectory) < 2:
            self.get_logger().warn("Not enough trajectory points to generate GIF.")
            return

        # Prepare data arrays
        x1 = [p[0][0] for p in self.trajectory]
        y1 = [p[0][1] for p in self.trajectory]
        x2 = [p[1][0] for p in self.trajectory]
        y2 = [p[1][1] for p in self.trajectory]
        x3 = [p[2][0] for p in self.trajectory]
        y3 = [p[2][1] for p in self.trajectory]

        all_x = x1 + x2 + x3
        all_y = y1 + y2 + y3
        # Include goal position in bounds
        gx, gy = float(GOAL_POSITION[0]), float(GOAL_POSITION[1])
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

        # Save final static figure
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

    def run_planner_step(self):
        if self.latest_odom_01 is None or self.latest_odom_02 is None or self.latest_odom_03 is None:
            # only for placeholder
            # time_start = time.perf_counter()
            # time_end = time.perf_counter()
            # print(f"Time taken: {time_end - time_start} seconds")
            self.get_logger().warn("Waiting for odometry...")
            return

        if True:
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
            
            # print("initial_state:", initial_state)
            # input("Press Enter to continue...")

            # sol = Main.nplayer_navigation(initial_state, guess)

            # result = Main.nplayer_hierarchy_navigation(initial_state)
            
            # self.get_logger().warn("Running hardware_nplayer_hierarchy_navigation...")
            # print("self.z_guess:", self.z_guess)
            # print("self.pre:", self.pre)
            # input("Press Enter to continue...")
            time_plan_start = time.perf_counter()
            result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(self.pre, initial_state, self.z_guess, silence_logs=False)
            time_plan_end = time.perf_counter()
            plan_time_ms = (time_plan_end - time_plan_start) * 1000.0
            # self.get_logger().info(f"Planning time: {plan_time_ms:.1f} ms")

            # print("result:", result)
            # input("Press Enter to continue...")

            # Guess should be updated
            next_states = result.x_next
            curr_controls = result.u_curr
            z_sol = result.z_sol

            # Update z_guess for warm-starting next iteration.
            self.z_guess = z_sol
            
            # Debug logging
            # self.get_logger().info(f"Robot positions: P1=[{state1[0]:.3f}, {state1[1]:.3f}], P2=[{state2[0]:.3f}, {state2[1]:.3f}], P3=[{state3[0]:.3f}, {state3[1]:.3f}]")
            # self.get_logger().info(f"Next states: {next_states}")
            # self.get_logger().info(f"Controls: {curr_controls}")
            
            # Extract control commands [vx, vy] for each robot
            u1 = curr_controls[0]  # [vx1, vy1]
            u2 = curr_controls[1]  # [vx2, vy2]
            u3 = curr_controls[2]  # [vx3, vy3]
            
            # Convert [vx, vy] to linear velocity and angular velocity
            # def convert_to_cmd_vel(self, vx, vy, pose, target_pose, current_theta)
            v1, omega1 = self.convert_to_cmd_vel(u1[0], u1[1], [state1[0], state1[1]], next_states[0], state1[2])
            v2, omega2 = self.convert_to_cmd_vel(u2[0], u2[1], [state2[0], state2[1]], next_states[1], state2[2])
            v3, omega3 = self.convert_to_cmd_vel(u3[0], u3[1], [state3[0], state3[1]], next_states[2], state3[2])

            # Record odometry-based trajectory
            self.trajectory.append(((state1[0], state1[1]), (state2[0], state2[1]), (state3[0], state3[1])))

            # Clip angular velocities for safety
            omega1 = np.clip(omega1, -0.3, 0.3)
            omega2 = np.clip(omega2, -0.3, 0.3)
            omega3 = np.clip(omega3, -0.3, 0.3)
            
            v1 = np.clip(v1, -0.2, 0.2)
            v2 = np.clip(v2, -0.2, 0.2)
            v3 = np.clip(v3, -0.2, 0.2)
            
            # self.get_logger().info(f"v1: {v1}, omega1: {omega1}, v2: {v2}, omega2: {omega2}, v3: {v3}, omega3: {omega3}")

            twist1 = Twist()
            twist1.linear.x = v1
            twist1.angular.z = omega1  # optional: compute omega

            twist2 = Twist()
            twist2.linear.x = v2
            twist2.angular.z = omega2

            twist3 = Twist()  # Placeholder for third robot if needed
            twist3.linear.x = v3
            twist3.angular.z = omega3

            # Write per-step CSV log using fixed timestep (0.1s)
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

            if not goal_reached(state3[:2], GOAL_POSITION):
                self.cmd_pub_01.publish(twist1)
                self.cmd_pub_02.publish(twist2)
                self.cmd_pub_03.publish(twist3)
            else:
                if not self._shutdown_initiated:
                    self.get_logger().info("Goal reached for evader, stopping.")
                    # Stop timer and publish zero velocities
                    self.timer.cancel()
                    self._publish_stop()
                    # Save trajectory GIF and shutdown
                    self._save_trajectory_gif()
                    self._shutdown_initiated = True
                    self.destroy_node()
                    rclpy.shutdown()

            # self.get_logger().info(f"Published: v1={v1:.2f}, v2={v2:.2f}")
            # self.get_logger().info(f"Published: omega1={omega1:.2f}, omega2={omega2:.2f}")


    def destroy_node(self):
        # Ensure CSV file is closed on shutdown
        self._close_csv()
        super().destroy_node()

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
