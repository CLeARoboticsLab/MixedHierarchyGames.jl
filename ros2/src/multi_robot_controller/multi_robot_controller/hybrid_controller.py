import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import math
import time
from pathlib import Path
import csv
from julia.api import Julia
# Set environment variable BEFORE Julia is initialized
import os
os.environ["GKSwstype"] = "nul"
os.environ["JULIA_SSL_LIBRARY"] = "system"
jl = Julia(compiled_modules=False)
from julia import Main
import roslibpy
from roslibpy.core import RosTimeoutError

GOAL_POSITION = [0.0, 0.0]
GOAL_THRESHOLD = 0.6  # meters - stop when robot 3 is within this distance

# Robot configuration - modify these for your setup
ROBOT_CONFIGS = [
    {'ip': '192.168.50.25', 'port': 9090, 'topic': '/bluebonnet/platform/cmd_vel'},  # Robot 1
    {'ip': '192.168.50.2', 'port': 9090, 'topic': '/lonebot/platform/cmd_vel'},     # Robot 2
    {'ip': '192.168.50.49', 'port': 9090, 'topic': '/hookem/platform/cmd_vel'},      # Robot 3
]

# Controller gains for Robot 3 (go_to_origin)
K_LINEAR = 0.5   # Proportional gain for linear velocity
K_ANGULAR = 1.0  # Proportional gain for angular velocity
MAX_LINEAR_VEL = 0.5   # m/s
MAX_ANGULAR_VEL = 0.5  # rad/s


def connect_to_robots():
    """Connect to robots 1 and 2 via roslibpy BEFORE Julia initialization.
    Note: Robot 3 is controlled separately by go_to_origin.py.
    Note: roslibpy uses Twisted reactor which can only be started once.
    We create all clients first, then start the reactor once."""
    print("Connecting to robots 1 and 2...")
    ros_clients = []
    ros_publishers = []
    
    # Step 1: Create clients for robots 1 and 2 only (robot 3 is controlled separately)
    for i, config in enumerate(ROBOT_CONFIGS[:2]):  # Only first 2 robots
        try:
            print(f"Creating client for robot {i+1} at {config['ip']}:{config['port']}...")
            client = roslibpy.Ros(host=config['ip'], port=config['port'])
            ros_clients.append(client)
        except Exception as e:
            print(f"✗ Error creating client for robot {i+1}: {e}")
            ros_clients.append(None)
    
    # Step 2: Start reactor once (using the first valid client)
    reactor_started = False
    
    for i, client in enumerate(ros_clients):
        if client is not None:
            try:
                print(f"Attempting to start reactor with robot {i+1} connection...")
                client.run()  # This starts the Twisted reactor (can only be done once)
                reactor_started = True
                print(f"✓ Reactor started (robot {i+1} may still be connecting...)")
                break
            except RosTimeoutError as e:
                print(f"✗ Robot {i+1} connection timeout, but reactor may be running")
                reactor_started = True
                break
            except Exception as e:
                error_str = str(e)
                if "ReactorNotRestartable" in error_str:
                    reactor_started = True
                    print(f"✓ Reactor already running (from previous attempt)")
                    break
                print(f"✗ Error starting reactor with robot {i+1}: {e}")
                continue
    
    if not reactor_started:
        print("✗ Failed to start reactor - no valid clients")
        return [None] * 2, [None] * 2  # Only robots 1 and 2
    
    # Step 3: Wait for connections and create publishers
    time.sleep(0.5)  # Give connections time to establish
    
    for i, (client, config) in enumerate(zip(ros_clients, ROBOT_CONFIGS[:2])):  # Only first 2 robots
        if client is None:
            ros_publishers.append(None)
            continue
            
        try:
            # Wait for connection with timeout
            max_wait_time = 5.0
            wait_interval = 0.1
            waited = 0.0
            while not client.is_connected and waited < max_wait_time:
                time.sleep(wait_interval)
                waited += wait_interval
            
            if client.is_connected:
                print(f"✓ Connected to robot {i+1} at {config['ip']}:{config['port']}")
                pub = roslibpy.Topic(client, config['topic'], 'geometry_msgs/msg/TwistStamped')
                pub.advertise()
                ros_publishers.append(pub)
            else:
                print(f"✗ Failed to connect to robot {i+1} at {config['ip']}:{config['port']} (timeout)")
                ros_publishers.append(None)
        except Exception as e:
            print(f"✗ Error setting up robot {i+1}: {e}")
            ros_publishers.append(None)
    
    print(f"Robot connection complete: {sum(1 for c in ros_clients if c is not None and c.is_connected)}/2 connected (robots 1 & 2)")
    return ros_clients, ros_publishers


# Only do this once — import and include your Julia code
def julia_init():
    # Go up to the main project root: ros2/src/multi_robot_controller/multi_robot_controller -> main project
    project_root = str(Path(__file__).resolve().parents[4])
    time_start = time.perf_counter()
    
    # Pre-load OpenSSL_jll to use system OpenSSL libraries
    Main.eval("""
        ENV["JULIA_SSL_LIBRARY"] = "system"
        ENV["GKSwstype"] = "nul"
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
        
        ENV["GKSwstype"] = "nul"
        
        try
            using Plots
            try
                gr(show = false, fmt = :png)
            catch e2
                @warn "gr() configuration failed" exception=e2
            end
        catch e
            @error "Failed to pre-load Plots with headless backend" exception=e
            @warn "This may cause Qt6/FreeType issues. Continuing anyway..."
        end

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
    
def goal_reached(position, goal_position, threshold=0.3):
    """
    Check if the robot has reached the goal position within a threshold.
    """
    distance = math.sqrt((position[0] - goal_position[0]) ** 2 + (position[1] - goal_position[1]) ** 2)
    return distance < threshold


def ros_time():
    """Generate ROS time stamp dictionary for TwistStamped messages."""
    t = time.time()
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    return {'sec': sec, 'nanosec': nanosec}


class HybridController(Node):
    def __init__(self, pre, ros_clients, ros_publishers):
        super().__init__('hybrid_controller')

        # Subscribers for pose (ROS 2) - vrpn_client_ros2 publishes PoseStamped
        self.odom_sub_01 = self.create_subscription(
            PoseStamped, '/vrpn_client_node/BlueBonnet/pose', self.odom_callback_01, 10)
        self.odom_sub_02 = self.create_subscription(
            PoseStamped, '/vrpn_client_node/Lonebot/pose', self.odom_callback_02, 10)
        self.odom_sub_03 = self.create_subscription(
            PoseStamped, '/vrpn_client_node/Husky/pose', self.odom_callback_03, 10)

        # Odometry buffers
        self.latest_odom_01 = None
        self.latest_odom_02 = None
        self.latest_odom_03 = None

        # Current commands to publish (updated by controllers, published at high rate)
        # Only control robots 1 and 2 (robot 3 is controlled separately by go_to_origin.py)
        self.current_commands = [(0.0, 0.0), (0.0, 0.0)]  # [(v1, omega1), (v2, omega2)]
        
        # Per-robot previous omega for low-pass filtering (for robots 1 and 2)
        self.prev_omega = [0.0, 0.0]
        
        # Track robot 3's initial position to detect when it moves from start
        self.robot3_initial_position = None
        self.robot3_moving = False
        self.robot3_movement_threshold = 0.1  # meters - activate robots 1 & 2 when robot 3 moves 10cm from initial position

        # Timer to run pursuit-evasion controller for robots 1 & 2 at 10 Hz
        self.pursuit_evasion_timer = self.create_timer(0.1, self.pursuit_evasion_loop)
        
        # High-rate publisher timer at 100 Hz (0.01s) for smoother movement
        self.publisher_timer = self.create_timer(0.01, self._publish_current_commands)

        self.pre = pre
        self.z_guess = None  # optional warm-start guess for internal solver variables
        
        # Trajectory logging (from odometry)
        self.trajectory = []  # list of ((x1,y1), (x2,y2), (x3,y3))
        self.project_root = str(Path(__file__).resolve().parents[4])
        self.csv_output_path = Path(self.project_root) / "ros2" / "trajectory.csv"
        self._shutdown_initiated = False

        # Use pre-established roslibpy connections (created before Julia init)
        self.ros_clients = ros_clients
        self.ros_publishers = ros_publishers
        
        # Log connection status
        connected_count = sum(1 for c in self.ros_clients if c is not None)
        self.get_logger().info(f"Using {connected_count}/2 pre-established robot connections (robots 1 & 2)")
        for i, client in enumerate(self.ros_clients):
            if client is not None and client.is_connected:
                self.get_logger().info(f"Robot {i+1} connection active")
            else:
                self.get_logger().warn(f"Robot {i+1} connection not available")

        self.get_logger().info("HybridController node started.")
        self.get_logger().info("Robot 1 & 2: Pursuit-Evasion (Julia solver)")
        self.get_logger().info("Robot 3: Controlled separately by go_to_origin.py")
        self.get_logger().info("Robots 1 & 2 will wait until Robot 3 starts moving")

    def odom_callback_01(self, msg):
        self.latest_odom_01 = msg

    def odom_callback_02(self, msg):
        self.latest_odom_02 = msg

    def odom_callback_03(self, msg):
        self.latest_odom_03 = msg
        # Check if robot 3 has moved from its initial position
        if self.latest_odom_03 is not None:
            current_pos = [msg.pose.position.x, msg.pose.position.y]
            
            # Store initial position on first callback
            if self.robot3_initial_position is None:
                self.robot3_initial_position = current_pos
                self.get_logger().info(f"Robot 3 initial position recorded: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
                return
            
            # Calculate distance from initial position
            distance_from_initial = math.sqrt(
                (current_pos[0] - self.robot3_initial_position[0]) ** 2 +
                (current_pos[1] - self.robot3_initial_position[1]) ** 2
            )
            
            # Activate robots 1 & 2 when robot 3 has moved 10cm from initial position
            if distance_from_initial > self.robot3_movement_threshold:
                if not self.robot3_moving:
                    self.robot3_moving = True
                    self.get_logger().info(
                        f"Robot 3 moved {distance_from_initial:.3f}m from initial position (threshold: {self.robot3_movement_threshold}m) - Activating robots 1 & 2")

    def convert_odom_to_state(self, msg):
        # Handle PoseStamped messages from vrpn_client_ros2
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        v = 0.0  # PoseStamped doesn't have velocity, default to 0
        
        # Convert quaternion to yaw (theta)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        return [x, y, theta, v]

    def convert_to_cmd_vel(self, vx, vy, pose, target_pose, current_theta, robot_idx=0):
        """Convert [vx, vy] to [v, omega] for robots 1 and 2 (pursuit-evasion)."""
        v = math.hypot(vx, vy)
        target_theta = math.atan2(target_pose[1] - pose[1], target_pose[0] - pose[0])
        
        # Normalize angle difference to [-pi, pi] to avoid discontinuities
        def _wrap_to_pi(angle):
            return (angle + math.pi) % (2.0 * math.pi) - math.pi
        
        angle_error = _wrap_to_pi(target_theta - current_theta)
        
        # Proportional controller for angular velocity
        k_omega = 1.5  # Normal gain for smaller robots
        omega_cmd = k_omega * angle_error
        
        # Low-pass filter to smooth out rapid changes
        alpha = 0.6  # Normal smoothing for smaller robots
        prev_omega = self.prev_omega[robot_idx] if robot_idx < len(self.prev_omega) else 0.0
        omega = alpha * omega_cmd + (1.0 - alpha) * prev_omega
        
        # Update stored value for next iteration
        if robot_idx < len(self.prev_omega):
            self.prev_omega[robot_idx] = omega
        
        return v, omega


    def _publish_current_commands(self):
        """High-rate publisher that sends current commands repeatedly for smoother movement.
        Runs at 100 Hz (every 0.01s) to publish commands multiple times during each 0.1s controller step.
        Only publishes to robots 1 and 2 (robot 3 is controlled separately)."""
        if self._shutdown_initiated:
            return
        
        # Only publish to robots 1 and 2 (robot 3 is controlled by go_to_origin.py)
        for i, (vx, omega) in enumerate(self.current_commands):
            if i < 2 and i < len(self.ros_publishers) and self.ros_publishers[i] is not None:
                try:
                    msg = {
                        'header': {'stamp': ros_time(), 'frame_id': 'teleop_twist_joy'},
                        'twist': {
                            'linear':  {'x': float(vx), 'y': 0.0, 'z': 0.0},
                            'angular': {'x': 0.0, 'y': 0.0, 'z': float(omega)}
                        }
                    }
                    self.ros_publishers[i].publish(roslibpy.Message(msg))
                except Exception as e:
                    self.get_logger().error(f"Error publishing to robot {i+1}: {e}")

    def _publish_stop(self):
        """Send stop commands to robots 1 and 2 (robot 3 is controlled separately)."""
        for i in range(2):  # Only stop robots 1 and 2
            if i < len(self.ros_publishers) and self.ros_publishers[i] is not None:
                try:
                    msg = {
                        'header': {'stamp': ros_time(), 'frame_id': 'teleop_twist_joy'},
                        'twist': {
                            'linear':  {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        }
                    }
                    self.ros_publishers[i].publish(roslibpy.Message(msg))
                except Exception as e:
                    self.get_logger().error(f"Error sending stop to robot {i+1}: {e}")

    def _save_trajectory_csv(self):
        """Save trajectory data to CSV file."""
        if len(self.trajectory) == 0:
            self.get_logger().warn("No trajectory data to save.")
            return

        self.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.csv_output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['robot1_x', 'robot1_y', 'robot2_x', 'robot2_y', 'robot3_x', 'robot3_y'])
                
                # Write trajectory data
                for point in self.trajectory:
                    writer.writerow([
                        point[0][0], point[0][1],  # Robot 1
                        point[1][0], point[1][1],  # Robot 2
                        point[2][0], point[2][1]   # Robot 3
                    ])
            
            self.get_logger().info(f"Saved trajectory CSV to: {self.csv_output_path} ({len(self.trajectory)} points)")
        except Exception as e:
            self.get_logger().error(f"Failed to save trajectory CSV: {e}")

    def pursuit_evasion_loop(self):
        """Control loop for robots 1 & 2 using Julia solver - runs at 10 Hz."""
        # Wait for pose data for all robots (solver needs robot 3's pose too)
        if self.latest_odom_01 is None or self.latest_odom_02 is None or self.latest_odom_03 is None:
            self.get_logger().warn("Waiting for odometry (all robots needed for solver)...")
            return

        # Convert odometry to state
        state1 = self.convert_odom_to_state(self.latest_odom_01)
        state2 = self.convert_odom_to_state(self.latest_odom_02)
        state3 = self.convert_odom_to_state(self.latest_odom_03)
        
        # Record trajectory
        self.trajectory.append(((state1[0], state1[1]), (state2[0], state2[1]), (state3[0], state3[1])))
        
        # Check if robot 3 has reached the origin - if so, stop robots 1 and 2
        distance3_to_origin = math.sqrt((state3[0] - GOAL_POSITION[0]) ** 2 + (state3[1] - GOAL_POSITION[1]) ** 2)
        if distance3_to_origin < GOAL_THRESHOLD:
            # Robot 3 reached origin - stop robots 1 and 2
            self.current_commands[0] = (0.0, 0.0)
            self.current_commands[1] = (0.0, 0.0)
            if not self._shutdown_initiated:
                self.get_logger().info(
                    f"Robot 3 reached origin! Distance: {distance3_to_origin:.3f}m < {GOAL_THRESHOLD}m - Stopping robots 1 & 2")
                # Save trajectory CSV and shutdown
                self._save_trajectory_csv()
                self._shutdown_initiated = True
                # Stop timers
                self.pursuit_evasion_timer.cancel()
                self.publisher_timer.cancel()
                # Publish stop commands
                self._publish_stop()
                # Clean up roslibpy connections
                for i in range(2):
                    if i < len(self.ros_publishers) and self.ros_publishers[i] is not None:
                        try:
                            self.ros_publishers[i].unadvertise()
                        except:
                            pass
                for i in range(2):
                    if i < len(self.ros_clients) and self.ros_clients[i] is not None:
                        try:
                            self.ros_clients[i].terminate()
                        except:
                            pass
                self.destroy_node()
                rclpy.shutdown()
            return

        # Robot 1 & 2: Use Julia solver (pursuit-evasion) - always run the solver
        # Note: Solver needs robot 3's actual pose to solve correctly, but we ignore robot 3's output
        julia_state1 = Main.eval(f"[{state1[0]}; {state1[1]}]")
        julia_state2 = Main.eval(f"[{state2[0]}; {state2[1]}]") 
        julia_state3 = Main.eval(f"[{state3[0]}; {state3[1]}]")  # Solver needs actual robot 3 pose
        initial_state = [julia_state1, julia_state2, julia_state3]
        
        result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(
            self.pre, initial_state, self.z_guess, silence_logs=False)

        # Update z_guess for warm-starting next iteration
        self.z_guess = result.z_sol
        
        # Extract control commands [vx, vy] for robots 1 and 2 (ignore robot 3's output)
        u1 = result.u_curr[0]  # [vx1, vy1]
        u2 = result.u_curr[1]  # [vx2, vy2]
        next_states = result.x_next
        
        # Convert [vx, vy] to linear velocity and angular velocity for robots 1 and 2
        v1, omega1 = self.convert_to_cmd_vel(u1[0], u1[1], [state1[0], state1[1]], next_states[0], state1[2], robot_idx=0)
        v2, omega2 = self.convert_to_cmd_vel(u2[0], u2[1], [state2[0], state2[1]], next_states[1], state2[2], robot_idx=1)

        # Check distances between robots - stop robots 1 and 2 if too close to each other or robot 3
        distance_12 = math.sqrt((state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2)
        distance_13 = math.sqrt((state1[0] - state3[0]) ** 2 + (state1[1] - state3[1]) ** 2)
        distance_23 = math.sqrt((state2[0] - state3[0]) ** 2 + (state2[1] - state3[1]) ** 2)
        collision_threshold = 0.6  # meters
        
        # Check if robots 1 and 2 are too close to each other
        stop_robot1 = False
        stop_robot2 = False
        
        if distance_12 < collision_threshold:
            # Stop both robots if they're too close to each other
            stop_robot1 = True
            stop_robot2 = True
            self.get_logger().warn(
                f"Robots 1 & 2 too close! Distance: {distance_12:.3f}m < {collision_threshold}m - Stopping both robots")
        
        # Check if robot 1 is too close to robot 3
        if distance_13 < collision_threshold:
            stop_robot1 = True
            self.get_logger().warn(
                f"Robot 1 too close to Robot 3! Distance: {distance_13:.3f}m < {collision_threshold}m - Stopping robot 1")
        
        # Check if robot 2 is too close to robot 3
        if distance_23 < collision_threshold:
            stop_robot2 = True
            self.get_logger().warn(
                f"Robot 2 too close to Robot 3! Distance: {distance_23:.3f}m < {collision_threshold}m - Stopping robot 2")
        
        # Apply stops
        if stop_robot1:
            v1 = 0.0
            omega1 = 0.0
        if stop_robot2:
            v2 = 0.0
            omega2 = 0.0
        
        # Only clip and apply deadband if robots are not stopped
        if not stop_robot1:
            # Clip velocities for robot 1
            omega1 = np.clip(omega1, -0.5, 0.5)
            v1 = np.clip(v1, -0.5, 0.5)
            # Apply deadband for robot 1
            deadband_v = 0.05
            deadband_omega = 0.05
            if abs(v1) < deadband_v:
                v1 = 0.0
            if abs(omega1) < deadband_omega:
                omega1 = 0.0
        
        if not stop_robot2:
            # Clip velocities for robot 2
            omega2 = np.clip(omega2, -0.5, 0.5)
            v2 = np.clip(v2, -0.5, 0.5)
            # Apply deadband for robot 2
            deadband_v = 0.05
            deadband_omega = 0.05
            if abs(v2) < deadband_v:
                v2 = 0.0
            if abs(omega2) < deadband_omega:
                omega2 = 0.0

        # Clip commands to 0 if robot 3 hasn't moved 10cm from initial position
        if not self.robot3_moving:
            # Robot 3 hasn't moved yet - set robots 1 and 2 commands to 0
            v1 = 0.0
            omega1 = 0.0
            v2 = 0.0
            omega2 = 0.0
            self.get_logger().info("Robot 3 not moved 10cm from initial position - keeping robots 1 & 2 stopped")
        
        # Update commands for robots 1 and 2 (robot 3 is controlled separately)
        self.current_commands[0] = (v1, omega1)
        self.current_commands[1] = (v2, omega2)

        # Debug logging
        distance_12 = math.sqrt((state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2)
        distance_13 = math.sqrt((state1[0] - state3[0]) ** 2 + (state1[1] - state3[1]) ** 2)
        distance_23 = math.sqrt((state2[0] - state3[0]) ** 2 + (state2[1] - state3[1]) ** 2)
        self.get_logger().info(
            f"Pursuit-Evasion: P1=[{state1[0]:.3f}, {state1[1]:.3f}], P2=[{state2[0]:.3f}, {state2[1]:.3f}], "
            f"Dist12={distance_12:.3f}m, Dist13={distance_13:.3f}m, Dist23={distance_23:.3f}m")
        self.get_logger().info(
            f"Pursuit-Evasion: v1={v1:.3f}, ω1={omega1:.3f}, v2={v2:.3f}, ω2={omega2:.3f}")


    def destroy_node(self):
        """Clean up roslibpy connections before destroying node."""
        # Cancel timers
        if hasattr(self, 'pursuit_evasion_timer'):
            self.pursuit_evasion_timer.cancel()
        if hasattr(self, 'publisher_timer'):
            self.publisher_timer.cancel()
        
        # Only clean up connections for robots 1 and 2 (robot 3 is controlled separately)
        for i in range(2):
            if i < len(self.ros_publishers) and self.ros_publishers[i] is not None:
                try:
                    self.ros_publishers[i].unadvertise()
                except:
                    pass
        for i in range(2):
            if i < len(self.ros_clients) and self.ros_clients[i] is not None:
                try:
                    self.ros_clients[i].terminate()
                except:
                    pass
        super().destroy_node()


def main(pre, ros_clients, ros_publishers, args=None):
    rclpy.init(args=args)
    node = HybridController(pre, ros_clients, ros_publishers)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # Connect to robots FIRST (before Julia initialization)
    ros_clients, ros_publishers = connect_to_robots()
    
    # Then initialize Julia (takes a long time)
    pre = julia_init()
    
    # Start the controller with pre-established connections
    main(pre, ros_clients, ros_publishers)
