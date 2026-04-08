import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math
import time
import roslibpy

# Robot configuration for Husky
ROBOT_IP = '192.168.50.49'
ROBOT_PORT = 9090
ROBOT_TOPIC = '/hookem/platform/cmd_vel'
POSE_TOPIC = '/vrpn_client_node/Husky/pose'

# Goal position (origin)
GOAL_POSITION = [0.0, 0.0]
GOAL_THRESHOLD = 0.6  # meters - stop when within this distance

# Controller gains
K_LINEAR = 0.5   # Proportional gain for linear velocity
K_ANGULAR = 1.0  # Proportional gain for angular velocity
MAX_LINEAR_VEL = 0.6   # m/s
MAX_ANGULAR_VEL = 0.6  # rad/s


def ros_time():
    """Generate ROS time stamp dictionary for TwistStamped messages."""
    t = time.time()
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    return {'sec': sec, 'nanosec': nanosec}


class GoToOriginController(Node):
    def __init__(self):
        super().__init__('go_to_origin_controller')

        # Subscribe to pose from vrpn_client_ros2
        self.pose_sub = self.create_subscription(
            PoseStamped, POSE_TOPIC, self.pose_callback, 10)

        # Current pose
        self.current_pose = None
        self.current_x = None
        self.current_y = None
        self.current_theta = None

        # Setup roslibpy connection to robot
        self.ros_client = None
        self.ros_publisher = None
        self._setup_robot_connection()

        # Current commands to publish (updated by controller, published at high rate)
        self.current_vx = 0.0
        self.current_omega = 0.0
        
        # Previous commands for low-pass filtering
        self.prev_vx = 0.0
        self.prev_omega = 0.0

        # Timer to run controller at 10 Hz (computes new commands)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # High-rate publisher timer at 100 Hz (0.01s) for smoother movement
        # This publishes the current commands repeatedly during each 0.1s controller step
        self.publisher_timer = self.create_timer(0.01, self._publish_current_command)

        self.get_logger().info("GoToOriginController node started.")
        self.get_logger().info(f"Goal: {GOAL_POSITION}, Threshold: {GOAL_THRESHOLD}m")

    def _setup_robot_connection(self):
        """Setup roslibpy connection to the robot."""
        try:
            self.ros_client = roslibpy.Ros(host=ROBOT_IP, port=ROBOT_PORT)
            self.ros_client.run()

            # Wait for connection to establish (with timeout)
            max_wait_time = 5.0  # seconds
            wait_interval = 0.1  # seconds
            waited = 0.0
            while not self.ros_client.is_connected and waited < max_wait_time:
                time.sleep(wait_interval)
                waited += wait_interval

            if self.ros_client.is_connected:
                self.get_logger().info(f"Connected to robot at {ROBOT_IP}:{ROBOT_PORT}")
                self.ros_publisher = roslibpy.Topic(
                    self.ros_client, ROBOT_TOPIC, 'geometry_msgs/msg/TwistStamped')
                self.ros_publisher.advertise()
            else:
                self.get_logger().error(
                    f"Failed to connect to robot at {ROBOT_IP}:{ROBOT_PORT} (timeout after {max_wait_time}s)")
                self.ros_client = None
                self.ros_publisher = None
        except Exception as e:
            self.get_logger().error(f"Error connecting to robot: {e}")
            self.ros_client = None
            self.ros_publisher = None

    def pose_callback(self, msg):
        """Callback for pose updates."""
        self.current_pose = msg
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y

        # Convert quaternion to yaw (theta)
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_theta = math.atan2(siny_cosp, cosy_cosp)

    def _distance_to_goal(self):
        """Calculate distance to goal."""
        if self.current_x is None or self.current_y is None:
            return None
        return math.sqrt(
            (self.current_x - GOAL_POSITION[0]) ** 2 +
            (self.current_y - GOAL_POSITION[1]) ** 2
        )

    def _angle_to_goal(self):
        """Calculate angle from current position to goal."""
        if self.current_x is None or self.current_y is None:
            return None
        return math.atan2(
            GOAL_POSITION[1] - self.current_y,
            GOAL_POSITION[0] - self.current_x
        )

    def _send_command(self, vx, omega):
        """Update current commands (will be published at high rate by publisher_timer)."""
        # Clip velocities to limits
        vx = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, vx))
        omega = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, omega))
        
        # Store commands for high-rate publishing
        self.current_vx = vx
        self.current_omega = omega
    
    def _publish_current_command(self):
        """High-rate publisher that sends current commands repeatedly for smoother movement.
        Runs at 100 Hz (every 0.01s) to publish commands multiple times during each 0.1s controller step."""
        if self.ros_publisher is None:
            return
        
        try:
            msg = {
                'header': {'stamp': ros_time(), 'frame_id': 'teleop_twist_joy'},
                'twist': {
                    'linear': {'x': float(self.current_vx), 'y': 0.0, 'z': 0.0},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': float(self.current_omega)}
                }
            }
            self.ros_publisher.publish(roslibpy.Message(msg))
        except Exception as e:
            self.get_logger().error(f"Error publishing command: {e}")

    def _send_stop(self):
        """Send stop command to robot."""
        self.current_vx = 0.0
        self.current_omega = 0.0
        self.prev_vx = 0.0
        self.prev_omega = 0.0

    def control_loop(self):
        """Main control loop - runs at 10 Hz."""
        # Wait for pose data
        if self.current_x is None or self.current_y is None or self.current_theta is None:
            self.get_logger().warn("Waiting for pose data...")
            return

        # Check if we've reached the goal
        distance = self._distance_to_goal()
        if distance is None:
            return

        if distance < GOAL_THRESHOLD:
            self.get_logger().info(
                f"Goal reached! Distance: {distance:.3f}m < {GOAL_THRESHOLD}m")
            # Stop timers and send stop command
            self.timer.cancel()
            self.publisher_timer.cancel()
            self.current_vx = 0.0
            self.current_omega = 0.0
            self._send_stop()
            # Publish stop a few more times to ensure robot stops
            for _ in range(5):
                self._publish_current_command()
                time.sleep(0.01)
            self._cleanup()
            self.destroy_node()
            rclpy.shutdown()
            return

        # Calculate angle to goal
        angle_to_goal = self._angle_to_goal()
        if angle_to_goal is None:
            return

        # Calculate angular error (normalize to [-pi, pi])
        angle_error = angle_to_goal - self.current_theta
        # Normalize angle to [-pi, pi]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Proportional controller
        # Linear velocity: proportional to distance, but reduce when angle error is large
        vx_cmd = K_LINEAR * distance * math.cos(angle_error)
        # Angular velocity: proportional to angle error
        omega_cmd = K_ANGULAR * angle_error
        
        # Apply low-pass filter to smooth out rapid changes and prevent stuck behavior
        # This helps prevent oscillation and makes motion smoother
        alpha_v = 0.7  # Filter coefficient for linear velocity: 70% new, 30% old
        alpha_omega = 0.6  # Filter coefficient for angular velocity: 60% new, 40% old (more smoothing)
        
        vx = alpha_v * vx_cmd + (1.0 - alpha_v) * self.prev_vx
        omega = alpha_omega * omega_cmd + (1.0 - alpha_omega) * self.prev_omega
        
        # Update stored values for next iteration
        self.prev_vx = vx
        self.prev_omega = omega
        
        # Apply deadband to prevent tiny oscillations that cause stuck behavior
        deadband_v = 0.05  # 5 cm/s threshold
        deadband_omega = 0.05  # 0.05 rad/s threshold
        if abs(vx) < deadband_v:
            vx = 0.0
        if abs(omega) < deadband_omega:
            omega = 0.0

        # Log status
        self.get_logger().info(
            f"Pos: [{self.current_x:.3f}, {self.current_y:.3f}], "
            f"Theta: {self.current_theta:.3f}, "
            f"Dist: {distance:.3f}m, "
            f"AngleErr: {math.degrees(angle_error):.1f}°, "
            f"Cmd: vx={vx:.3f}, omega={omega:.3f}")

        # Update commands (will be published at high rate by publisher_timer)
        self._send_command(vx, omega)

    def _cleanup(self):
        """Clean up roslibpy connections."""
        if self.ros_publisher is not None:
            try:
                self.ros_publisher.unadvertise()
            except:
                pass
        if self.ros_client is not None:
            try:
                self.ros_client.terminate()
            except:
                pass

    def destroy_node(self):
        """Clean up before destroying node."""
        # Cancel timers
        if hasattr(self, 'timer'):
            self.timer.cancel()
        if hasattr(self, 'publisher_timer'):
            self.publisher_timer.cancel()
        self._cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GoToOriginController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
