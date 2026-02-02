import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math
import time
import roslibpy

# Robot configuration for lonebot
ROBOT_IP = '192.168.50.2'
ROBOT_PORT = 9090
ROBOT_TOPIC = '/lonebot/platform/cmd_vel'
POSE_TOPIC = '/vrpn_client_node/Lonebot/pose'

# Goal position (origin)
GOAL_POSITION = [0.0, 0.0]
GOAL_THRESHOLD = 0.3  # meters - stop when within this distance

# Controller gains
K_LINEAR = 0.5   # Proportional gain for linear velocity
K_ANGULAR = 1.0  # Proportional gain for angular velocity
MAX_LINEAR_VEL = 0.5   # m/s
MAX_ANGULAR_VEL = 0.5  # rad/s


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

        # Timer to run controller at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

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
        """Send velocity command to robot via roslibpy."""
        if self.ros_publisher is None:
            self.get_logger().warn("No publisher available, cannot send command")
            return

        try:
            # Clip velocities to limits
            vx = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, vx))
            omega = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, omega))

            msg = {
                'header': {'stamp': ros_time(), 'frame_id': 'teleop_twist_joy'},
                'twist': {
                    'linear': {'x': float(vx), 'y': 0.0, 'z': 0.0},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': float(omega)}
                }
            }
            self.ros_publisher.publish(roslibpy.Message(msg))
        except Exception as e:
            self.get_logger().error(f"Error sending command: {e}")

    def _send_stop(self):
        """Send stop command to robot."""
        self._send_command(0.0, 0.0)

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
            self._send_stop()
            self.timer.cancel()
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
        vx = K_LINEAR * distance * math.cos(angle_error)
        # Angular velocity: proportional to angle error
        omega = K_ANGULAR * angle_error

        # Log status
        self.get_logger().info(
            f"Pos: [{self.current_x:.3f}, {self.current_y:.3f}], "
            f"Theta: {self.current_theta:.3f}, "
            f"Dist: {distance:.3f}m, "
            f"AngleErr: {math.degrees(angle_error):.1f}°, "
            f"Cmd: vx={vx:.3f}, omega={omega:.3f}")

        # Send command
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
