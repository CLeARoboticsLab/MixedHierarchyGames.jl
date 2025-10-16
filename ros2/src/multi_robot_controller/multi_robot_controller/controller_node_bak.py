import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class MultiRobotController(Node):
    def __init__(self):
        super().__init__('multi_robot_controller')

        # Publishers
        self.cmd_pub_01 = self.create_publisher(Twist, '/cmd_vel_01', 10)
        self.cmd_pub_02 = self.create_publisher(Twist, '/cmd_vel_02', 10)

        # Subscribers
        self.odom_sub_01 = self.create_subscription(
            Odometry, '/odom_01', self.odom_callback_01, 10)
        self.odom_sub_02 = self.create_subscription(
            Odometry, '/odom_02', self.odom_callback_02, 10)

        self.get_logger().info("MultiRobotController node has been started.")

    def odom_callback_01(self, msg: Odometry):
        twist = Twist()
        twist.linear.x = 0.1  # Example command
        twist.angular.z = 0.2
        self.cmd_pub_01.publish(twist)
        self.get_logger().info("Published to /cmd_vel_01")

    def odom_callback_02(self, msg: Odometry):
        twist = Twist()
        twist.linear.x = 0.1
        twist.angular.z = -0.2
        self.cmd_pub_02.publish(twist)
        self.get_logger().info("Published to /cmd_vel_02")


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
