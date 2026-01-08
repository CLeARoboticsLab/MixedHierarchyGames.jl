import rclpy
from rclpy.node import Node
from pathlib import Path
import csv
import time
import roslibpy


# Robot configuration
ROBOT_CONFIGS = [
    {'ip': '192.168.131.3', 'port': 9090, 'topic': '/bluebonnet/platform/cmd_vel'},
    {'ip': '192.168.131.4', 'port': 9090, 'topic': '/lonebot/platform/cmd_vel'},
    {'ip': '192.168.131.5', 'port': 9090, 'topic': '/skoll/platform/cmd_vel'},
]


def ros_time():
    t = time.time()
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    return {'sec': sec, 'nanosec': nanosec}


class CsvCmdVelPlayer(Node):
    def __init__(self, csv_path: Path, dt: float = 0.1):
        super().__init__('csv_cmd_vel_player')
        self.dt = float(dt)
        # Allow overriding csv_path via ROS parameter
        self.declare_parameter('csv_path', str(csv_path))
        csv_param = self.get_parameter('csv_path').get_parameter_value().string_value
        resolved_csv = Path(csv_param).expanduser().resolve()
        self.controls = self._load_controls(resolved_csv)
        self.index = 0
        self._shutdown = False

        self.ros_clients = []
        self.ros_publishers = []
        self._init_roslibpy_publishers()

        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(f"Loaded {len(self.controls)} control steps from {resolved_csv}")

    def _load_controls(self, csv_path: Path):
        rows = []
        if not csv_path.exists():
            raise FileNotFoundError(f"Controls CSV not found: {csv_path}")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            required = [
                'robot1_v', 'robot1_omega',
                'robot2_v', 'robot2_omega',
                'robot3_v', 'robot3_omega',
            ]
            for r in reader:
                row = []
                for key in required:
                    row.append(float(r[key]))
                rows.append(row)
        return rows

    def _init_roslibpy_publishers(self):
        for i, config in enumerate(ROBOT_CONFIGS):
            try:
                client = roslibpy.Ros(host=config['ip'], port=config['port'])
                client.run()
                if client.is_connected:
                    pub = roslibpy.Topic(client, config['topic'], 'geometry_msgs/msg/TwistStamped')
                    pub.advertise()
                    self.ros_clients.append(client)
                    self.ros_publishers.append(pub)
                    self.get_logger().info(f"Connected to robot {i+1} at {config['ip']}:{config['port']} → {config['topic']}")
                else:
                    self.get_logger().warn(f"Failed to connect to robot {i+1} at {config['ip']}:{config['port']}")
                    self.ros_clients.append(None)
                    self.ros_publishers.append(None)
            except Exception as e:
                self.get_logger().error(f"Error connecting to robot {i+1}: {e}")
                self.ros_clients.append(None)
                self.ros_publishers.append(None)

    def _publish_stop_all(self):
        for pub in self.ros_publishers:
            if pub is None:
                continue
            try:
                msg = {
                    'header': {'stamp': ros_time(), 'frame_id': 'csv_cmd_vel_player'},
                    'twist': {
                        'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                        'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    }
                }
                pub.publish(roslibpy.Message(msg))
            except Exception as e:
                self.get_logger().error(f"Failed to publish stop: {e}")

    def _tick(self):
        if self._shutdown:
            return
        if self.index >= len(self.controls):
            self.get_logger().info("All control steps published. Stopping.")
            self._publish_stop_all()
            self._shutdown = True
            self.timer.cancel()
            self._cleanup()
            self.destroy_node()
            rclpy.shutdown()
            return

        c = self.controls[self.index]
        v1, w1, v2, w2, v3, w3 = c
        cmds = [(v1, w1), (v2, w2), (v3, w3)]

        for i, (v, w) in enumerate(cmds):
            pub = self.ros_publishers[i] if i < len(self.ros_publishers) else None
            if pub is None:
                continue
            try:
                msg = {
                    'header': {'stamp': ros_time(), 'frame_id': 'csv_cmd_vel_player'},
                    'twist': {
                        'linear': {'x': float(v), 'y': 0.0, 'z': 0.0},
                        'angular': {'x': 0.0, 'y': 0.0, 'z': float(w)}
                    }
                }
                pub.publish(roslibpy.Message(msg))
            except Exception as e:
                self.get_logger().error(f"Failed to publish to robot {i+1}: {e}")

        self.index += 1

    def _cleanup(self):
        for pub in self.ros_publishers:
            if pub is not None:
                try:
                    pub.unadvertise()
                except Exception:
                    pass
        for client in self.ros_clients:
            if client is not None:
                try:
                    client.terminate()
                except Exception:
                    pass


def main(args=None):
    rclpy.init(args=args)
    # Try to locate the workspace 'ros2' root above the installed package
    here = Path(__file__).resolve()
    ros2_root = None
    for p in here.parents:
        if p.name == 'ros2':
            ros2_root = p
            break
    # Default search order for the CSV
    candidates = []
    if ros2_root is not None:
        candidates.append(ros2_root / "receding_horizon_controls.csv")
    candidates.append(Path.cwd() / "receding_horizon_controls.csv")
    # Pick the first existing candidate, else fall back to ros2_root path
    default_csv = None
    for c in candidates:
        if c.exists():
            default_csv = c
            break
    if default_csv is None:
        default_csv = candidates[0]
    csv_path = default_csv
    node = CsvCmdVelPlayer(csv_path=csv_path, dt=0.1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._publish_stop_all()
    finally:
        node._cleanup()
        node.destroy_node()
        rclpy.shutdown()


