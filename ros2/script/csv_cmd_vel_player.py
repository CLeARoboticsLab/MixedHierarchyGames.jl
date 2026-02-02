#!/usr/bin/env python3
"""
csv_cmd_vel_player.py - Play back control commands from CSV file to robots

This script connects to robots via roslibpy and sends velocity commands
from a CSV file to control the robots.
"""

import time
import csv
from pathlib import Path
import roslibpy

# Robot configuration
ROBOT_CONFIGS = [
    {'ip': '192.168.131.3', 'port': 9090, 'topic': '/bluebonnet/platform/cmd_vel'},
    {'ip': '192.168.131.4', 'port': 9090, 'topic': '/lonebot/platform/cmd_vel'},
    {'ip': '192.168.131.1', 'port': 9090, 'topic': '/hookem/platform/cmd_vel'},
]


def ros_time():
    """Generate ROS time stamp dictionary for TwistStamped messages."""
    t = time.time()
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    return {'sec': sec, 'nanosec': nanosec}


class CsvCmdVelPlayer:
    def __init__(self, csv_path: Path, dt: float = 0.1):
        self.dt = float(dt)
        self.csv_path = Path(csv_path).expanduser().resolve()
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Controls CSV not found: {self.csv_path}")
        
        self.controls = self._load_controls()
        self.index = 0
        self._shutdown = False

        self.clients = []
        self.publishers = []
        self._init_roslibpy_publishers()
        
        print(f"Loaded {len(self.controls)} control steps from {self.csv_path}")

    def _load_controls(self):
        """Load control commands from CSV file."""
        rows = []
        with open(self.csv_path, 'r') as f:
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
        """Initialize roslibpy connections to robots."""
        for i, config in enumerate(ROBOT_CONFIGS):
            try:
                print(f"Connecting to robot {i+1} at {config['ip']}:{config['port']}...")
                client = roslibpy.Ros(host=config['ip'], port=config['port'])
                client.run()
                
                if client.is_connected:
                    print(f"  ✓ Connected to robot {i+1}")
                    pub = roslibpy.Topic(client, config['topic'], 'geometry_msgs/msg/TwistStamped')
                    pub.advertise()
                    self.clients.append(client)
                    self.publishers.append(pub)
                else:
                    print(f"  ✗ Failed to connect to robot {i+1}")
                    self.clients.append(None)
                    self.publishers.append(None)
            except Exception as e:
                print(f"  ✗ Error connecting to robot {i+1}: {e}")
                self.clients.append(None)
                self.publishers.append(None)
        
        if len(self.publishers) != len(ROBOT_CONFIGS):
            print(f"Warning: Only {len([p for p in self.publishers if p is not None])}/{len(ROBOT_CONFIGS)} robots connected")

    def _publish_stop_all(self):
        """Send stop commands to all robots."""
        for pub in self.publishers:
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
                print(f"Failed to publish stop: {e}")

    def run(self):
        """Run the control loop."""
        print(f"\n=== Starting CSV command playback ===")
        print(f"Playing {len(self.controls)} control steps at {1.0/self.dt:.1f} Hz")
        
        # Publish rate - send commands multiple times per CSV row to maintain continuous motion
        publish_rate = 1000.0  # Hz - publish 50 times per second (higher rate for smoother motion)
        publish_interval = 1.0 / publish_rate  # 0.02 seconds
        
        try:
            while not self._shutdown and self.index < len(self.controls):
                c = self.controls[self.index]
                v1, w1, v2, w2, v3, w3 = c
                # Multiply robot 3 velocities by 3
                v1 = v1 * 1.5
                w1 = w1 * 1.3
                v3 = v3 * 1.25
                w3 = w3 * 1.25
                cmds = [(v1, w1), (v2, w2), (v3, w3)]

                # Publish commands continuously for the duration of this CSV row
                # This prevents the robot from stopping between commands
                t_end = time.time() + self.dt
                while time.time() < t_end:
                    for i, (v, w) in enumerate(cmds):
                        pub = self.publishers[i] if i < len(self.publishers) else None
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
                            print(f"Failed to publish to robot {i+1}: {e}")
                    
                    time.sleep(publish_interval)

                self.index += 1
                
                # Progress indicator
                if self.index % 100 == 0:
                    progress = (self.index / len(self.controls)) * 100
                    print(f"Progress: {self.index}/{len(self.controls)} ({progress:.1f}%)")
            
            print("\n=== Playback complete ===")
            self._publish_stop_all()
        except KeyboardInterrupt:
            print("\n=== Interrupted by user ===")
            self._publish_stop_all()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up connections - using safe pattern from test2robot.py"""
        for pub in self.publishers:
            if pub is not None:
                try:
                    pub.unadvertise()
                except Exception:
                    pass
        for client in self.clients:
            if client is not None and client.is_connected:
                # Check internal structure to avoid AttributeError in terminate()
                # The terminate() method checks for _thread attribute which may not exist
                factory = getattr(client, 'factory', None)
                if factory is not None:
                    manager = getattr(factory, 'manager', None)
                    if manager is not None:
                        # Only terminate if manager has _thread attribute
                        # This prevents AttributeError when terminate() tries to access it
                        thread = getattr(manager, '_thread', None)
                        if thread is not None:
                            try:
                                client.terminate()
                            except Exception:
                                pass
        print("Disconnected from all robots")


def main():
    """Main entry point."""
    import sys
    
    # Try to locate the CSV file
    script_dir = Path(__file__).resolve().parent
    ros2_root = script_dir.parent
    candidates = [
        ros2_root / "receding_horizon_controls.csv",
        Path.cwd() / "receding_horizon_controls.csv",
    ]
    
    # Check command line argument
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        # Pick the first existing candidate
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            csv_path = candidates[0]  # Default to first candidate
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print(f"Please provide a CSV file path as argument or place 'receding_horizon_controls.csv' in the current directory.")
        return 1
    
    print(f"Using CSV file: {csv_path}")
    
    player = CsvCmdVelPlayer(csv_path=csv_path, dt=0.1)
    player.run()
    return 0


if __name__ == '__main__':
    exit(main())

