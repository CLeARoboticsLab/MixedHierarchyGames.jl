#!/usr/bin/env python3
"""
test2robot.py - Control two robots simultaneously with basic cmd_vel commands

This script connects to two robots via roslibpy and sends velocity commands
to both robots at the same time.
"""

import time
import roslibpy

# Robot configurations
ROBOT_CONFIGS = [
    {'name': 'Husky', 'ip': '192.168.131.1', 'port': 9090, 'topic': '/skoll/platform/cmd_vel'},
    {'name': 'Bluebonnet', 'ip': '192.168.131.4', 'port': 9090, 'topic': '/lonebot/platform/cmd_vel'},
]

def ros_time():
    """Generate ROS time stamp"""
    t = time.time()
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    return {'sec': sec, 'nanosec': nanosec}

def create_twist_stamped(vx, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0):
    """Create a TwistStamped message"""
    return {
        'header': {'stamp': ros_time(), 'frame_id': 'teleop_twist_joy'},
        'twist': {
            'linear': {'x': vx, 'y': vy, 'z': vz},
            'angular': {'x': wx, 'y': wy, 'z': wz}
        }
    }

class TwoRobotController:
    def __init__(self):
        self.clients = []
        self.publishers = []
        
        # Connect to both robots
        for config in ROBOT_CONFIGS:
            print(f"Connecting to {config['name']} at {config['ip']}:{config['port']}...")
            client = roslibpy.Ros(host=config['ip'], port=config['port'])
            client.run()
            
            if client.is_connected:
                print(f"  ✓ Connected to {config['name']}")
            else:
                print(f"  ✗ Failed to connect to {config['name']}")
                continue
                
            pub = roslibpy.Topic(client, config['topic'], 'geometry_msgs/msg/TwistStamped')
            pub.advertise()
            
            self.clients.append(client)
            self.publishers.append(pub)
        
        if len(self.publishers) != len(ROBOT_CONFIGS):
            print(f"Warning: Only {len(self.publishers)}/{len(ROBOT_CONFIGS)} robots connected")
    
    def send_to_all(self, vx, duration=0.1):
        """
        Send the same forward command to all robots
        
        Args:
            vx: Linear velocity in x direction (m/s)
            duration: How long to send the command (seconds)
        """
        t_end = time.time() + duration
        while time.time() < t_end:
            msg = create_twist_stamped(vx, wz=0.0)
            for pub in self.publishers:
                pub.publish(roslibpy.Message(msg))
            time.sleep(0.05)  # 20 Hz
    
    def send_individual(self, robot_idx, vx, duration=0.1):
        """
        Send forward command to a specific robot
        
        Args:
            robot_idx: Index of robot (0=Husky, 1=Bluebonnet)
            vx: Linear velocity in x direction (m/s)
            duration: How long to send the command (seconds)
        """
        if robot_idx >= len(self.publishers):
            print(f"Error: Robot index {robot_idx} out of range")
            return
        
        t_end = time.time() + duration
        while time.time() < t_end:
            msg = create_twist_stamped(vx, wz=0.0)
            self.publishers[robot_idx].publish(roslibpy.Message(msg))
            time.sleep(0.05)  # 20 Hz
    
    def stop_all(self, duration=0.5):
        """Stop all robots"""
        self.send_to_all(0.0, duration)
    
    def cleanup(self):
        """Clean up connections"""
        for pub in self.publishers:
            if pub is not None:
                pub.unadvertise()
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
                            client.terminate()
        print("Disconnected from all robots")

def main():
    """Example usage"""
    controller = TwoRobotController()
    
    if len(controller.publishers) < 2:
        print("Error: Need at least 2 robots connected")
        controller.cleanup()
        return
    
    print("\n=== Starting forward movement test ===")
    
    # Test 1: Both robots move forward
    print("\n1. Both robots forward (0.3 m/s) for 3 seconds...")
    for _ in range(30):
        controller.send_to_all(0.3, duration=0.1)
    
    # Stop
    print("   Stopping...")
    controller.stop_all()
    time.sleep(1)
    
    # Test 2: Both robots move forward at different speed
    print("\n2. Both robots forward (0.2 m/s) for 2 seconds...")
    for _ in range(20):
        controller.send_to_all(0.2, duration=0.1)
    
    # Stop
    print("   Stopping...")
    controller.stop_all()
    time.sleep(1)
    
    # Test 3: Individual control - both forward
    print("\n3. Husky forward (0.2 m/s), Bluebonnet forward (0.2 m/s) for 2 seconds...")
    for _ in range(20):
        controller.send_individual(0, 0.2, duration=0.1)  # Husky forward
        controller.send_individual(1, 0.2, duration=0.1)  # Bluebonnet forward
    
    # Stop
    print("   Stopping...")
    controller.stop_all()
    
    print("\n=== Test sequence complete ===")
    
    controller.cleanup()

if __name__ == '__main__':
    main()

