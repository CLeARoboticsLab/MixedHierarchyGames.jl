# send_cmdvel_stamped.py
import time
import roslibpy

# ROBOT_IP = '192.168.131.3' #wired connection
ROBOT_IP = '192.168.131.4'
PORT = 9090
# TOPIC = '/bluebonnet/platform/cmd_vel'  # expects geometry_msgs/TwistStamped
TOPIC = '/lonebot/platform/cmd_vel'

client = roslibpy.Ros(host=ROBOT_IP, port=PORT)
client.run()
print('Connected:', client.is_connected)

pub = roslibpy.Topic(client, TOPIC, 'geometry_msgs/msg/TwistStamped')
pub.advertise()                 # <-- IMPORTANT

def ros_time():
    t = time.time()
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    return {'sec': sec, 'nanosec': nanosec}

def send(vx, wz, duration=3.0):
    t_end = time.time() + duration
    while time.time() < t_end:
        msg = {
            'header': {'stamp': ros_time(), 'frame_id': 'teleop_twist_joy'},
            'twist': {
                'linear':  {'x': vx, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': wz}
            }
        }
        pub.publish(roslibpy.Message(msg))
        time.sleep(0.05)  # 20 Hz

# drive forward, then stop
for t in range(30):
    send(-0.5, 0.0, duration=0.1)
send(0.0, 0.0, duration=0.5)

pub.unadvertise()
client.terminate()
print('Done')
