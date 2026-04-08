#!/usr/bin/env python3
"""
test_jetracer.py — Sanity-check roslibpy → ROS 1 rosbridge → JetRacer steering/throttle.

Matches `multi_jetracer_controller.py`: `std_msgs/Float64` on /jetracer/steering and
/jetracer/throttle (ROS 1 type string `std_msgs/Float64`).

Prereqs on the JetRacer: roscore + rosbridge (`roslaunch rosbridge_server rosbridge_websocket.launch`).

Usage:
  python3 test_jetracer.py

Edit ROBOT_IP / topics below if needed.
"""

import time

import roslibpy

# --- Edit for your JetRacer ---
ROBOT_IP = "192.168.50.32"
PORT = 9090
STEERING_TOPIC = "/jetracer/steering"
THROTTLE_TOPIC = "/jetracer/throttle"
# ROS 1 rosbridge uses std_msgs/Float64; use std_msgs/msg/Float64 only if you bridge as ROS 2
FLOAT_MSG_TYPE = "std_msgs/Float64"


def float64(data: float) -> dict:
    return {"data": float(data)}


def send_steering_throttle(steering_pub, throttle_pub, steering: float, throttle: float, duration: float):
    """Publish steering and throttle at ~100 Hz for duration seconds."""
    t_end = time.time() + duration
    while time.time() < t_end:
        steering_pub.publish(roslibpy.Message(float64(steering)))
        throttle_pub.publish(roslibpy.Message(float64(throttle)))
        time.sleep(0.01)


def main():
    client = roslibpy.Ros(host=ROBOT_IP, port=PORT)
    client.run()
    print("Connected:", client.is_connected)

    steering_pub = roslibpy.Topic(client, STEERING_TOPIC, FLOAT_MSG_TYPE)
    throttle_pub = roslibpy.Topic(client, THROTTLE_TOPIC, FLOAT_MSG_TYPE)
    steering_pub.advertise()
    throttle_pub.advertise()

    try:
        # Hold neutral briefly so subscribers see advertisers
        time.sleep(0.2)
        print("Neutral (0, 0) …")
        send_steering_throttle(steering_pub, throttle_pub, 0.0, 0.0, duration=0.5)

        # Small forward throttle, straight steering — tune magnitude for your setup
        print("Small forward throttle (steering=0, throttle=0.15) …")
        send_steering_throttle(steering_pub, throttle_pub, 0.0, 0.15, duration=2.0)

        print("Stop …")
        send_steering_throttle(steering_pub, throttle_pub, 0.0, 0.0, duration=0.5)

        # Optional: gentle steering check (car still, wheels turn) — comment out if undesired
        print("Gentle steering wiggle (throttle=0) …")
        send_steering_throttle(steering_pub, throttle_pub, 0.2, 0.0, duration=0.5)
        send_steering_throttle(steering_pub, throttle_pub, -0.2, 0.0, duration=0.5)
        send_steering_throttle(steering_pub, throttle_pub, 0.0, 0.0, duration=0.5)

    finally:
        send_steering_throttle(steering_pub, throttle_pub, 0.0, 0.0, duration=0.3)
        steering_pub.unadvertise()
        throttle_pub.unadvertise()
        client.terminate()
        print("Done")


if __name__ == "__main__":
    main()
