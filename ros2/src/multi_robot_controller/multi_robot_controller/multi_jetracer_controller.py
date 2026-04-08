"""
Multi JetRacer + Jackal pursuit-evasion controller (laptop + Vicon + rosbridge).

Architecture (typical setup)
----------------------------
- **Laptop (ROS 2):** Runs this node, `vrpn_client_ros2` (or similar) for Vicon, and Julia.
  Subscribes to `PoseStamped` topics such as `/vrpn_client_node/<object_name>/pose`.
- **Agents 1–2:** `TwistStamped` via rosbridge (`convert_to_cmd_vel` like differential drive).
- **Agent 3 (JetRacer, ROS 1):** `std_msgs/Float64` on `/jetracer/steering` and `/jetracer/throttle`.
  PID tracks solver `(vx, vy)` as speed reference and MPC **next state** as lookahead for heading.

Vicon names must match Motive / VRPN. **Solver player order** (must match `initial_state`):
  - **Agent 1:** lonebot (Jackal) — pursuer
  - **Agent 2:** Bluebonnet (or second diff-drive)
  - **Agent 3:** JetRacer — evader → origin; run stops when agent 3 reaches `GOAL_POSITION`.

`VRPN_POSE_TOPICS` and `ROBOT_CONFIGS` use that same order (index 0 = agent 1, …).

The Julia stack (`HardwareFunctions.hardware_nplayer_hierarchy_navigation`) is unchanged from
`pursuit_evasion_controller.py`: same preoptimization and MPC solve.
"""

import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from julia.api import Julia
import roslibpy
from roslibpy.core import RosTimeoutError

# Set environment variable BEFORE Julia is initialized
os.environ["GKSwstype"] = "nul"
os.environ["JULIA_SSL_LIBRARY"] = "system"
jl = Julia(compiled_modules=False)
from julia import Main

GOAL_POSITION = [0.0, 0.0]

# Vicon / VRPN: same order as Julia agents — (1) lonebot, (2) og_jetracer2, (3) og_jetracer1
VRPN_POSE_TOPICS = [
    "/vrpn_client_node/Lonebot/pose",
    "/vrpn_client_node/BlueBonnet/pose",
    "/vrpn_client_node/jetracer1/pose",
]

# Robot command interfaces via roslibpy (same index order as agents above)
# msg_type:
#   - 'jetracer'       -> ROS1: std_msgs/Float64 on steering_topic + throttle_topic
#   - 'twist'          -> ROS 1: geometry_msgs/Twist (single topic, e.g. /cmd_vel)
#   - 'twist_stamped'  -> ROS 2: geometry_msgs/msg/TwistStamped
#
# JetRacer: solver planar controls (vx, vy) in world frame -> Float64 steering / throttle (see limits below).
ROBOT_CONFIGS = [
    {
        # agent 1 — lonebot (pursuer)
        "ip": "192.168.50.2",
        "port": 9090,
        "topic": "/lonebot/platform/cmd_vel",
        "msg_type": "twist_stamped",
    },
    {
        # agent 2 — Bluebonnet
        "ip": "192.168.50.25",
        "port": 9090,
        "topic": "/Bluebonnet/platform/cmd_vel",
        "msg_type": "twist_stamped",
    },
    {
        # agent 3 — jetracer1 (evader → origin)
        "ip": "192.168.50.32",
        "port": 9090,
        "msg_type": "jetracer",
        "steering_topic": "/jetracer/steering",
        "throttle_topic": "/jetracer/throttle",
        "float_bridge": "ros1",
    },

]

_MSG_TYPE_TO_ROSLIBPY_TYPE = {
    "twist": "geometry_msgs/Twist",
    "twist_stamped": "geometry_msgs/msg/TwistStamped",
}

_FLOAT_MSG_TYPE = {
    "ros1": "std_msgs/Float64",
    "ros2": "std_msgs/msg/Float64",
}

# Planner period (must match create_timer for run_planner_step)
PLANNER_DT = 0.1

# JetRacer PID (agent 3): track speed ref ||(vx,vy)|| and heading toward next MPC state
JETRACER_MAX_THROTTLE = 0.15  # throttle Float64 is >= 0 only (no reverse / brake on this topic)
# Steering: heading PID -> saturate -> one low-pass (symmetric +/−, no stacked filters).
JETRACER_MAX_STEERING = 0.055
JETRACER_KP_HEAD = 0.38
JETRACER_KI_HEAD = 0.03
JETRACER_KD_HEAD = 0.02
JETRACER_HEADING_I_CLAMP = 0.2
# steering_filt <- (1-a)*steering_filt + a * clipped_PID  each planner step
JETRACER_STEERING_SMOOTH_ALPHA = 0.28

JETRACER_KFF_THROTTLE = 0.35
JETRACER_KP_SPEED = 0.45
JETRACER_KI_SPEED = 0.12
JETRACER_SPEED_I_CLAMP = 0.25
# throttle_filt <- (1-b)*throttle_filt + b * clipped_throttle_cmd
JETRACER_THROTTLE_SMOOTH_ALPHA = 0.35

JETRACER_THROTTLE_DEADBAND = 0.015
JETRACER_STEERING_DEADBAND = 0.003
# If True: always publish steering 0 (throttle-only); heading PID skipped to avoid windup.
JETRACER_STEERING_DISABLED = True


def _float_msg_type_for_bridge(config):
    b = config.get("float_bridge", "ros1")
    if b not in _FLOAT_MSG_TYPE:
        raise ValueError(f"Unknown float_bridge {b!r}; use 'ros1' or 'ros2'")
    return _FLOAT_MSG_TYPE[b]


def _topic_type_for_config(config):
    mt = config.get("msg_type", "twist_stamped")
    if mt == "jetracer":
        raise ValueError("jetracer uses steering/throttle topics; do not call _topic_type_for_config")
    if mt not in _MSG_TYPE_TO_ROSLIBPY_TYPE:
        raise ValueError(
            f"Unknown msg_type {mt!r}; use one of {list(_MSG_TYPE_TO_ROSLIBPY_TYPE) + ['jetracer']}"
        )
    return _MSG_TYPE_TO_ROSLIBPY_TYPE[mt]


def _make_publisher_entry(client, config):
    """Build publisher handle for one robot: dict with 'kind' and topic publisher(s)."""
    mt = config.get("msg_type", "twist_stamped")
    if mt == "jetracer":
        ft = _float_msg_type_for_bridge(config)
        st = config.get("steering_topic", "/jetracer/steering")
        tt = config.get("throttle_topic", "/jetracer/throttle")
        steering_pub = roslibpy.Topic(client, st, ft)
        throttle_pub = roslibpy.Topic(client, tt, ft)
        steering_pub.advertise()
        throttle_pub.advertise()
        return {"kind": "jetracer", "steering": steering_pub, "throttle": throttle_pub}
    topic_type = _topic_type_for_config(config)
    topic = config["topic"]
    pub = roslibpy.Topic(client, topic, topic_type)
    pub.advertise()
    return {"kind": mt, "pub": pub}


def connect_to_robots():
    """Connect to robots via roslibpy BEFORE Julia initialization."""
    print("Connecting to robots...")
    ros_clients = []
    ros_publishers = []

    for i, config in enumerate(ROBOT_CONFIGS):
        try:
            print(f"Creating client for robot {i + 1} at {config['ip']}:{config['port']}...")
            client = roslibpy.Ros(host=config["ip"], port=config["port"])
            ros_clients.append(client)
        except Exception as e:
            print(f"✗ Error creating client for robot {i + 1}: {e}")
            ros_clients.append(None)

    reactor_started = False
    for i, client in enumerate(ros_clients):
        if client is not None:
            try:
                print(f"Attempting to start reactor with robot {i + 1} connection...")
                client.run()
                reactor_started = True
                print(f"✓ Reactor started (robot {i + 1} may still be connecting...)")
                break
            except RosTimeoutError:
                print(f"✗ Robot {i + 1} connection timeout, but reactor may be running")
                reactor_started = True
                break
            except Exception as e:
                error_str = str(e)
                if "ReactorNotRestartable" in error_str:
                    reactor_started = True
                    print("✓ Reactor already running (from previous attempt)")
                    break
                print(f"✗ Error starting reactor with robot {i + 1}: {e}")
                continue

    if not reactor_started:
        print("✗ Failed to start reactor - no valid clients")
        return [None] * len(ROBOT_CONFIGS), [None] * len(ROBOT_CONFIGS)

    time.sleep(0.5)

    for i, (client, config) in enumerate(zip(ros_clients, ROBOT_CONFIGS)):
        if client is None:
            ros_publishers.append(None)
            continue
        try:
            max_wait_time = 5.0
            wait_interval = 0.1
            waited = 0.0
            while not client.is_connected and waited < max_wait_time:
                time.sleep(wait_interval)
                waited += wait_interval

            if client.is_connected:
                print(f"✓ Connected to robot {i + 1} at {config['ip']}:{config['port']}")
                entry = _make_publisher_entry(client, config)
                ros_publishers.append(entry)
            else:
                print(f"✗ Failed to connect to robot {i + 1} at {config['ip']}:{config['port']} (timeout)")
                ros_publishers.append(None)
        except Exception as e:
            print(f"✗ Error setting up robot {i + 1}: {e}")
            ros_publishers.append(None)

    n_ok = sum(1 for c in ros_clients if c is not None and c.is_connected)
    print(f"Robot connection complete: {n_ok}/{len(ROBOT_CONFIGS)} connected")
    return ros_clients, ros_publishers


def julia_init():
    project_root = str(Path(__file__).resolve().parents[4])
    time_start = time.perf_counter()

    Main.eval(
        """
        ENV["JULIA_SSL_LIBRARY"] = "system"
        ENV["GKSwstype"] = "nul"
        try
            using OpenSSL_jll
        catch e
            @warn "OpenSSL_jll preload failed, continuing anyway" exception=e
        end
    """
    )

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

    time_middle = time.perf_counter()
    pre = Main.HardwareFunctions.build_lq_preoptimization(10, 0.1, silence_logs=True)
    time_end = time.perf_counter()
    print(f"Time taken: {time_end - time_middle} seconds")
    print("Preoptimization built successfully")
    print(f"Total time taken: {time_end - time_start} seconds")
    return pre


def goal_reached(position, goal_position, threshold=0.3):
    distance = math.sqrt((position[0] - goal_position[0]) ** 2 + (position[1] - goal_position[1]) ** 2)
    return distance < threshold


def ros_time():
    t = time.time()
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    return {"sec": sec, "nanosec": nanosec}


def twist_stamped_message(vx, omega):
    return {
        "header": {"stamp": ros_time(), "frame_id": "teleop_twist_joy"},
        "twist": {
            "linear": {"x": float(vx), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": float(omega)},
        },
    }


def twist_message(vx, omega):
    return {
        "linear": {"x": float(vx), "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": float(omega)},
    }


def float64_msg(data):
    return {"data": float(data)}


def publish_robot_cmd(pub_entry, cmd_linear, cmd_angular):
    """Send one command step: Twist uses (vx, omega); JetRacer uses (steering, throttle) as Float64."""
    kind = pub_entry["kind"]
    if kind == "jetracer":
        steering, throttle = float(cmd_linear), float(cmd_angular)
        pub_entry["steering"].publish(roslibpy.Message(float64_msg(steering)))
        pub_entry["throttle"].publish(roslibpy.Message(float64_msg(throttle)))
    elif kind == "twist":
        pub_entry["pub"].publish(roslibpy.Message(twist_message(cmd_linear, cmd_angular)))
    else:
        pub_entry["pub"].publish(roslibpy.Message(twist_stamped_message(cmd_linear, cmd_angular)))


def unadvertise_publisher_entry(pub_entry):
    if pub_entry is None:
        return
    if pub_entry["kind"] == "jetracer":
        pub_entry["steering"].unadvertise()
        pub_entry["throttle"].unadvertise()
    else:
        pub_entry["pub"].unadvertise()


class MultiJetRacerController(Node):
    def __init__(self, pre, ros_clients, ros_publishers):
        super().__init__("multi_jetracer_controller")

        self.odom_subs = []
        pose_attrs = ["latest_pose_01", "latest_pose_02", "latest_pose_03"]
        for topic, attr_name in zip(VRPN_POSE_TOPICS, pose_attrs):
            setattr(self, attr_name, None)
            cb = self._make_pose_callback(attr_name)
            sub = self.create_subscription(PoseStamped, topic, cb, 10)
            self.odom_subs.append(sub)
            self.get_logger().info(f"Subscribing Vicon pose: {topic}")

        self.current_commands = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        self.prev_omega = [0.0, 0.0, 0.0]

        self._jr_prev_pose_xy = None
        self._jr_prev_heading_err = 0.0
        self._jr_int_heading = 0.0
        self._jr_int_speed = 0.0
        self._jr_steering_filt = 0.0
        self._jr_throttle_filt = 0.0

        self.timer = self.create_timer(0.1, self.run_planner_step)
        self.publisher_timer = self.create_timer(0.01, self._publish_current_commands)

        self.pre = pre
        self.z_guess = None

        self.trajectory = []
        self.project_root = str(Path(__file__).resolve().parents[4])
        self.csv_output_path = Path(self.project_root) / "ros2" / "trajectory_multi_jetracer.csv"
        self._shutdown_initiated = False

        self.ros_clients = ros_clients
        self.ros_publishers = ros_publishers

        connected_count = sum(1 for c in self.ros_clients if c is not None)
        self.get_logger().info(f"Using {connected_count}/{len(ROBOT_CONFIGS)} pre-established robot connections")
        for i, client in enumerate(self.ros_clients):
            if client is not None and client.is_connected:
                self.get_logger().info(f"Robot {i + 1} connection active")
            else:
                self.get_logger().warn(f"Robot {i + 1} connection not available")

        self.get_logger().info("MultiJetRacerController node started.")

    def _make_pose_callback(self, attr_name):
        def _cb(msg):
            setattr(self, attr_name, msg)

        return _cb

    def convert_odom_to_state(self, msg):
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear.x
        else:
            x = msg.pose.position.x
            y = msg.pose.position.y
            q = msg.pose.orientation
            v = 0.0

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        return [x, y, theta, v]

    def _planar_heading_error(self, pose_xy, target_xy, current_theta):
        """Orientation residual (rad): heading toward target waypoint vs current yaw, wrapped to [-pi, pi]."""
        target_theta = math.atan2(target_xy[1] - pose_xy[1], target_xy[0] - pose_xy[0])
        return (target_theta - current_theta + math.pi) % (2.0 * math.pi) - math.pi

    def _reset_jetracer_pid(self):
        self._jr_prev_pose_xy = None
        self._jr_prev_heading_err = 0.0
        self._jr_int_heading = 0.0
        self._jr_int_speed = 0.0
        self._jr_steering_filt = 0.0
        self._jr_throttle_filt = 0.0

    def _jetracer_pid_steering_throttle(self, vx, vy, pose_xy, theta, next_xy, dt):
        """
        PID on heading (optional) and speed vs ref ||(vx,vy)|| toward MPC next state.
        Measured speed from Vicon position delta / dt (no wheel odometry).
        Throttle: clip then first-order smooth. Steering: 0 if JETRACER_STEERING_DISABLED.
        """
        v_ref = math.hypot(vx, vy)
        if self._jr_prev_pose_xy is None:
            v_meas = 0.0
        else:
            px, py = self._jr_prev_pose_xy
            v_meas = math.hypot(pose_xy[0] - px, pose_xy[1] - py) / max(dt, 1e-6)
        self._jr_prev_pose_xy = (float(pose_xy[0]), float(pose_xy[1]))

        e_h = self._planar_heading_error(pose_xy, next_xy, theta)

        if JETRACER_STEERING_DISABLED:
            self._jr_steering_filt = 0.0
            self._jr_int_heading = 0.0
            self._jr_prev_heading_err = e_h
            steer_out = 0.0
        else:
            de_h = (e_h - self._jr_prev_heading_err) / max(dt, 1e-6)
            self._jr_prev_heading_err = e_h

            self._jr_int_heading += e_h * dt
            self._jr_int_heading = float(
                np.clip(self._jr_int_heading, -JETRACER_HEADING_I_CLAMP, JETRACER_HEADING_I_CLAMP)
            )

            u_steering = (
                JETRACER_KP_HEAD * e_h
                + JETRACER_KI_HEAD * self._jr_int_heading
                + JETRACER_KD_HEAD * de_h
            )
            u_steer_sat = float(np.clip(u_steering, -JETRACER_MAX_STEERING, JETRACER_MAX_STEERING))
            a_s = JETRACER_STEERING_SMOOTH_ALPHA
            self._jr_steering_filt = (1.0 - a_s) * self._jr_steering_filt + a_s * u_steer_sat
            steering = float(np.clip(self._jr_steering_filt, -JETRACER_MAX_STEERING, JETRACER_MAX_STEERING))
            self._jr_steering_filt = steering
            steer_out = steering
            if abs(steer_out) < JETRACER_STEERING_DEADBAND:
                steer_out = 0.0

        e_v = v_ref - v_meas
        self._jr_int_speed += e_v * dt
        self._jr_int_speed = float(
            np.clip(self._jr_int_speed, -JETRACER_SPEED_I_CLAMP, JETRACER_SPEED_I_CLAMP)
        )
        u_throttle = (
            JETRACER_KFF_THROTTLE * v_ref
            + JETRACER_KP_SPEED * e_v
            + JETRACER_KI_SPEED * self._jr_int_speed
        )
        throttle_raw = float(np.clip(max(0.0, u_throttle), 0.0, JETRACER_MAX_THROTTLE))

        if v_ref < 0.04:
            self._jr_int_speed *= 0.85
            throttle_raw = 0.0

        a_t = JETRACER_THROTTLE_SMOOTH_ALPHA
        self._jr_throttle_filt = (1.0 - a_t) * self._jr_throttle_filt + a_t * throttle_raw
        throttle = float(np.clip(self._jr_throttle_filt, 0.0, JETRACER_MAX_THROTTLE))

        if throttle < JETRACER_THROTTLE_DEADBAND:
            throttle = 0.0
        return steer_out, throttle, e_h, v_ref, v_meas

    def convert_to_cmd_vel(self, vx, vy, pose, target_pose, current_theta, robot_idx=0):
        v = math.hypot(vx, vy)
        angle_error = self._planar_heading_error(pose, target_pose, current_theta)

        # Index 0: lonebot (Jackal) — gentler gains (same idea as large robot in pursuit_evasion)
        if robot_idx == 0:
            k_omega = 0.8
        else:
            k_omega = 1.5
        omega_cmd = k_omega * angle_error

        if robot_idx == 0:
            alpha = 0.4
        else:
            alpha = 0.6
        prev_omega = self.prev_omega[robot_idx] if robot_idx < len(self.prev_omega) else 0.0
        omega = alpha * omega_cmd + (1.0 - alpha) * prev_omega

        if robot_idx < len(self.prev_omega):
            self.prev_omega[robot_idx] = omega

        return v, omega

    def _publish_stop(self):
        for i, pub_entry in enumerate(self.ros_publishers):
            if pub_entry is None:
                continue
            try:
                publish_robot_cmd(pub_entry, 0.0, 0.0)
            except Exception as e:
                self.get_logger().error(f"Error sending stop to robot {i + 1}: {e}")

    def _publish_current_commands(self):
        if self._shutdown_initiated:
            return
        for i, (vx, omega) in enumerate(self.current_commands):
            if i >= len(self.ros_publishers) or self.ros_publishers[i] is None:
                continue
            try:
                publish_robot_cmd(self.ros_publishers[i], vx, omega)
            except Exception as e:
                self.get_logger().error(f"Error publishing to robot {i + 1}: {e}")

    def _save_trajectory_csv(self):
        if len(self.trajectory) == 0:
            self.get_logger().warn("No trajectory data to save.")
            return

        self.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.csv_output_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["lonebot_x", "lonebot_y", "bluebonnet_x", "bluebonnet_y", "jetracer_x", "jetracer_y"]
                )
                for point in self.trajectory:
                    writer.writerow(
                        [
                            point[0][0],
                            point[0][1],
                            point[1][0],
                            point[1][1],
                            point[2][0],
                            point[2][1],
                        ]
                    )
            self.get_logger().info(
                f"Saved trajectory CSV to: {self.csv_output_path} ({len(self.trajectory)} points)"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to save trajectory CSV: {e}")

    def run_planner_step(self):
        if (
            self.latest_pose_01 is None
            or self.latest_pose_02 is None
            or self.latest_pose_03 is None
        ):
            self.get_logger().warn("Waiting for all Vicon poses (lonebot, Bluebonnet, JetRacer)...")
            return

        # state1 = agent1 lonebot, state2 = agent2 Bluebonnet, state3 = agent3 JetRacer (evader → origin)
        state1 = self.convert_odom_to_state(self.latest_pose_01)
        state2 = self.convert_odom_to_state(self.latest_pose_02)
        state3 = self.convert_odom_to_state(self.latest_pose_03)

        julia_state1 = Main.eval(f"[{state1[0]}; {state1[1]}]")
        julia_state2 = Main.eval(f"[{state2[0]}; {state2[1]}]")
        julia_state3 = Main.eval(f"[{state3[0]}; {state3[1]}]")
        initial_state = [julia_state1, julia_state2, julia_state3]

        result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(
            self.pre, initial_state, self.z_guess, silence_logs=False
        )

        next_states = result.x_next
        curr_controls = result.u_curr
        z_sol = result.z_sol
        self.z_guess = z_sol

        self.get_logger().info(
            f"Positions — A1 lonebot=[{state1[0]:.3f}, {state1[1]:.3f}], "
            f"A2 Bluebonnet=[{state2[0]:.3f}, {state2[1]:.3f}], "
            f"A3 JetRacer=[{state3[0]:.3f}, {state3[1]:.3f}]"
        )
        self.get_logger().info(f"Next states: {next_states}")
        self.get_logger().info(f"Controls: {curr_controls}")

        u1, u2, u3 = curr_controls[0], curr_controls[1], curr_controls[2]
        states = (state1, state2, state3)
        controls = (u1, u2, u3)

        self.trajectory.append(((state1[0], state1[1]), (state2[0], state2[1]), (state3[0], state3[1])))

        current_commands = []
        log_parts = []
        for i in range(3):
            u = controls[i]
            st = states[i]
            cfg = ROBOT_CONFIGS[i]
            if cfg.get("msg_type") == "jetracer":
                vx, vy = u[0], u[1]
                nx, ny = float(next_states[i][0]), float(next_states[i][1])
                steering, throttle, e_h, v_ref, v_meas = self._jetracer_pid_steering_throttle(
                    vx, vy, [st[0], st[1]], st[2], [nx, ny], PLANNER_DT
                )
                current_commands.append((steering, throttle))
                log_parts.append(
                    f"A3 JetRacer vref={v_ref:.3f} vm={v_meas:.3f} eh={e_h:.3f} "
                    f"steer={steering:.3f} thr={throttle:.3f}"
                )
            else:
                v, omega = self.convert_to_cmd_vel(
                    u[0], u[1], [st[0], st[1]], next_states[i], st[2], robot_idx=i
                )
                if i == 0:
                    omega = float(np.clip(omega, -0.3, 0.3))
                    deadband_v, deadband_omega = 0.1, 0.08
                else:
                    omega = float(np.clip(omega, -0.5, 0.5))
                    deadband_v, deadband_omega = 0.05, 0.05
                v = float(np.clip(v, -0.5, 0.5))
                if abs(v) < deadband_v:
                    v = 0.0
                if abs(omega) < deadband_omega:
                    omega = 0.0
                current_commands.append((v, omega))
                tag = "A1 lonebot" if i == 0 else "A2 Bluebonnet"
                log_parts.append(f"{tag} v={v:.3f} w={omega:.3f}")

        self.get_logger().info(", ".join(log_parts))

        # Stop when agent 3 (JetRacer) reaches the origin
        if not goal_reached(state3[:2], GOAL_POSITION):
            self.current_commands = current_commands
        else:
            if not self._shutdown_initiated:
                self.get_logger().info("Goal reached for agent 3 (JetRacer / evader), stopping.")
                self._reset_jetracer_pid()
                self.timer.cancel()
                self.publisher_timer.cancel()
                self.current_commands = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
                self._publish_stop()
                self._save_trajectory_csv()
                self._shutdown_initiated = True
                for pub_entry in self.ros_publishers:
                    if pub_entry is not None:
                        try:
                            unadvertise_publisher_entry(pub_entry)
                        except Exception:
                            pass
                for client in self.ros_clients:
                    if client is not None:
                        try:
                            client.terminate()
                        except Exception:
                            pass
                self.destroy_node()
                rclpy.shutdown()

    def destroy_node(self):
        for pub_entry in self.ros_publishers:
            if pub_entry is not None:
                try:
                    unadvertise_publisher_entry(pub_entry)
                except Exception:
                    pass
        for client in self.ros_clients:
            if client is not None:
                try:
                    client.terminate()
                except Exception:
                    pass
        super().destroy_node()


def spin(pre, ros_clients, ros_publishers, args=None):
    """Run the ROS node after Julia and roslibpy are ready."""
    rclpy.init(args=args)
    node = MultiJetRacerController(pre, ros_clients, ros_publishers)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    """Entry point for `ros2 run` and direct execution: connect robots, load Julia, then spin."""
    ros_clients, ros_publishers = connect_to_robots()
    pre = julia_init()
    spin(pre, ros_clients, ros_publishers, args=args)


if __name__ == "__main__":
    main()
