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

# Matplotlib imports with non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

GOAL_POSITION = [0.0, 0.0]
GOAL_THRESHOLD = 0.3  # Distance threshold to consider goal reached
DT = 0.1  # Time step (seconds)

# Initial positions for the three robots (can be modified)
# Format: [x, y, theta] where theta is initial orientation in radians
INITIAL_STATES = [
    [-3.0, 1.0, 0.0],   # Robot 1: [x, y, theta]
    [-2.0, -2.0, 0.0],  # Robot 2: [x, y, theta]
    [3.0, -3.0, 3.14],   # Robot 3: [x, y, theta]
]


def julia_init():
    """Initialize Julia and load necessary modules."""
    # Go up to the main project root: ros2/src/multi_robot_controller/multi_robot_controller -> main project
    project_root = str(Path(__file__).resolve().parents[4])
    time_start = time.perf_counter()
    
    # Pre-load OpenSSL_jll to use system OpenSSL libraries
    # Set environment variables to avoid Qt6/FreeType compatibility issues
    # CRITICAL: Set GKSwstype BEFORE any Plots/GR loading
    Main.eval("""
        ENV["JULIA_SSL_LIBRARY"] = "system"
        # Set headless backend BEFORE Plots is loaded to avoid Qt6
        ENV["GKSwstype"] = "nul"
        # Load OpenSSL_jll with system libraries
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
        
        # CRITICAL: Set GKSwstype BEFORE Plots loads (automatic_solver.jl may use Plots)
        # This must be set in Julia's ENV before any Plots/GR code runs
        ENV["GKSwstype"] = "nul"
        
        # Pre-load Plots with null backend to avoid Qt6 loading
        # This must happen BEFORE automatic_solver.jl is included
        # Wrap in try-catch to continue even if Plots fails to load
        try
            using Plots
            # Force GR to use null backend (no GUI, no Qt6)
            # This should prevent Qt6 from loading
            try
                gr(show = false, fmt = :png)
            catch e2
                @warn "gr() configuration failed" exception=e2
                # Continue anyway - might still work
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
    print(f"Julia initialization time: {time_middle - time_start:.2f} seconds")
    print(f"Preoptimization build time: {time_end - time_middle:.2f} seconds")
    print(f"Total initialization time: {time_end - time_start:.2f} seconds")
    print(f"Preoptimization built successfully\n")
    return pre
    

def goal_reached(position, goal_position, threshold=GOAL_THRESHOLD):
    """Check if the robot has reached the goal position within a threshold."""
    distance = math.sqrt((position[0] - goal_position[0]) ** 2 + 
                        (position[1] - goal_position[1]) ** 2)
    return distance < threshold


def convert_to_cmd_vel(vx, vy, current_pose, target_pose, current_theta, dt=DT):
    """
    Convert [vx, vy] control to differential drive [v, omega] control.
    
    Parameters:
    -----------
    vx, vy : Velocity components in world frame
    current_pose : [x, y] current position
    target_pose : [x, y] target position (from solver)
    current_theta : Current orientation (radians)
    dt : Time step (seconds)
    
    Returns:
    --------
    v : Linear velocity (m/s)
    omega : Angular velocity (rad/s)
    """
    # Linear velocity magnitude
    v = math.hypot(vx, vy)
    
    # Target orientation (direction to target position)
    target_theta = math.atan2(target_pose[1] - current_pose[1], 
                              target_pose[0] - current_pose[0])
    
    # Angular velocity to align with target direction
    # Normalize angle difference to [-pi, pi]
    angle_diff = target_theta - current_theta
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    
    omega = angle_diff / dt
    
    return v, omega


def diff_drive_dynamics(x, y, theta, v, omega, dt=DT):
    """
    Simulate differential drive robot dynamics.
    
    Parameters:
    -----------
    x, y, theta : Current state [x, y, orientation]
    v : Linear velocity (m/s)
    omega : Angular velocity (rad/s)
    dt : Time step (seconds)
    
    Returns:
    --------
    x_new, y_new, theta_new : New state after dt
    """
    # Differential drive kinematics
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = theta + omega * dt
    
    # Normalize theta to [-pi, pi]
    while theta_new > math.pi:
        theta_new -= 2 * math.pi
    while theta_new < -math.pi:
        theta_new += 2 * math.pi
    
    return x_new, y_new, theta_new


def normalize_julia_result(x):
    """Normalize Julia/PyCall results to Python lists."""
    try:
        tl = x.tolist()
        if tl is not None:
            return normalize_julia_result(tl)
    except:
        pass
    if isinstance(x, (list, tuple)):
        return [normalize_julia_result(xx) for xx in x]
    try:
        return float(x)
    except:
        return x


def run_receding_horizon_planning(pre, initial_states, goal_position, dt=DT, max_iterations=1000):
    """
    Run receding horizon planning until robot 3 reaches the goal.
    Uses differential drive dynamics for robot simulation.
    
    Parameters:
    -----------
    pre : Preoptimization object from Julia
    initial_states : List of [x, y, theta] for each robot
    goal_position : [x, y] goal position for robot 3
    dt : Time step (seconds)
    max_iterations : Maximum number of planning iterations
    
    Returns:
    --------
    trajectories : List of ((x1, y1, theta1), (x2, y2, theta2), (x3, y3, theta3)) tuples
    controls : List of ((v1, omega1), (v2, omega2), (v3, omega3)) tuples (diff-drive controls)
    """
    # Initialize current states [x, y, theta] for each robot
    current_states = [list(state) for state in initial_states]
    
    # Storage for trajectories and controls
    trajectories = []
    controls = []
    
    z_guess = None  # Warm-start guess for internal solver variables
    iteration = 0
    
    print(f"Starting receding horizon planning with differential drive dynamics...")
    print(f"Initial states:")
    print(f"  Robot 1: x={current_states[0][0]:.3f}, y={current_states[0][1]:.3f}, theta={current_states[0][2]:.3f}")
    print(f"  Robot 2: x={current_states[1][0]:.3f}, y={current_states[1][1]:.3f}, theta={current_states[1][2]:.3f}")
    print(f"  Robot 3: x={current_states[2][0]:.3f}, y={current_states[2][1]:.3f}, theta={current_states[2][2]:.3f}")
    print(f"Goal position (Robot 3): {goal_position}\n")
    
    while iteration < max_iterations:
        iteration += 1
        
        # Record current trajectory (with orientation)
        trajectories.append((
            tuple(current_states[0]),  # [x, y, theta]
            tuple(current_states[1]),  # [x, y, theta]
            tuple(current_states[2])   # [x, y, theta]
        ))
        
        # Check if robot 3 has reached the goal (using position only)
        if goal_reached([current_states[2][0], current_states[2][1]], goal_position):
            print(f"\nGoal reached at iteration {iteration}!")
            print(f"Final Robot 3 state: x={current_states[2][0]:.3f}, y={current_states[2][1]:.3f}, theta={current_states[2][2]:.3f}")
            print(f"Goal position: {goal_position}")
            print(f"Distance: {math.sqrt((current_states[2][0] - goal_position[0])**2 + (current_states[2][1] - goal_position[1])**2):.3f}")
            break
        
        # Prepare current states for Julia solver (only position [px, py])
        # Julia solver expects vector of vectors: [[px; py], [px; py], [px; py]]
        julia_state1 = Main.eval(f"[{current_states[0][0]}; {current_states[0][1]}]")
        julia_state2 = Main.eval(f"[{current_states[1][0]}; {current_states[1][1]}]") 
        julia_state3 = Main.eval(f"[{current_states[2][0]}; {current_states[2][1]}]")
        initial_state = [julia_state1, julia_state2, julia_state3]
        
        # Call Julia solver
        try:
            if z_guess is None:
                result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(
                    pre, initial_state, silence_logs=True)
            else:
                result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(
                    pre, initial_state, z_guess, silence_logs=True)
        except Exception as e:
            print(f"Error in Julia solver at iteration {iteration}: {e}")
            break
        
        # Extract results
        next_states_2d = normalize_julia_result(result.x_next)  # [[x, y], [x, y], [x, y]]
        curr_controls_2d = normalize_julia_result(result.u_curr)  # [[vx, vy], [vx, vy], [vx, vy]]
        z_sol = result.z_sol
        
        # Update z_guess for warm-starting next iteration
        z_guess = z_sol
        
        # Convert [vx, vy] controls to differential drive [v, omega] controls
        v1, omega1 = convert_to_cmd_vel(
            curr_controls_2d[0][0], curr_controls_2d[0][1],
            [current_states[0][0], current_states[0][1]],
            next_states_2d[0],
            current_states[0][2], dt
        )
        v2, omega2 = convert_to_cmd_vel(
            curr_controls_2d[1][0], curr_controls_2d[1][1],
            [current_states[1][0], current_states[1][1]],
            next_states_2d[1],
            current_states[1][2], dt
        )
        v3, omega3 = convert_to_cmd_vel(
            curr_controls_2d[2][0], curr_controls_2d[2][1],
            [current_states[2][0], current_states[2][1]],
            next_states_2d[2],
            current_states[2][2], dt
        )
        
        # Clip velocities for safety
        v1 = np.clip(v1, -0.4, 0.4)
        v2 = np.clip(v2, -0.4, 0.4)
        v3 = np.clip(v3, -0.4, 0.4)
        omega1 = np.clip(omega1, -0.5, 0.5)
        omega2 = np.clip(omega2, -0.5, 0.5)
        omega3 = np.clip(omega3, -0.5, 0.5)
        
        # Record diff-drive controls
        controls.append((
            (v1, omega1),  # [v, omega] for Robot 1
            (v2, omega2),  # [v, omega] for Robot 2
            (v3, omega3)  # [v, omega] for Robot 3
        ))
        
        # Simulate differential drive dynamics to update robot states
        x1_new, y1_new, theta1_new = diff_drive_dynamics(
            current_states[0][0], current_states[0][1], current_states[0][2], v1, omega1, dt
        )
        x2_new, y2_new, theta2_new = diff_drive_dynamics(
            current_states[1][0], current_states[1][1], current_states[1][2], v2, omega2, dt
        )
        x3_new, y3_new, theta3_new = diff_drive_dynamics(
            current_states[2][0], current_states[2][1], current_states[2][2], v3, omega3, dt
        )
        
        # Update current states
        current_states[0] = [x1_new, y1_new, theta1_new]
        current_states[1] = [x2_new, y2_new, theta2_new]
        current_states[2] = [x3_new, y3_new, theta3_new]
        
        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Robot 3 at [{current_states[2][0]:.3f}, {current_states[2][1]:.3f}], "
                  f"theta={current_states[2][2]:.3f}, "
                  f"distance to goal: {math.sqrt((current_states[2][0] - goal_position[0])**2 + (current_states[2][1] - goal_position[1])**2):.3f}")
    
    if iteration >= max_iterations:
        print(f"\nMaximum iterations ({max_iterations}) reached.")
    
    print(f"\nPlanning completed. Total iterations: {iteration}")
    print(f"Total trajectory points: {len(trajectories)}")
    print(f"Total control points: {len(controls)}")
    
    return trajectories, controls


def save_results_to_csv(trajectories, controls, output_dir=None):
    """Save trajectories and controls to CSV files."""
    if output_dir is None:
        # Save to project root / ros2 directory
        project_root = Path(__file__).resolve().parents[4]
        output_dir = project_root / "ros2"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectories (with orientation)
    trajectories_path = output_dir / "receding_horizon_trajectories.csv"
    try:
        with open(trajectories_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['robot1_x', 'robot1_y', 'robot1_theta', 
                           'robot2_x', 'robot2_y', 'robot2_theta',
                           'robot3_x', 'robot3_y', 'robot3_theta'])
            
            # Write trajectory data
            for point in trajectories:
                writer.writerow([
                    point[0][0], point[0][1], point[0][2],  # Robot 1: x, y, theta
                    point[1][0], point[1][1], point[1][2],  # Robot 2: x, y, theta
                    point[2][0], point[2][1], point[2][2]   # Robot 3: x, y, theta
                ])
        
        print(f"Saved trajectories to: {trajectories_path} ({len(trajectories)} points)")
    except Exception as e:
        print(f"Failed to save trajectories CSV: {e}")
    
    # Save controls (differential drive: v, omega)
    controls_path = output_dir / "receding_horizon_controls.csv"
    try:
        with open(controls_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['robot1_v', 'robot1_omega', 
                           'robot2_v', 'robot2_omega',
                           'robot3_v', 'robot3_omega'])
            
            # Write control data
            for control in controls:
                writer.writerow([
                    control[0][0], control[0][1],  # Robot 1: v, omega
                    control[1][0], control[1][1],  # Robot 2: v, omega
                    control[2][0], control[2][1]   # Robot 3: v, omega
                ])
        
        print(f"Saved controls to: {controls_path} ({len(controls)} points)")
    except Exception as e:
        print(f"Failed to save controls CSV: {e}")


def plot_trajectories(trajectories, goal_position, initial_states, output_dir=None):
    """
    Plot the trajectories of all three robots and save as PNG.
    Shows orientation arrows at regular intervals.
    
    Parameters:
    -----------
    trajectories : List of ((x1, y1, theta1), (x2, y2, theta2), (x3, y3, theta3)) tuples
    goal_position : [x, y] goal position for robot 3
    initial_states : List of [x, y, theta] initial states
    output_dir : Directory to save the plot
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[4]
        output_dir = project_root / "ros2"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract trajectories for each robot
    r1_x = [point[0][0] for point in trajectories]
    r1_y = [point[0][1] for point in trajectories]
    r1_theta = [point[0][2] for point in trajectories]
    r2_x = [point[1][0] for point in trajectories]
    r2_y = [point[1][1] for point in trajectories]
    r2_theta = [point[1][2] for point in trajectories]
    r3_x = [point[2][0] for point in trajectories]
    r3_y = [point[2][1] for point in trajectories]
    r3_theta = [point[2][2] for point in trajectories]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectories
    ax.plot(r1_x, r1_y, 'b-', linewidth=2, label='Robot 1', alpha=0.7)
    ax.plot(r2_x, r2_y, 'g-', linewidth=2, label='Robot 2', alpha=0.7)
    ax.plot(r3_x, r3_y, 'r-', linewidth=2, label='Robot 3', alpha=0.7)
    
    # Plot orientation arrows at regular intervals (every 10th point)
    arrow_interval = max(1, len(trajectories) // 20)
    arrow_length = 0.3
    
    for i in range(0, len(trajectories), arrow_interval):
        # Robot 1
        dx1 = arrow_length * math.cos(r1_theta[i])
        dy1 = arrow_length * math.sin(r1_theta[i])
        ax.arrow(r1_x[i], r1_y[i], dx1, dy1, head_width=0.1, head_length=0.1, 
                fc='blue', ec='blue', alpha=0.5, length_includes_head=True)
        
        # Robot 2
        dx2 = arrow_length * math.cos(r2_theta[i])
        dy2 = arrow_length * math.sin(r2_theta[i])
        ax.arrow(r2_x[i], r2_y[i], dx2, dy2, head_width=0.1, head_length=0.1, 
                fc='green', ec='green', alpha=0.5, length_includes_head=True)
        
        # Robot 3
        dx3 = arrow_length * math.cos(r3_theta[i])
        dy3 = arrow_length * math.sin(r3_theta[i])
        ax.arrow(r3_x[i], r3_y[i], dx3, dy3, head_width=0.1, head_length=0.1, 
                fc='red', ec='red', alpha=0.5, length_includes_head=True)
    
    # Plot start positions
    ax.plot(initial_states[0][0], initial_states[0][1], 'bo', 
            markersize=12, label='Robot 1 Start', markeredgecolor='black', markeredgewidth=2)
    ax.plot(initial_states[1][0], initial_states[1][1], 'go', 
            markersize=12, label='Robot 2 Start', markeredgecolor='black', markeredgewidth=2)
    ax.plot(initial_states[2][0], initial_states[2][1], 'ro', 
            markersize=12, label='Robot 3 Start', markeredgecolor='black', markeredgewidth=2)
    
    # Plot end positions
    ax.plot(r1_x[-1], r1_y[-1], 'bs', markersize=12, 
            label='Robot 1 End', markeredgecolor='black', markeredgewidth=2)
    ax.plot(r2_x[-1], r2_y[-1], 'gs', markersize=12, 
            label='Robot 2 End', markeredgecolor='black', markeredgewidth=2)
    ax.plot(r3_x[-1], r3_y[-1], 'rs', markersize=12, 
            label='Robot 3 End', markeredgecolor='black', markeredgewidth=2)
    
    # Plot goal position
    goal_circle = Circle(goal_position, GOAL_THRESHOLD, color='orange', 
                         fill=False, linestyle='--', linewidth=2, label='Goal Region')
    ax.add_patch(goal_circle)
    ax.plot(goal_position[0], goal_position[1], 'k*', markersize=15, 
            label='Goal', markeredgecolor='white', markeredgewidth=1)
    
    # Set labels and title
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Receding Horizon Planning Trajectories (Differential Drive)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Adjust layout and save
    plt.tight_layout()
    png_path = output_dir / "receding_horizon_trajectories.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory plot to: {png_path}")


def create_trajectory_gif(trajectories, goal_position, initial_states, output_dir=None, fps=10):
    """
    Create an animated GIF showing the trajectory evolution over time.
    Shows robot orientation as arrows.
    
    Parameters:
    -----------
    trajectories : List of ((x1, y1, theta1), (x2, y2, theta2), (x3, y3, theta3)) tuples
    goal_position : [x, y] goal position for robot 3
    initial_states : List of [x, y, theta] initial states
    output_dir : Directory to save the GIF
    fps : Frames per second for the animation
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[4]
        output_dir = project_root / "ros2"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract trajectories for each robot
    r1_x = np.array([point[0][0] for point in trajectories])
    r1_y = np.array([point[0][1] for point in trajectories])
    r1_theta = np.array([point[0][2] for point in trajectories])
    r2_x = np.array([point[1][0] for point in trajectories])
    r2_y = np.array([point[1][1] for point in trajectories])
    r2_theta = np.array([point[1][2] for point in trajectories])
    r3_x = np.array([point[2][0] for point in trajectories])
    r3_y = np.array([point[2][1] for point in trajectories])
    r3_theta = np.array([point[2][2] for point in trajectories])
    
    # Determine axis limits with some padding
    all_x = np.concatenate([r1_x, r2_x, r3_x, [goal_position[0]]])
    all_y = np.concatenate([r1_y, r2_y, r3_y, [goal_position[1]]])
    x_min, x_max = all_x.min() - 1, all_x.max() + 1
    y_min, y_max = all_y.min() - 1, all_y.max() + 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initialize plot elements
    line1, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Robot 1')
    line2, = ax.plot([], [], 'g-', linewidth=2, alpha=0.7, label='Robot 2')
    line3, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Robot 3')
    
    point1, = ax.plot([], [], 'bo', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    point2, = ax.plot([], [], 'go', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    point3, = ax.plot([], [], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    
    # Arrow length for orientation display
    arrow_length = 0.3
    
    # Plot start positions
    ax.plot(initial_states[0][0], initial_states[0][1], 'bs', 
            markersize=8, markeredgecolor='black', markeredgewidth=1, alpha=0.5)
    ax.plot(initial_states[1][0], initial_states[1][1], 'gs', 
            markersize=8, markeredgecolor='black', markeredgewidth=1, alpha=0.5)
    ax.plot(initial_states[2][0], initial_states[2][1], 'rs', 
            markersize=8, markeredgecolor='black', markeredgewidth=1, alpha=0.5)
    
    # Plot goal position
    goal_circle = Circle(goal_position, GOAL_THRESHOLD, color='orange', 
                         fill=False, linestyle='--', linewidth=2)
    ax.add_patch(goal_circle)
    ax.plot(goal_position[0], goal_position[1], 'k*', markersize=15, 
            markeredgecolor='white', markeredgewidth=1)
    
    # Set up axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Receding Horizon Planning Animation (Differential Drive)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Store arrows in a list for proper handling
    arrows = []
    
    # Animation function
    def animate(frame):
        # Show trajectory up to current frame
        line1.set_data(r1_x[:frame+1], r1_y[:frame+1])
        line2.set_data(r2_x[:frame+1], r2_y[:frame+1])
        line3.set_data(r3_x[:frame+1], r3_y[:frame+1])
        
        # Show current positions and orientations
        if frame < len(trajectories):
            point1.set_data([r1_x[frame]], [r1_y[frame]])
            point2.set_data([r2_x[frame]], [r2_y[frame]])
            point3.set_data([r3_x[frame]], [r3_y[frame]])
            
            # Remove old arrows
            for arrow in arrows:
                arrow.remove()
            arrows.clear()
            
            # Update orientation arrows
            dx1 = arrow_length * math.cos(r1_theta[frame])
            dy1 = arrow_length * math.sin(r1_theta[frame])
            dx2 = arrow_length * math.cos(r2_theta[frame])
            dy2 = arrow_length * math.sin(r2_theta[frame])
            dx3 = arrow_length * math.cos(r3_theta[frame])
            dy3 = arrow_length * math.sin(r3_theta[frame])
            
            # Create new arrows
            arrow1 = ax.arrow(r1_x[frame], r1_y[frame], dx1, dy1, 
                            head_width=0.15, head_length=0.15, 
                            fc='blue', ec='blue', alpha=0.8, length_includes_head=True)
            arrow2 = ax.arrow(r2_x[frame], r2_y[frame], dx2, dy2, 
                            head_width=0.15, head_length=0.15, 
                            fc='green', ec='green', alpha=0.8, length_includes_head=True)
            arrow3 = ax.arrow(r3_x[frame], r3_y[frame], dx3, dy3, 
                            head_width=0.15, head_length=0.15, 
                            fc='red', ec='red', alpha=0.8, length_includes_head=True)
            
            arrows.extend([arrow1, arrow2, arrow3])
        
        # Update title with iteration number
        ax.set_title(f'Receding Horizon Planning Animation (Iteration {frame+1}/{len(trajectories)})', 
                    fontsize=14, fontweight='bold')
        
        return [line1, line2, line3, point1, point2, point3] + arrows
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(trajectories), 
                                   interval=1000/fps, blit=False, repeat=True)
    
    # Save as GIF
    gif_path = output_dir / "receding_horizon_trajectories.gif"
    try:
        anim.save(gif_path, writer='pillow', fps=fps)
        print(f"Saved trajectory animation to: {gif_path}")
    except Exception as e:
        print(f"Failed to save GIF (trying with imagemagick): {e}")
        try:
            anim.save(gif_path, writer='imagemagick', fps=fps)
            print(f"Saved trajectory animation to: {gif_path}")
        except Exception as e2:
            print(f"Failed to save GIF: {e2}")
            print("Note: You may need to install pillow: pip install pillow")
    
    plt.close()


def main():
    """Main function to run receding horizon planning."""
    print("=" * 60)
    print("Receding Horizon Planning (No ROS)")
    print("=" * 60)
    print()
    
    # Initialize Julia
    print("Initializing Julia...")
    pre = julia_init()
    
    # Run receding horizon planning
    print("\n" + "=" * 60)
    trajectories, controls = run_receding_horizon_planning(
        pre, 
        INITIAL_STATES, 
        GOAL_POSITION,
        dt=DT
    )
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    save_results_to_csv(trajectories, controls)
    
    # Plot and save visualizations
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    plot_trajectories(trajectories, GOAL_POSITION, INITIAL_STATES)
    create_trajectory_gif(trajectories, GOAL_POSITION, INITIAL_STATES)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

