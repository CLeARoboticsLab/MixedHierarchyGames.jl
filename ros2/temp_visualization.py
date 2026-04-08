#!/usr/bin/env python3
"""
Visualize robot trajectories from trajectory.csv
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

def load_trajectories(csv_file):
    """Load robot trajectories from CSV file"""
    robot1_x = []
    robot1_y = []
    robot2_x = []
    robot2_y = []
    robot3_x = []
    robot3_y = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            robot1_x.append(float(row['robot1_x']))
            robot1_y.append(float(row['robot1_y']))
            robot2_x.append(float(row['robot2_x']))
            robot2_y.append(float(row['robot2_y']))
            robot3_x.append(float(row['robot3_x']))
            robot3_y.append(float(row['robot3_y']))
    
    return (robot1_x, robot1_y), (robot2_x, robot2_y), (robot3_x, robot3_y)

def plot_trajectories(robot1, robot2, robot3):
    """Plot the trajectories of all three robots"""
    r1_x, r1_y = robot1
    r2_x, r2_y = robot2
    r3_x, r3_y = robot3
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot trajectories
    ax.plot(r1_x, r1_y, 'b-', linewidth=2, label='Robot 1 (BlueBonnet)', alpha=0.7)
    ax.plot(r2_x, r2_y, 'r-', linewidth=2, label='Robot 2 (Lonebot)', alpha=0.7)
    ax.plot(r3_x, r3_y, 'g-', linewidth=2, label='Robot 3 (Husky)', alpha=0.7)
    
    # Mark start positions
    ax.plot(r1_x[0], r1_y[0], 'bo', markersize=10, label='Robot 1 Start', markeredgecolor='black', markeredgewidth=1)
    ax.plot(r2_x[0], r2_y[0], 'ro', markersize=10, label='Robot 2 Start', markeredgecolor='black', markeredgewidth=1)
    ax.plot(r3_x[0], r3_y[0], 'go', markersize=10, label='Robot 3 Start', markeredgecolor='black', markeredgewidth=1)
    
    # Mark end positions
    ax.plot(r1_x[-1], r1_y[-1], 'bs', markersize=10, label='Robot 1 End', markeredgecolor='black', markeredgewidth=1)
    ax.plot(r2_x[-1], r2_y[-1], 'rs', markersize=10, label='Robot 2 End', markeredgecolor='black', markeredgewidth=1)
    ax.plot(r3_x[-1], r3_y[-1], 'gs', markersize=10, label='Robot 3 End', markeredgecolor='black', markeredgewidth=1)
    
    # Mark origin (goal for robot 3)
    ax.plot(0, 0, 'k*', markersize=15, label='Origin (Goal)', markeredgecolor='white', markeredgewidth=1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Robot Trajectories', fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    # Add some statistics
    stats_text = f'Total Steps: {len(r1_x)}\n'
    stats_text += f'Robot 1 Distance: {np.sum(np.sqrt(np.diff(r1_x)**2 + np.diff(r1_y)**2)):.2f} m\n'
    stats_text += f'Robot 2 Distance: {np.sum(np.sqrt(np.diff(r2_x)**2 + np.diff(r2_y)**2)):.2f} m\n'
    stats_text += f'Robot 3 Distance: {np.sum(np.sqrt(np.diff(r3_x)**2 + np.diff(r3_y)**2)):.2f} m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def main():
    csv_file = 'trajectory.csv'
    
    print(f"Loading trajectories from {csv_file}...")
    robot1, robot2, robot3 = load_trajectories(csv_file)
    
    print(f"Loaded {len(robot1[0])} trajectory points")
    print(f"Robot 1: Start=({robot1[0][0]:.3f}, {robot1[1][0]:.3f}), End=({robot1[0][-1]:.3f}, {robot1[1][-1]:.3f})")
    print(f"Robot 2: Start=({robot2[0][0]:.3f}, {robot2[1][0]:.3f}), End=({robot2[0][-1]:.3f}, {robot2[1][-1]:.3f})")
    print(f"Robot 3: Start=({robot3[0][0]:.3f}, {robot3[1][0]:.3f}), End=({robot3[0][-1]:.3f}, {robot3[1][-1]:.3f})")
    
    print("Plotting trajectories...")
    fig = plot_trajectories(robot1, robot2, robot3)
    
    # Save figure
    output_file = 'trajectory_visualization.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    
    # Show plot
    plt.show()

if __name__ == '__main__':
    main()
