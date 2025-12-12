# Docker Setup for Stackelberg Hierarchy Games

This Docker setup provides a complete development environment with:
- **Ubuntu 24.04** base image
- **ROS 2 Jazzy** (Jazzy Jalisco)
- **Julia 1.10.4** with all project dependencies
- **PyJulia** (via `juliacall` Python package) for Python-Julia integration
- **Python 3** with necessary packages

## Quick Start

### 1. Allow X11 forwarding (for GUI applications):
```bash
xhost +local:root
```

### 2. Build and start the container:
```bash
./exec_in_docker.sh rebuild
```

This will:
- Build the Docker image with your user ID/GID (prevents permission issues)
- Install all Julia packages from `Project.toml`
- Install Python packages (juliacall, numpy) in the container venv
- Start the container and open an interactive shell

### 3. Daily usage:

**Open interactive shell:**
```bash
./exec_in_docker.sh
```

**View container logs:**
```bash
./exec_in_docker.sh logs
```

**Force rebuild (if dependencies change):**
```bash
./exec_in_docker.sh rebuild
```

## Inside the Container

The container automatically:
- Sources ROS 2 Jazzy environment (`/opt/ros/jazzy/setup.bash`)
- Sets up Python venv in PATH (`/opt/venv/bin`)
  - The venv is created with `--system-site-packages` to access ROS packages
  - Includes juliacall, numpy, PyYAML, and other dependencies
  - Can access system-installed ROS packages (rclpy, etc.)
- Starts in `/home/developer/workspace/ros2` directory

### Building ROS Workspace

Once inside the container:
```bash
# Source ROS environment (already done by entrypoint)
source /opt/ros/jazzy/setup.bash

# Build the workspace
colcon build

# Source the workspace
source install/local_setup.bash

# Run the controller node
python3 src/multi_robot_controller/multi_robot_controller/controller_node.py
```

Or use the provided script:
```bash
./run_node.sh
```

## User ID Mapping

The container is configured to use your host user's UID/GID (currently: 1011/1011). This means:
- Files created in the container will be owned by you on the host
- You can edit files from the host without permission issues
- The workspace is mounted as a volume, so changes persist

## Project Structure

- `/home/developer/workspace` - Main project directory (mounted from host)
- `/home/developer/workspace/ros2` - ROS 2 workspace
- `/opt/venv` - Python virtual environment (persists in image)
- `/opt/julia-1.10.4` - Julia installation

## Notes

- The workspace is mounted from the host, so all changes are immediately visible
- Python packages are installed in `/opt/venv` which persists in the image
- Julia packages are installed in the project directory (persist via volume mount)
- X11 forwarding is enabled for GUI applications
- Container runs in privileged mode for hardware access (devices, GPU, etc.)

