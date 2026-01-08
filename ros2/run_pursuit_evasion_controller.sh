#!/usr/bin/env bash
# Build and run the pursuit_evasion_controller node

colcon build
source install/setup.bash

# Increase memory limits if needed (uncomment and adjust as necessary)
# Note: These limits are in KB. Adjust based on your system's available RAM
ulimit -v 16777216  # 16GB virtual memory limit (in KB)
# ulimit -m 16777216  # 16GB physical memory limit (in KB)

# Set environment variables to avoid Qt6/FreeType compatibility issues
# Use headless backend for GR (Graphics Rendering) to avoid Qt6 dependency
export GKSwstype=nul
export JULIA_SSL_LIBRARY=system

# Increase Julia memory limits if needed (uncomment and adjust as necessary)
export JULIA_GC_MAX_MEMORY=17179869184  # 16GB in bytes (increase if you have more RAM)
export JULIA_GC_FULL_COLLECTIONS=1     # Force full GC more frequently

# Workaround for Qt6/FreeType compatibility issue:
# LD_PRELOAD forces Qt6 to use system FreeType library instead of Julia artifacts
# This fixes the "undefined symbol: FT_Get_Paint" error
FREETYPE_LIB="/usr/lib/x86_64-linux-gnu/libfreetype.so.6"
if [ ! -f "$FREETYPE_LIB" ]; then
    # Try alternative location
    FREETYPE_LIB="/lib/x86_64-linux-gnu/libfreetype.so.6"
fi

if [ -f "$FREETYPE_LIB" ]; then
    export LD_PRELOAD="$FREETYPE_LIB"
    echo "Using LD_PRELOAD workaround for FreeType compatibility: $FREETYPE_LIB"
else
    echo "Warning: FreeType library not found. Qt6/FreeType issues may occur."
fi

# Use venv Python explicitly to ensure we have all required packages (rclpy, yaml, etc.)
if [ -f /opt/venv/bin/python3 ]; then
    /opt/venv/bin/python3 src/multi_robot_controller/multi_robot_controller/pursuit_evasion_controller.py
else
    python3 src/multi_robot_controller/multi_robot_controller/pursuit_evasion_controller.py
fi

