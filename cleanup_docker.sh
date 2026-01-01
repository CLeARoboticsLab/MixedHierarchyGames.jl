#!/usr/bin/env bash
# Script to clean up Docker resources and free disk space

set -euo pipefail

echo "=== Docker Disk Usage Summary ==="
docker system df

echo ""
echo "=== Cleaning up Docker resources ==="

# Remove all stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove all unused images (not just dangling)
echo "Removing unused images..."
docker image prune -a -f

# Remove build cache
echo "Removing build cache..."
docker builder prune -af

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

echo ""
echo "=== Final Disk Usage Summary ==="
docker system df

echo ""
echo "Cleanup complete!"

