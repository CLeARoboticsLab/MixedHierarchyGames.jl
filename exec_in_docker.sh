#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=docker/docker-compose.yml
IMAGE=stackelberg/ros-jazzy-julia:latest
SERVICE=sh-game-dev

# Get current user's UID and GID (use different var names since UID is readonly in bash)
# If running with sudo, use the original user's UID/GID from SUDO_UID
if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
    export HOST_UID=${HOST_UID:-$SUDO_UID}
    export HOST_GID=${HOST_GID:-$SUDO_GID}
else
    export HOST_UID=${HOST_UID:-$(id -u)}
    export HOST_GID=${HOST_GID:-$(id -g)}
fi

# Handle rebuild request: stop, rebuild, and start fresh
if [ "${1-}" = "rebuild" ]; then
  echo "Rebuilding and restarting docker service..."
  echo "Using UID=$HOST_UID GID=$HOST_GID"
  if [ -f "$COMPOSE_FILE" ]; then
    docker compose -f "$COMPOSE_FILE" down
    docker compose -f "$COMPOSE_FILE" build --build-arg UID=$HOST_UID --build-arg GID=$HOST_GID
    docker compose -f "$COMPOSE_FILE" up -d
    shift  # Remove 'rebuild' from arguments so rest of script works normally
  else
    echo "Compose file $COMPOSE_FILE not found."
    exit 1
  fi
fi

echo "Looking for a running container for image $IMAGE..."
CONTAINER_ID=$(docker ps --filter ancestor=${IMAGE} --format "{{.ID}}" | head -n1 || true)

if [ -z "$CONTAINER_ID" ]; then
  echo "No running container found for image $IMAGE. Attempting to start compose service ($COMPOSE_FILE -> $SERVICE)..."
  echo "Using UID=$HOST_UID GID=$HOST_GID"
  if [ -f "$COMPOSE_FILE" ]; then
    docker compose -f "$COMPOSE_FILE" build --build-arg UID=$HOST_UID --build-arg GID=$HOST_GID
    docker compose -f "$COMPOSE_FILE" up -d
  else
    echo "Compose file $COMPOSE_FILE not found. Please start the container manually and re-run this script."
    exit 1
  fi

  # wait for the service container to appear
  echo -n "Waiting for container to start"
  for i in {1..60}; do
    CONTAINER_ID=$(docker compose -f "$COMPOSE_FILE" ps -q $SERVICE || true)
    if [ -n "$CONTAINER_ID" ]; then
      echo " -> started (container id $CONTAINER_ID)"
      break
    fi
    echo -n "."
    sleep 1
  done
  echo

  if [ -z "$CONTAINER_ID" ]; then
    # fallback to any container from the image
    CONTAINER_ID=$(docker ps --filter ancestor=${IMAGE} --format "{{.ID}}" | head -n1 || true)
    if [ -z "$CONTAINER_ID" ]; then
      echo "Timed out waiting for container to start. Check 'docker compose -f $COMPOSE_FILE ps' for error details."
      exit 2
    fi
  fi
fi

echo "Container $CONTAINER_ID is ready."

# Default behaviour: open an interactive shell
# If the first argument is 'logs', tail the service logs instead
if [ "${1-}" = "logs" ]; then
  echo "Tailing logs for service $SERVICE (ctrl-C to stop)"
  # First show recent logs (last 50 lines), then follow new ones
  docker compose -f "$COMPOSE_FILE" logs --tail 50 --no-log-prefix $SERVICE
  docker compose -f "$COMPOSE_FILE" logs -f --no-log-prefix $SERVICE
else
  echo "Opening interactive shell in container $CONTAINER_ID"
  docker exec -it "$CONTAINER_ID" bash
fi

