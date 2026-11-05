#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
domain_id="${ROS_DOMAIN_ID:-0}"
lock_file="/tmp/rebotarm_camera_sim_domain_${domain_id}.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "A reBotArm camera simulation is already running in ROS_DOMAIN_ID=$domain_id." >&2
    echo "Stop the existing grasp_sim launch before starting another instance." >&2
    exit 1
fi
set +u
source /opt/ros/humble/setup.bash
source "$project_root/install/setup.bash"
set -u
# Expose source changes without replacing existing build/install artifacts.
export PYTHONPATH="$project_root/src/rebotarm_description:$project_root/src/rebotarm_planning:$project_root/src/rebotarm_mujoco_rs:$project_root/src/rebotarm_vision:$project_root/src/rebotarm_tasks:$project_root/src/rebotarm_agent:${PYTHONPATH:-}"
if ! python3 -c 'import mujoco' >/dev/null 2>&1; then
    python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    owner_site="$(dirname -- "$project_root")/.local/lib/python${python_version}/site-packages"
    if [[ -d "$owner_site/mujoco" ]]; then
        export PYTHONPATH="$owner_site:$PYTHONPATH"
    fi
fi
python3 -c 'import mujoco, cv2, scipy, rclpy'
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$project_root/log/camera_sim}"
exec ros2 launch rebotarm_bringup grasp.launch.py backend:=mujoco \
    arm_namespace:=rebotarm use_viewer:="${REBOTARM_VIEWER:-false}" "$@"
