#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    cat <<'EOF'
在工作空间根目录运行：
  ./scripts/start_uart_bridge.sh /dev/ttyCH343USB0 # 启动，WCH 厂商驱动对应此名称
  ./scripts/start_uart_bridge.sh enable         # 另一终端：使能
  ./scripts/start_uart_bridge.sh disable        # 另一终端：失能

自动加载环境，串口参数读取统一硬件配置，默认关闭重力前馈。
标定完成后可在启动命令末尾加 gravity_feedforward:=true。
EOF
    exit 0
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uart_action="$1"
shift
# ROS 的环境脚本可能引用未定义变量，加载期间暂时关闭 nounset。
set +u
source /opt/ros/humble/setup.bash
source "$project_root/install/setup.bash"
set -u
export PYTHONPATH="$project_root/src/rebotarm_uart:${PYTHONPATH:-}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$project_root/log/uart_bridge}"
case "$uart_action" in
    enable|disable)
        exec ros2 service call "/rebotarm/$uart_action" std_srvs/srv/Trigger '{}'
        ;;
    *)
        # 同时启动串口桥和位置适配器；实际发包由 enable 服务开启。
        exec ros2 launch rebotarm_bringup real_uart.launch.py "port:=$uart_action" "$@"
        ;;
esac
