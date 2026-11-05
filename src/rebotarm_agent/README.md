# rebotarm_agent

MCP 工具与文本客户端。实际抓取流程、夹爪等待及视觉验证位于 `rebotarm_tasks`，
IK、碰撞和轨迹执行位于 `rebotarm_planning`。默认命名空间为 `/rebotarm`，
当前模型使用 DM4340P / DM4310 / PA43；`rs_*` 仅为历史入口兼容名。

## 启动

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rebotarm_bringup grasp.launch.py backend:=mujoco
# 另一个终端，需可导入 fastmcp 的 Python 环境
ros2 launch rebotarm_agent rebotarm_mcp.launch.py motion_mode:=allow
```

MCP 默认地址 `http://127.0.0.1:8081/mcp`。运动默认锁定，需显式设置
`motion_mode:=allow`。也可在统一启动入口传 `start_mcp:=true` 和运动模式。
真机必须通过 `config:=/绝对路径/pick_sdk.yaml` 或 `pick_uart.yaml` 加载对应标定配置。

`python_executable:=...` 用于选择可同时导入 ROS 2 与 fastmcp 的解释器。
只运行抓取 CLI 时不需要 fastmcp 或语言模型：

```bash
ros2 run rebotarm_tasks pick --color red --action-timeout-padding-sec 40
```

`ros2 run rebotarm_agent rebotarm_pick` 仅为旧脚本保留兼容转发；新增脚本使用上面的 tasks 入口。兼容层不保存任务算法，可在外部客户端迁移后移除。
文本客户端可用 `rebotarm_text_agent.launch.py`，通过 `mcp_url`、`base_url`、
`model` 和 `python_executable` 配置连接。

## 工具和单位

- `diagnose_ros`、`get_robot_status`：接口及反馈新鲜度。
- `detect_blocks`：RGB-D 色块检测。
- `pick_color`：调用共享抓取任务，必须经过接触候选、抬升和保持后的视觉验证。
- `set_gripper_opening_mm`：兼容命令刻度 0～90；实际指间距根据任务配置换算，
  并非直接等于这个数值。
- `ik_check`、`move_to_pose`、`move_joints`：共享运动接口。
- `record_start`、`record_stop`、`record_replay`：规划层录制回放。

MuJoCo 的夹爪 `max_effort` 为 N，PA43 为 Nm，由配置 backend 决定。
完整架构和标定入口见 `../../docs/src_architecture_zh.md`。
