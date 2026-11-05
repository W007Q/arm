import os
import copy


def hardware_controller_namespace(parameters, arm_namespace, trajectory_controller=''):
    """让 MoveIt 的标准轨迹 Action 与设备命名空间一致。

    YAML 提供默认控制器配置；启动时选择地址与对应的执行计时方式。
    返回副本，避免修改被 RViz 或其他 launch 共用的原始参数。
    """
    namespace = str(arm_namespace).strip('/')
    if not namespace:
        raise ValueError('arm_namespace must not be empty')
    suffix = str(trajectory_controller).strip('/')
    if suffix not in ('', 'arm_controller'):
        raise ValueError('trajectory_controller must be empty or arm_controller')
    controller_name = f'{namespace}/{suffix}' if suffix else namespace
    output = copy.deepcopy(parameters)
    manager = output['moveit_simple_controller_manager']
    old_names = manager['controller_names']
    if len(old_names) != 1:
        raise ValueError('hardware launch requires exactly one arm trajectory controller')
    controller = manager.pop(old_names[0])
    manager['controller_names'] = [controller_name]
    manager[controller_name] = controller
    if not suffix:
        # task_server 按实测进度与限速延长轨迹，原始规划时长不是执行期限。
        # 由后端继续检查反馈、碰撞、跟踪误差及 4 秒无进展；JTC 保留 YAML 计时。
        output['trajectory_execution']['execution_duration_monitoring'] = False
    return output


def moveit_parameters(moveit_config):
    parameters = moveit_config.to_dict()
    ompl = parameters.setdefault("ompl", {})
    ompl["planning_plugin"] = "ompl_interface/OMPLPlanner"

    if os.environ.get("ROS_DISTRO") == "humble":
        ompl["request_adapters"] = " ".join(
            [
                "default_planner_request_adapters/AddTimeOptimalParameterization",
                "default_planner_request_adapters/ResolveConstraintFrames",
                "default_planner_request_adapters/FixWorkspaceBounds",
                "default_planner_request_adapters/FixStartStateBounds",
                "default_planner_request_adapters/FixStartStateCollision",
            ]
        )
        ompl.pop("response_adapters", None)

    return parameters
