"""统一抓取入口：按后端选择仿真或真机，共用规划器和视觉节点。"""
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xacro
from rebotarm_description.parameters import load_mounts


def _setup(context):
    get=lambda k:LaunchConfiguration(k).perform(context)
    backend=get('backend')
    if backend not in ('mujoco','sdk','uart'):
        raise ValueError('backend must be mujoco, sdk or uart')
    share=lambda pkg:Path(get_package_share_directory(pkg))
    profile=get('task_config') or str(share('rebotarm_tasks')/'config'/f'pick_{backend}.yaml')
    from rebotarm_tasks.settings import load_settings
    settings=load_settings(profile)
    if settings.backend!=backend:raise ValueError('task profile backend differs from launch backend')
    ns=get('arm_namespace')
    scene=get('model_path') or str(share('rebotarm_description')/('mjcf/grasp_scene.xml' if backend=='mujoco' else 'mjcf/hardware_scene.xml'))
    model=share('rebotarm_description')/'urdf/rebotarm_with_d435.urdf.xacro'
    nodes=[]
    def include(pkg,name,args):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share(pkg)/'launch'/name)),launch_arguments=args.items())
    if backend=='mujoco':
        # 仿真提供手指关节实测值，由下方 gripper_state 统一为抓取完成反馈。
        nodes.append(include('rebotarm_mujoco_rs','mujoco_rs.launch.py',{
            'arm_namespace':ns,'model_path':scene,'authority_mode':'true','simulation_mode':'physics',
            'subscribe_commands':'true','provide_gripper_services':'true','gripper_kp':'120.0',
            'gravity_bias_scale':get('gravity_bias_scale'),
            'gripper_kd':'8.0','gripper_force_limit':str(settings.sim_force_limit_n),
            'use_viewer':get('use_viewer'),'enable_task_tools':'false'}))
        nodes.extend([
            Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
                 parameters=[{'robot_description': xacro.process_file(str(model), mappings={
                     'd435_internal_tf': 'true', 'd435_collision': get('d435_collision')}).toxml()}],
                 remappings=[('/joint_states', f'/{ns}/joint_states')]),
            Node(package='rebotarm_mujoco_rs',executable='rs_scene_camera',output='screen',parameters=[{'arm_namespace':ns,'model_path':scene,'camera_name':'d435_color','frame_id':'camera_color_optical_frame'}]),
            Node(package='rebotarm_vision',executable='rgbd_detector',name='rebotarm_rgbd_detector',output='screen',parameters=[{
                'arm_namespace':ns,
                'use_pose_topic':True,
                'image_topic':f'/{ns}/mujoco/d435/image_raw',
                'depth_topic':f'/{ns}/mujoco/d435/depth',
                'camera_info_topic':f'/{ns}/mujoco/d435/camera_info',
                'pose_topic':f'/{ns}/mujoco/d435/pose',
                # 保留仿真来源前缀，避免真机任务误用仿真检测结果。
                'detection_source':'mujoco_yolo_rgbd' if get('detector_backend') == 'yolo' else 'mujoco_rgbd',
                'detector_backend':get('detector_backend'),
                'yolo_model':get('yolo_model'),
                'yolo_confidence':float(get('yolo_confidence')),
                'yolo_iou':float(get('yolo_iou')),
                'yolo_imgsz':int(get('yolo_imgsz')),
                'yolo_device':ParameterValue(LaunchConfiguration('yolo_device'), value_type=str),
                'yolo_class_map':ParameterValue(LaunchConfiguration('yolo_class_map'), value_type=str),
            }])])
    else:
        if backend=='sdk':
            nodes.append(include('rebotarm_bringup','bringup.launch.py',{'arm_namespace':ns,'channel':get('channel'),'hardware_config':get('hardware_config') or str(share('rebotarm_description')/'config/rebotarm_hardware.yaml'),'provide_arm_services':'false','auto_enable':'false','safe_home_on_shutdown':'false','ee_frame_id':'gripper_tcp'}))
        else:
            if not get('port'):raise ValueError('backend uart requires port:=<STM32 serial device>')
            config=get('uart_config') or str(share('rebotarm_uart')/'config/uart_grasp.yaml')
            nodes.extend([
                Node(package='rebotarm_uart',executable='uart_bridge',name='rebotarm_uart_bridge',output='screen',parameters=[config,{'port':get('port'),'arm_namespace':ns,'gripper_open_rad':settings.motor_open_rad,'gripper_close_rad':settings.motor_close_rad,'gripper_finger_travel_m':settings.total_travel_m/2}]),
                Node(package='rebotarm_uart',executable='posvel_to_mit',name='rebotarm_posvel_to_mit',output='screen',parameters=[config,{'arm_namespace':ns,'sdk_urdf':str(model),'gravity_feedforward':ParameterValue(LaunchConfiguration('gravity_feedforward'), value_type=bool),'gripper_open_rad':settings.motor_open_rad,'gripper_close_rad':settings.motor_close_rad}]),
                Node(package='robot_state_publisher',executable='robot_state_publisher',output='screen',parameters=[{'robot_description':xacro.process_file(str(model), mappings={'d435_internal_tf':'false','d435_collision':get('d435_collision')}).toxml()}],remappings=[('/joint_states',f'/{ns}/joint_states')])])
        if get('start_camera').lower()=='true':
            nodes.append(include('rebotarm_vision','real_d435.launch.py',{
                'arm_namespace':ns,
                'detector_backend':get('detector_backend'),
                'detection_source':get('detection_source'),
                'yolo_model':get('yolo_model'),
                'yolo_confidence':get('yolo_confidence'),
                'yolo_iou':get('yolo_iou'),
                'yolo_imgsz':get('yolo_imgsz'),
                'yolo_device':get('yolo_device'),
                'yolo_class_map':get('yolo_class_map'),
            }))
    nodes.append(Node(package='rebotarm_tasks',executable='gripper_state',output='screen',parameters=[{'arm_namespace':ns,'config_path':profile}]))
    planner_params=[] if backend=='mujoco' else [get('planning_config') or str(share('rebotarm_planning')/'config/planning_hardware.yaml')]
    planner_params.append({'backend':backend,'arm_namespace':ns,'model_path':scene,'kinematics_backend':'mujoco' if backend=='mujoco' else 'sdk','sdk_urdf':str(model),'sdk_end_effector_frame':'gripper_tcp','tcp_offset':[0.,0.,0.] if backend!='mujoco' else load_mounts()['tcp']['xyz_m']})
    # 实体相机碰撞与是否启动图像流无关；两端使用同一个开关。
    planner_params.append({'d435_collision': ParameterValue(LaunchConfiguration('d435_collision'), value_type=bool)})
    nodes.append(Node(package='rebotarm_planning',executable='task_server',name='rebotarm_rs_task_server',output='screen',parameters=planner_params))
    if get('start_mcp').lower()=='true':
        nodes.append(Node(package='rebotarm_agent',executable='rebotarm_mcp_server',output='screen',arguments=['--namespace',ns,'--config',profile,'--motion-mode',get('motion_mode'),'--action-timeout-padding-sec','40']))
    return nodes


def generate_launch_description():
    defaults = {
        'backend': 'mujoco', 'arm_namespace': 'rebotarm',
        'task_config': '', 'model_path': '', 'planning_config': '',
        'hardware_config': '', 'uart_config': '', 'channel': '', 'port': '',
        'start_camera': 'true', 'start_mcp': 'false', 'motion_mode': 'locked',
        'use_viewer': 'false', 'gravity_bias_scale': '1.0', 'gravity_feedforward': 'false',
        # 正式抓取使用训练权重和 GPU；调参可直接通过同名启动参数覆盖。
        'detector_backend': 'yolo', 'detection_source': 'camera_rgbd',
        'yolo_model': 'models/best.pt', 'yolo_device': '0',
        'yolo_confidence': '0.45', 'yolo_iou': '0.50', 'yolo_imgsz': '640',
        'yolo_class_map': '{}',
    }
    return LaunchDescription(
        [DeclareLaunchArgument(k,default_value=v) for k,v in defaults.items()]
        + [DeclareLaunchArgument('d435_collision', default_value='true')]
        + [OpaqueFunction(function=_setup)])
