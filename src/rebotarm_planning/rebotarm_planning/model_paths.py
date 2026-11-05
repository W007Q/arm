from pathlib import Path

def find_default_model():
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('rebotarm_description'))/'mjcf/grasp_scene.xml'
