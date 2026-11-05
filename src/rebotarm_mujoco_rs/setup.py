from glob import glob
from setuptools import find_packages, setup

package_name = "rebotarm_mujoco_rs"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        # Legacy model paths are symlinks to description, not separate sources.
        (f"share/{package_name}/models", glob("models/*.xml")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="reBotArm Maintainers",
    maintainer_email="support@example.com",
    description="MuJoCo device adapter for the mixed DM/PA43 robot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mujoco_rs_sync = rebotarm_mujoco_rs.mujoco_sync:main",
            "rs_scene_camera = rebotarm_mujoco_rs.scene_camera:main",
            "rs_scene_detector = rebotarm_mujoco_rs.scene_detector:main",
            "rs_task_server = rebotarm_mujoco_rs.task_server:main",
            "rs_gravity_scan = rebotarm_mujoco_rs.gravity_scan:main",
            "rs_pingpong_controller = rebotarm_mujoco_rs.pingpong_controller:main",
        ],
    },
)
