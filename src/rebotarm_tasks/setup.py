from glob import glob
from setuptools import find_packages, setup

package_name = "rebotarm_tasks"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "pick = rebotarm_tasks.pick_cli:main",
        "gripper_state = rebotarm_tasks.gripper_state_node:main",
        "pingpong_controller = rebotarm_tasks.pingpong_controller:main",
    ]},
)
