from glob import glob
from setuptools import find_packages, setup

package_name = "rebotarm_vision"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "requirements-yolo.txt"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "rgbd_detector = rebotarm_vision.rgbd_detector:main",
    ]},
)
