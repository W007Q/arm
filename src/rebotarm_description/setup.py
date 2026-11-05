from glob import glob
from setuptools import find_packages, setup

package_name = "rebotarm_description"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/urdf", glob("urdf/*")),
        (f"share/{package_name}/meshes", glob("meshes/*")),
        (f"share/{package_name}/mjcf", glob("mjcf/*")),
        (f"share/{package_name}/config", glob("config/*")),
    ],
    install_requires=["setuptools", "PyYAML", "numpy", "scipy"],
    zip_safe=True,
)
