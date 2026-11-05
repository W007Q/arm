from setuptools import find_packages, setup

setup(
    name="rebotarm_pingpong",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/rebotarm_pingpong"]),
        ("share/rebotarm_pingpong", ["package.xml", "README.md", "requirements.txt"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": ["pingpong = rebotarm_pingpong.cli:main"]},
)
