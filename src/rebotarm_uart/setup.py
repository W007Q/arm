from glob import glob
from setuptools import setup, find_packages
setup(name='rebotarm_uart',version='0.1.0',packages=find_packages(exclude=['test']),
      data_files=[('share/ament_index/resource_index/packages',['resource/rebotarm_uart']),
                  ('share/rebotarm_uart',['package.xml','README.md']),
                  ('share/rebotarm_uart/config',glob('config/*.yaml')),
                  ('share/rebotarm_uart/launch',glob('launch/*.launch.py'))],
      install_requires=['setuptools'],zip_safe=True,
      maintainer='reBotArm Maintainers',maintainer_email='support@example.com',
      description='Host ROS 2 UART MIT bridge for STM32',license='Apache-2.0',
      entry_points={'console_scripts':['uart_bridge = rebotarm_uart.bridge:main',
                                      'mit_hold = rebotarm_uart.hold:main',
                                      'posvel_to_mit = rebotarm_uart.posvel_to_mit:main']})
