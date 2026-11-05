from glob import glob
from setuptools import find_packages, setup
name='rebotarm_planning'
setup(name=name, version='0.1.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages',['resource/'+name]),
                  ('share/'+name,['package.xml']),('share/'+name+'/config',glob('config/*.yaml'))],
      install_requires=['setuptools'], entry_points={'console_scripts':['task_server = rebotarm_planning.task_server:main']})
