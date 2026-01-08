import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'ros2_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='teamone',
    maintainer_email='teamone@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'nav = ros2_project.integrated_navigation:main',
            'nav2 = ros2_project.integrated_navigation_apf:main',
            'nav3 = ros2_project.integrated_navigation_teb:main',
            'dwa = ros2_project.dwa_navigation:main',
        ],
    },
)
