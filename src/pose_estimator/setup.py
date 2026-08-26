from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pose_estimator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zwt',
    maintainer_email='zwt@todo.com',
    description='目标物体6DoF位姿估计',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pose_estimation_node = pose_estimator.pose_estimation_node:main',
        ],
    },
)