from setuptools import setup

package_name = 'yolo_node'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zwt',
    maintainer_email='zwt@example.com',
    description='ROS2 YOLO Detection Node using OpenVINO',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = yolo_node.yolo_detector:main',
        ],
    },
)