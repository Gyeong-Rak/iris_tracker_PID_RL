from setuptools import setup

package_name = 'RL_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gr',
    maintainer_email='rudkfr5978@naver.com',
    description='Quadrotor Controller using Reinforcement Learning',
    license='NONE',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detection = RL_controller.yolo_detector:main',
            'iris_controller = RL_controller.iris_controller:main',
            'iris_camera_controller_PID = RL_controller.iris_camera_controller_PID:main',
        ],
    },
)
