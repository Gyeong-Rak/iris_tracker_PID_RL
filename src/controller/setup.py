from setuptools import setup

package_name = 'controller'

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
    description='Quadrotor Controller',
    license='NONE',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'iris_controller = controller.iris_controller:main',
            'iris_camera_controller_PID = controller.iris_camera_controller_PID_setpoint:main',
        ],
    },
)
