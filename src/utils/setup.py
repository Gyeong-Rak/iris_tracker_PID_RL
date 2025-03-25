from setuptools import setup

package_name = 'utils'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = utils.YOLOv8:main',
            'fastSAM_detector = utils.FastSAM:main',
            'RL_online_train = utils.RL_online_train:main',
        ],
    },
)
