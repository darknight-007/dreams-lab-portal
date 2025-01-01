from setuptools import setup

package_name = 'quiz_interface'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dreams Lab',
    maintainer_email='your@email.com',
    description='Quiz interface for PX4 SITL',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'px4_bridge = quiz_interface.px4_bridge:main',
            'quiz_validator = quiz_interface.quiz_validator:main',
        ],
    },
) 