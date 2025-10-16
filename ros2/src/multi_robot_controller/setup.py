from setuptools import setup

package_name = 'multi_robot_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','juliacall'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Control two robots by subscribing to odometry and publishing velocity commands.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'controller_node = multi_robot_controller.controller_node:main',
        ],
    },
)
