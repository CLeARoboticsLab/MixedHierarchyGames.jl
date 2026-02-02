from setuptools import setup
from setuptools import find_packages

package_name = 'mixed_hierarchy_games'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'julia', 'roslibpy'],
    zip_safe=True,
    maintainer='Tianyu Qiu',
    maintainer_email='tianyuqiu@utexas.edu',
    description='Mixed hierarchy games controller for multi-robot systems.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'controller_node = mixed_hierarchy_games.controller_node:main',
        ],
    },
)
