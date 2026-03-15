from setuptools import find_packages, setup

package_name = 'image'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mypi4b',
    maintainer_email='mypi4b@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
	    'sign_detection = image.sign_detection_node:main',
	    'vehicle_actuator = image.vehicle_actuator_node:main',
        ],
    },
)
