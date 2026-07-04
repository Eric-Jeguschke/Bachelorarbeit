from setuptools import find_packages, setup

package_name = 'speaker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['models/*']}, # colcon build kopiert ohne das hier nur .py-Dateien
    include_package_data=True, # aktiviert package_data
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ericj',
    maintainer_email='eric.jeguschke@gmail.com',
    description='Speaker package - TTS und Audio-Ausgabe Service',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'speaker_node = speaker.node:main',
        ],
    },
)
