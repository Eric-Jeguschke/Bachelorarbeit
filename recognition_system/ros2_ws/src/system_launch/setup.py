import os
from glob import glob
from setuptools import setup

package_name = "system_launch"

setup(
    name=package_name,
    version="0.0.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ericj",
    maintainer_email="eric.jeguschke@gmail.com",
    license="Apache-2.0",
)
