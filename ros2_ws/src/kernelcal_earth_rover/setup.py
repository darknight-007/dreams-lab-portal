"""Colcon (ament_python) build for the earth_rover edge pipeline.

The package layout mirrors the dataflow on the rover:

    sensor_msgs/PointCloud2  --> sq_fit_node      --> Superquadric stream
    sensor_msgs/Image (RGB / multispectral)
    sensor_msgs/Image (UV-VIS-NIR)               -+
    LiDAR-intensity scalar field                  +-> attributor_node
                                                  |   (PropertyId + Spectrum)
                                                  v
                                          observation_publisher_node
                                                  |
                                                  v
                              POST /api/v1/observe (deepgis-xr)

Each node is registered as a console-script entrypoint so it can be
run with ``ros2 run kernelcal_earth_rover <node>`` after a ``colcon
build && source install/setup.bash`` from ``ros2_ws``.
"""

from setuptools import setup
from glob import glob

package_name = "kernelcal_earth_rover"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dreams Lab",
    maintainer_email="dreams-lab@asu.edu",
    description=(
        "earth_rover edge pipeline: superquadric fitting, multispectral "
        "+ LiDAR-intensity attribution, and observation publishing to "
        "deepgis-xr."
    ),
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sq_fit_node = kernelcal_earth_rover.sq_fit_node:main",
            "attributor_node = kernelcal_earth_rover.attributor_node:main",
            "observation_publisher_node = kernelcal_earth_rover.observation_publisher_node:main",
        ],
    },
)
