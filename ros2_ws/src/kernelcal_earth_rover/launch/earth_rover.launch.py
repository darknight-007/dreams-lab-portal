"""Bring up the full earth_rover edge pipeline.

Three nodes share a single YAML config under ``config/earth_rover.yaml``:

* ``sq_fit_node`` -- consumes a clustered point cloud and emits SQs.
* ``attributor_node`` -- joins SQs with multispectral / hyperspectral
  / LiDAR-intensity feeds.
* ``observation_publisher_node`` -- POSTs attributed SQs to deepgis-xr.

Bring everything up with::

    ros2 launch kernelcal_earth_rover earth_rover.launch.py \
        config_file:=/path/to/your/earth_rover.yaml \
        endpoint_url:=https://deepgis.example/api/v1/observe/ \
        source_id:=earth_rover_07

Most overrides land cleanly on the rclpy ``parameter_yaml_file`` plumbing,
so per-deployment tuning lives in YAML, not in this Python launch file.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_PACKAGE = "kernelcal_earth_rover"


def generate_launch_description() -> LaunchDescription:
    default_config = os.path.join(
        get_package_share_directory(_PACKAGE),
        "config",
        "earth_rover.yaml",
    )

    args = [
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="YAML rclpy parameter file shared by all nodes.",
        ),
        DeclareLaunchArgument(
            "endpoint_url",
            default_value="http://localhost:8000/api/v1/observe/",
            description="deepgis-xr POST /api/v1/observe URL.",
        ),
        DeclareLaunchArgument(
            "source_id",
            default_value="earth_rover_01",
            description="Logical producer id used as the kernel-source key.",
        ),
        DeclareLaunchArgument(
            "session_id",
            default_value="",
            description="Optional mission/session label; defaults to source_id.",
        ),
        DeclareLaunchArgument(
            "auth_token",
            default_value="",
            description="Optional Bearer token for the observe endpoint.",
        ),
    ]

    config_file = LaunchConfiguration("config_file")

    nodes = [
        Node(
            package=_PACKAGE,
            executable="sq_fit_node",
            name="sq_fit_node",
            parameters=[config_file, {"stream_id": LaunchConfiguration("source_id")}],
            output="screen",
        ),
        Node(
            package=_PACKAGE,
            executable="attributor_node",
            name="attributor_node",
            parameters=[config_file],
            output="screen",
        ),
        Node(
            package=_PACKAGE,
            executable="observation_publisher_node",
            name="observation_publisher_node",
            parameters=[
                config_file,
                {
                    "endpoint_url": LaunchConfiguration("endpoint_url"),
                    "source_id": LaunchConfiguration("source_id"),
                    "session_id": LaunchConfiguration("session_id"),
                    "auth_token": LaunchConfiguration("auth_token"),
                },
            ],
            output="screen",
        ),
    ]

    return LaunchDescription([*args, *nodes])
