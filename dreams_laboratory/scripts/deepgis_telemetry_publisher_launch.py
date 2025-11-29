#!/usr/bin/env python3
"""
Launch file for DeepGIS Telemetry Publisher

Usage:
    ros2 launch deepgis_telemetry_publisher_launch.py
    
    # With custom parameters:
    ros2 launch deepgis_telemetry_publisher_launch.py \
        deepgis_api_url:=http://localhost:8000 \
        session_id:=my_custom_session
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from datetime import datetime


def generate_launch_description():
    """Generate launch description with configurable parameters."""
    
    # Default session ID with timestamp
    default_session_id = f'mavros_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'deepgis_api_url',
            default_value='http://192.168.0.186:8080',
            description='DeepGIS API base URL'
        ),
        
        DeclareLaunchArgument(
            'api_key',
            default_value='',
            description='Optional API key for authentication'
        ),
        
        DeclareLaunchArgument(
            'asset_name',
            default_value='MAVROS Vehicle',
            description='Name of the vehicle/asset'
        ),
        
        DeclareLaunchArgument(
            'session_id',
            default_value=default_session_id,
            description='Unique session identifier'
        ),
        
        DeclareLaunchArgument(
            'project_title',
            default_value='MAVROS Data Collection',
            description='Project title for this mission'
        ),
        
        DeclareLaunchArgument(
            'flight_mode',
            default_value='AUTO',
            description='Flight mode (MANUAL, AUTO, GUIDED, etc.)'
        ),
        
        DeclareLaunchArgument(
            'mission_type',
            default_value='Telemetry Collection',
            description='Type of mission being performed'
        ),
        
        DeclareLaunchArgument(
            'mavros_namespace',
            default_value='/mavros',
            description='MAVROS namespace'
        ),
        
        DeclareLaunchArgument(
            'publish_rate',
            default_value='1.0',
            description='Publishing rate in Hz'
        ),
        
        DeclareLaunchArgument(
            'batch_size',
            default_value='10',
            description='Number of samples to accumulate before batch upload'
        ),
        
        DeclareLaunchArgument(
            'enable_batch_mode',
            default_value='true',
            description='Enable batch mode (true/false)'
        ),
        
        # DeepGIS Telemetry Publisher Node
        Node(
            package='dreams_laboratory',  # Adjust to your package name
            executable='deepgis_telemetry_publisher.py',
            name='deepgis_telemetry_publisher',
            output='screen',
            parameters=[{
                'deepgis_api_url': LaunchConfiguration('deepgis_api_url'),
                'api_key': LaunchConfiguration('api_key'),
                'asset_name': LaunchConfiguration('asset_name'),
                'session_id': LaunchConfiguration('session_id'),
                'project_title': LaunchConfiguration('project_title'),
                'flight_mode': LaunchConfiguration('flight_mode'),
                'mission_type': LaunchConfiguration('mission_type'),
                'mavros_namespace': LaunchConfiguration('mavros_namespace'),
                'publish_rate': LaunchConfiguration('publish_rate'),
                'batch_size': LaunchConfiguration('batch_size'),
                'enable_batch_mode': LaunchConfiguration('enable_batch_mode'),
            }]
        ),
    ])

