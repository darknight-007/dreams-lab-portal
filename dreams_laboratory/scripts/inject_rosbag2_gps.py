#!/usr/bin/env python3
"""
Script to extract GPS and Odometry data from rosbag2 files and inject them to DeepGIS via telemetry API.

This script reads rosbag2 files, extracts GPS messages from the /fmu/out/vehicle_gps_position topic
and odometry messages from common odometry topics, and posts them to the DeepGIS telemetry API.

Usage:
    python scripts/inject_rosbag2_gps.py <rosbag2_path> [--url API_URL] [--session-id SESSION_ID] [--gps-topic TOPIC] [--odom-topic TOPIC]

Requirements:
    - rosbag2_py (pip install rosbag2_py)
    - rclpy (for ROS2 message types)
    - requests (for API calls)

Example:
    python scripts/inject_rosbag2_gps.py /mnt/tesseract-store/trike-backup/rosbag2_2023_11_17-09_54_09
    python scripts/inject_rosbag2_gps.py /path/to/rosbag --url http://localhost:8000/api/telemetry --session-id my_session
    python scripts/inject_rosbag2_gps.py /path/to/rosbag --odom-topic /fmu/out/vehicle_local_position
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import argparse
import requests
import json

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
except ImportError as e:
    print(f"Error: Missing required ROS2 library: {e}")
    print("Please install: pip install rosbag2_py rclpy")
    sys.exit(1)

# ============================================================================
# CONFIGURATION - Set your API base URL here
# ============================================================================
# Default API base URL (can be overridden via command-line argument)
API_BASE_URL = "http://192.168.0.186:8080/api/telemetry"

# Alternative URLs (uncomment to use):
# API_BASE_URL = "http://localhost:8000/api/telemetry"  # Local development
# API_BASE_URL = "https://deepgis.org/api/telemetry"   # Production HTTPS
# API_BASE_URL = "http://172.20.0.10/api/telemetry"    # Docker internal network

# Default GPS topic name
DEFAULT_GPS_TOPIC = "/fmu/out/vehicle_gps_position"

# Default Odometry topic names (will try multiple)
DEFAULT_ODOM_TOPICS = [
    "/fmu/out/vehicle_local_position",  # PX4 local position
    "/odom",                             # Standard ROS odometry
    "/odometry/filtered",                # robot_localization output
    "/t265/odom/sample",                 # Intel RealSense T265
]

# PX4 VehicleGPSPosition message type
PX4_GPS_MSG_TYPE = "px4_msgs/msg/VehicleGPSPosition"

# PX4 VehicleLocalPosition message type
PX4_LOCAL_POSITION_MSG_TYPE = "px4_msgs/msg/VehicleLocalPosition"

# Standard ROS odometry message type
ROS_ODOM_MSG_TYPE = "nav_msgs/msg/Odometry"


def discover_topics_in_rosbag(rosbag_path):
    """
    Scan rosbag and discover all available topics with their types.
    
    Returns dict with topic categories:
    {
        'gps': [{'name': '/topic', 'type': 'type', 'score': int}],
        'odometry': [{'name': '/topic', 'type': 'type', 'score': int}],
        'all': [{'name': '/topic', 'type': 'type'}]
    }
    """
    rosbag_path = Path(rosbag_path)
    
    if not rosbag_path.exists() or not rosbag_path.is_dir():
        return {'gps': [], 'odometry': [], 'all': []}
    
    try:
        # Create storage options
        storage_options = rosbag2_py.StorageOptions(
            uri=str(rosbag_path),
            storage_id='sqlite3'
        )
        
        # Create converter options
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        # Open rosbag
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get all topics
        topic_types = reader.get_all_topics_and_types()
        
        all_topics = []
        gps_topics = []
        odom_topics = []
        
        # GPS topic detection keywords and scoring
        gps_keywords = {
            'gps': 10,
            'navsat': 10,
            'fix': 8,
            'gnss': 10,
            'position': 3,
            'vehicle_gps_position': 15,
            'sensor_gps': 12,
        }
        
        gps_type_keywords = {
            'GPS': 15,
            'NavSat': 15,
            'Fix': 10,
            'GNSS': 15,
        }
        
        # Odometry topic detection keywords and scoring
        odom_keywords = {
            'odom': 10,
            'odometry': 10,
            'local_position': 12,
            'vehicle_local_position': 15,
            'pose': 5,
            't265': 8,
            'vio': 8,
            'visual_odometry': 10,
        }
        
        odom_type_keywords = {
            'Odometry': 15,
            'LocalPosition': 12,
            'VehicleLocalPosition': 15,
            'Pose': 5,
        }
        
        for topic_metadata in topic_types:
            topic_name = topic_metadata.name
            topic_type = topic_metadata.type
            
            all_topics.append({
                'name': topic_name,
                'type': topic_type
            })
            
            # Score GPS topics
            gps_score = 0
            topic_lower = topic_name.lower()
            for keyword, score in gps_keywords.items():
                if keyword in topic_lower:
                    gps_score += score
            
            for keyword, score in gps_type_keywords.items():
                if keyword in topic_type:
                    gps_score += score
            
            if gps_score > 0:
                gps_topics.append({
                    'name': topic_name,
                    'type': topic_type,
                    'score': gps_score
                })
            
            # Score odometry topics
            odom_score = 0
            for keyword, score in odom_keywords.items():
                if keyword in topic_lower:
                    odom_score += score
            
            for keyword, score in odom_type_keywords.items():
                if keyword in topic_type:
                    odom_score += score
            
            if odom_score > 0:
                odom_topics.append({
                    'name': topic_name,
                    'type': topic_type,
                    'score': odom_score
                })
        
        # Sort by score (highest first)
        gps_topics.sort(key=lambda x: x['score'], reverse=True)
        odom_topics.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'gps': gps_topics,
            'odometry': odom_topics,
            'all': all_topics
        }
        
    except Exception as e:
        print(f"⚠ Warning: Could not discover topics: {e}")
        return {'gps': [], 'odometry': [], 'all': []}


def generate_session_id_from_rosbag(rosbag_path):
    """Generate a session ID from rosbag path"""
    rosbag_name = Path(rosbag_path).name
    # Extract date/time from rosbag name if possible (e.g., rosbag2_2023_11_17-09_54_09)
    if 'rosbag2_' in rosbag_name:
        parts = rosbag_name.replace('rosbag2_', '').split('-')
        if len(parts) >= 2:
            date_part = parts[0].replace('_', '')
            time_part = parts[1].replace('_', '')
            return f"rosbag_{date_part}_{time_part}"
    # Fallback to using the directory name
    return f"rosbag_{rosbag_name}"


def create_session_via_api(base_url, session_id, asset_name='RV Karin Valentine', rosbag_path=None):
    """
    Create session via REST API. Works on any remote machine.
    Returns (success, created) tuple.
    """
    try:
        notes = f'GPS data extracted from rosbag2 file'
        if rosbag_path:
            notes += f': {Path(rosbag_path).name}'
        
        session_data = {
            'session_id': session_id,
            'asset_name': asset_name,
            'project_title': 'Rosbag2 Import',
            'flight_mode': 'AUTO',
            'mission_type': 'Data Import',
            'notes': notes
        }
        
        url = f"{base_url}/session/create/"
        response = requests.post(
            url,
            json=session_data,
            headers={'Content-Type': 'application/json'},
            timeout=10.0
        )
        
        # Try to parse JSON response
        try:
            if response.content:
                result = response.json()
            else:
                result = {}
        except (ValueError, json.JSONDecodeError) as e:
            # Response is not JSON - might be HTML error page
            print(f"   ⚠ Non-JSON response received (HTTP {response.status_code})")
            print(f"   Response preview: {response.text[:200] if response.text else 'Empty response'}")
            print(f"   URL attempted: {url}")
            return False, False
        
        if response.status_code == 201:
            return True, result.get('created', True)
        elif response.status_code == 200:
            # Session already exists
            return True, False
        else:
            print(f"   ⚠ API error (HTTP {response.status_code}): {result}")
            return False, False
            
    except requests.exceptions.ConnectionError as e:
        print(f"   ⚠ Connection error - cannot reach API server: {e}")
        return False, False
    except requests.exceptions.Timeout:
        print(f"   ⚠ Request timeout")
        return False, False
    except Exception as e:
        print(f"   ⚠ Error creating session via API: {e}")
        import traceback
        if '--debug' in sys.argv or '-d' in sys.argv:
            print(f"   Traceback: {traceback.format_exc()}")
        return False, False


def read_rosbag2_gps(rosbag_path, topic_name=DEFAULT_GPS_TOPIC):
    """
    Read GPS messages from rosbag2 file.
    
    Returns list of GPS data dictionaries ready for API posting.
    """
    rosbag_path = Path(rosbag_path)
    
    if not rosbag_path.exists():
        raise FileNotFoundError(f"Rosbag path does not exist: {rosbag_path}")
    
    if not rosbag_path.is_dir():
        raise ValueError(f"Rosbag path must be a directory: {rosbag_path}")
    
    print(f"Reading rosbag2 from: {rosbag_path}")
    print(f"Looking for GPS topic: {topic_name}")
    
    # Create storage options
    storage_options = rosbag2_py.StorageOptions(
        uri=str(rosbag_path),
        storage_id='sqlite3'
    )
    
    # Create converter options
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    # Open rosbag
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    
    # Find GPS topic
    gps_topic_info = None
    for topic_metadata in topic_types:
        if topic_metadata.name == topic_name:
            gps_topic_info = topic_metadata
            break
    
    if not gps_topic_info:
        print(f"\n⚠ Warning: GPS topic '{topic_name}' not found in rosbag.")
        print(f"Available topics:")
        for topic_metadata in topic_types:
            print(f"  - {topic_metadata.name} ({topic_metadata.type})")
        raise ValueError(f"GPS topic '{topic_name}' not found in rosbag")
    
    print(f"Found GPS topic: {gps_topic_info.name} (type: {gps_topic_info.type})")
    
    # Get message type
    try:
        msg_type = get_message(gps_topic_info.type)
    except Exception as e:
        print(f"⚠ Warning: Could not load message type '{gps_topic_info.type}'")
        print(f"Trying alternative: {PX4_GPS_MSG_TYPE}")
        try:
            msg_type = get_message(PX4_GPS_MSG_TYPE)
        except Exception as e2:
            raise ValueError(f"Could not load message type. Error: {e2}")
    
    # Read messages
    gps_points = []
    message_count = 0
    
    # Set filter to only read GPS topic
    reader.set_read_filter(rosbag2_py.StorageFilter(topics=[topic_name]))
    
    print("\nReading GPS messages from rosbag...")
    
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        if topic != topic_name:
            continue
        
        try:
            # Deserialize message
            msg = deserialize_message(data, msg_type)
            
            # Convert to API format
            # PX4 VehicleGPSPosition uses:
            # - lat/lon in 1e7 degrees (int32)
            # - alt in mm (int32)
            # - hdop, eph, epv in cm (uint16)
            
            # Convert timestamp (nanoseconds since epoch)
            # Rosbag2 timestamps are in nanoseconds since epoch
            timestamp_sec = timestamp / 1e9
            timestamp_dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
            
            # Use message timestamp_usec if available, otherwise use bag timestamp
            if hasattr(msg, 'timestamp') and msg.timestamp > 0:
                timestamp_usec = int(msg.timestamp)  # Already in microseconds from PX4
            else:
                timestamp_usec = int(timestamp / 1000)  # Convert ns to microseconds
            
            # Extract GPS data
            gps_data = {
                'timestamp': timestamp_dt.isoformat(),
                'timestamp_usec': timestamp_usec,
                'latitude': float(msg.lat) / 1e7 if hasattr(msg, 'lat') else None,
                'longitude': float(msg.lon) / 1e7 if hasattr(msg, 'lon') else None,
                'altitude': float(msg.alt) / 1000.0 if hasattr(msg, 'alt') else None,  # mm to m
                'fix_type': int(msg.fix_type) if hasattr(msg, 'fix_type') else None,
                'satellites_visible': int(msg.satellites_visible) if hasattr(msg, 'satellites_visible') else None,
                'satellites_used': int(msg.satellites_used) if hasattr(msg, 'satellites_used') else None,
                'hdop': float(msg.hdop) / 100.0 if hasattr(msg, 'hdop') and msg.hdop > 0 else None,
                'vdop': float(msg.vdop) / 100.0 if hasattr(msg, 'vdop') and msg.vdop > 0 else None,
                'pdop': float(msg.pdop) / 100.0 if hasattr(msg, 'pdop') and msg.pdop > 0 else None,
                'eph': float(msg.eph) / 100.0 if hasattr(msg, 'eph') and msg.eph > 0 else None,
                'epv': float(msg.epv) / 100.0 if hasattr(msg, 'epv') and msg.epv > 0 else None,
                's_variance_m_s': float(msg.s_variance_m_s) if hasattr(msg, 's_variance_m_s') else None,
                'vel_n_m_s': float(msg.vel_n_m_s) if hasattr(msg, 'vel_n_m_s') else None,
                'vel_e_m_s': float(msg.vel_e_m_s) if hasattr(msg, 'vel_e_m_s') else None,
                'vel_d_m_s': float(msg.vel_d_m_s) if hasattr(msg, 'vel_d_m_s') else None,
                'vel_m_s': float(msg.vel_m_s) if hasattr(msg, 'vel_m_s') else None,
                'cog_rad': float(msg.cog_rad) if hasattr(msg, 'cog_rad') else None,
                'time_utc_usec': int(msg.time_utc_usec) if hasattr(msg, 'time_utc_usec') and msg.time_utc_usec > 0 else None,
                'noise_per_ms': int(msg.noise_per_ms) if hasattr(msg, 'noise_per_ms') else None,
                'jamming_indicator': int(msg.jamming_indicator) if hasattr(msg, 'jamming_indicator') else None,
                'jamming_state': int(msg.jamming_state) if hasattr(msg, 'jamming_state') else None,
                'device_id': int(msg.device_id) if hasattr(msg, 'device_id') else None,
            }
            
            # Filter out None values for optional fields (but keep required ones)
            gps_data_clean = {k: v for k, v in gps_data.items() if v is not None or k in ['timestamp', 'timestamp_usec', 'latitude', 'longitude', 'altitude', 'fix_type']}
            
            gps_points.append(gps_data_clean)
            message_count += 1
            
            if message_count % 100 == 0:
                print(f"  Read {message_count} GPS messages...", end='\r', flush=True)
                
        except Exception as e:
            print(f"\n⚠ Warning: Error deserializing message at timestamp {timestamp}: {e}")
            continue
    
    print(f"\n✓ Read {len(gps_points)} GPS messages from rosbag")
    
    return gps_points


def read_rosbag2_odometry(rosbag_path, topic_names=None):
    """
    Read odometry messages from rosbag2 file.
    
    Supports multiple odometry message types:
    - px4_msgs/msg/VehicleLocalPosition (PX4 local position)
    - nav_msgs/msg/Odometry (standard ROS odometry)
    
    Returns list of odometry data dictionaries ready for API posting.
    """
    rosbag_path = Path(rosbag_path)
    
    if not rosbag_path.exists():
        raise FileNotFoundError(f"Rosbag path does not exist: {rosbag_path}")
    
    if not rosbag_path.is_dir():
        raise ValueError(f"Rosbag path must be a directory: {rosbag_path}")
    
    # Use default topics if none specified
    if topic_names is None:
        topic_names = DEFAULT_ODOM_TOPICS
    elif isinstance(topic_names, str):
        topic_names = [topic_names]
    
    print(f"Reading rosbag2 from: {rosbag_path}")
    print(f"Looking for odometry topics: {', '.join(topic_names)}")
    
    # Create storage options
    storage_options = rosbag2_py.StorageOptions(
        uri=str(rosbag_path),
        storage_id='sqlite3'
    )
    
    # Create converter options
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    # Open rosbag
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    
    # Find odometry topic
    odom_topic_info = None
    for topic_name in topic_names:
        for topic_metadata in topic_types:
            if topic_metadata.name == topic_name:
                odom_topic_info = topic_metadata
                break
        if odom_topic_info:
            break
    
    if not odom_topic_info:
        print(f"\n⚠ Warning: No odometry topic found in rosbag.")
        print(f"Looked for: {', '.join(topic_names)}")
        print(f"Available topics:")
        for topic_metadata in topic_types:
            print(f"  - {topic_metadata.name} ({topic_metadata.type})")
        return []
    
    print(f"Found odometry topic: {odom_topic_info.name} (type: {odom_topic_info.type})")
    
    # Get message type
    try:
        msg_type = get_message(odom_topic_info.type)
    except Exception as e:
        print(f"⚠ Warning: Could not load message type '{odom_topic_info.type}'")
        print(f"Error: {e}")
        return []
    
    # Read messages
    odom_points = []
    message_count = 0
    
    # Set filter to only read odometry topic
    reader.set_read_filter(rosbag2_py.StorageFilter(topics=[odom_topic_info.name]))
    
    print("\nReading odometry messages from rosbag...")
    
    # Determine message type category
    is_px4_local = 'VehicleLocalPosition' in odom_topic_info.type
    is_ros_odom = 'Odometry' in odom_topic_info.type
    
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        if topic != odom_topic_info.name:
            continue
        
        try:
            # Deserialize message
            msg = deserialize_message(data, msg_type)
            
            # Convert timestamp
            timestamp_sec = timestamp / 1e9
            timestamp_dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
            
            # Extract timestamp from message if available
            if hasattr(msg, 'timestamp') and msg.timestamp > 0:
                timestamp_usec = int(msg.timestamp)  # Already in microseconds from PX4
            else:
                timestamp_usec = int(timestamp / 1000)  # Convert ns to microseconds
            
            # Parse based on message type
            odom_data = {
                'timestamp': timestamp_dt.isoformat(),
                'timestamp_usec': timestamp_usec,
            }
            
            if is_px4_local:
                # PX4 VehicleLocalPosition format
                # Positions in NED frame
                odom_data.update({
                    'x': float(msg.x) if hasattr(msg, 'x') else None,
                    'y': float(msg.y) if hasattr(msg, 'y') else None,
                    'z': float(msg.z) if hasattr(msg, 'z') else None,
                    'vx': float(msg.vx) if hasattr(msg, 'vx') else None,
                    'vy': float(msg.vy) if hasattr(msg, 'vy') else None,
                    'vz': float(msg.vz) if hasattr(msg, 'vz') else None,
                    'heading': float(msg.heading) if hasattr(msg, 'heading') else None,
                    'heading_rate': float(msg.heading_rate) if hasattr(msg, 'heading_rate') else None,
                    'xy_valid': bool(msg.xy_valid) if hasattr(msg, 'xy_valid') else None,
                    'z_valid': bool(msg.z_valid) if hasattr(msg, 'z_valid') else None,
                    'v_xy_valid': bool(msg.v_xy_valid) if hasattr(msg, 'v_xy_valid') else None,
                    'v_z_valid': bool(msg.v_z_valid) if hasattr(msg, 'v_z_valid') else None,
                    'ref_lat': float(msg.ref_lat) if hasattr(msg, 'ref_lat') else None,
                    'ref_lon': float(msg.ref_lon) if hasattr(msg, 'ref_lon') else None,
                    'ref_alt': float(msg.ref_alt) if hasattr(msg, 'ref_alt') else None,
                    'eph': float(msg.eph) if hasattr(msg, 'eph') else None,
                    'epv': float(msg.epv) if hasattr(msg, 'epv') else None,
                    'evh': float(msg.evh) if hasattr(msg, 'evh') else None,
                    'evv': float(msg.evv) if hasattr(msg, 'evv') else None,
                })
            
            elif is_ros_odom:
                # Standard ROS nav_msgs/Odometry format
                # Position from pose
                if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                    pose = msg.pose.pose
                    odom_data.update({
                        'x': float(pose.position.x) if hasattr(pose, 'position') else None,
                        'y': float(pose.position.y) if hasattr(pose, 'position') else None,
                        'z': float(pose.position.z) if hasattr(pose, 'position') else None,
                    })
                    
                    # Extract heading from quaternion if available
                    if hasattr(pose, 'orientation'):
                        # Calculate heading (yaw) from quaternion
                        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
                        import math
                        q = pose.orientation
                        heading = math.atan2(
                            2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                        )
                        odom_data['heading'] = float(heading)
                    
                    # Covariance (9 elements - 3x3 matrix for x, y, z)
                    if hasattr(msg.pose, 'covariance'):
                        cov = list(msg.pose.covariance)
                        if len(cov) >= 36:  # 6x6 covariance matrix
                            # Extract position covariance (x, y, z from 6x6)
                            odom_data['position_covariance'] = [
                                cov[0], cov[1], cov[2],      # Row 0 (x)
                                cov[6], cov[7], cov[8],      # Row 1 (y)
                                cov[12], cov[13], cov[14]    # Row 2 (z)
                            ]
                
                # Velocity from twist
                if hasattr(msg, 'twist') and hasattr(msg.twist, 'twist'):
                    twist = msg.twist.twist
                    odom_data.update({
                        'vx': float(twist.linear.x) if hasattr(twist, 'linear') else None,
                        'vy': float(twist.linear.y) if hasattr(twist, 'linear') else None,
                        'vz': float(twist.linear.z) if hasattr(twist, 'linear') else None,
                        'heading_rate': float(twist.angular.z) if hasattr(twist, 'angular') else None,
                    })
                    
                    # Velocity covariance
                    if hasattr(msg.twist, 'covariance'):
                        cov = list(msg.twist.covariance)
                        if len(cov) >= 36:  # 6x6 covariance matrix
                            odom_data['velocity_covariance'] = [
                                cov[0], cov[1], cov[2],      # Row 0 (vx)
                                cov[6], cov[7], cov[8],      # Row 1 (vy)
                                cov[12], cov[13], cov[14]    # Row 2 (vz)
                            ]
            
            # Filter out None values
            odom_data_clean = {k: v for k, v in odom_data.items() if v is not None}
            
            # Only add if we have valid position data
            if 'x' in odom_data_clean and 'y' in odom_data_clean and 'z' in odom_data_clean:
                odom_points.append(odom_data_clean)
                message_count += 1
                
                if message_count % 100 == 0:
                    print(f"  Read {message_count} odometry messages...", end='\r', flush=True)
                    
        except Exception as e:
            print(f"\n⚠ Warning: Error deserializing message at timestamp {timestamp}: {e}")
            continue
    
    print(f"\n✓ Read {len(odom_points)} odometry messages from rosbag")
    
    return odom_points


def post_gps_raw_via_api(base_url, session_id, gps_points):
    """Post GPS raw telemetry data via REST API"""
    results = {
        'gps_raw': {'success': 0, 'errors': []}
    }
    
    print(f"\nPosting GPS raw telemetry data via API for {len(gps_points)} points...")
    
    for i, gps_data in enumerate(gps_points):
        if (i + 1) % 50 == 0:
            print(f"  Processing point {i + 1}/{len(gps_points)}...")
        
        # Add session_id to the data
        gps_data['session_id'] = session_id
        
        try:
            response = requests.post(
                f"{base_url}/gps-fix-raw/",
                json=gps_data,
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            if response.status_code == 201:
                results['gps_raw']['success'] += 1
            else:
                error_data = response.json() if response.content else {'message': 'Unknown error'}
                results['gps_raw']['errors'].append({
                    'index': i,
                    'status': response.status_code,
                    'error': error_data
                })
                if len(results['gps_raw']['errors']) <= 5:  # Only print first 5 errors
                    print(f"\n  ⚠ Error at point {i + 1}: {error_data}")
        except Exception as e:
            results['gps_raw']['errors'].append({
                'index': i,
                'error': str(e)
            })
            if len(results['gps_raw']['errors']) <= 5:
                print(f"\n  ⚠ Exception at point {i + 1}: {e}")
    
    return results


def post_local_position_odom_via_api(base_url, session_id, odom_points):
    """Post local position odometry telemetry data via REST API"""
    results = {
        'local_position_odom': {'success': 0, 'errors': []}
    }
    
    print(f"\nPosting local position odometry data via API for {len(odom_points)} points...")
    
    for i, odom_data in enumerate(odom_points):
        if (i + 1) % 50 == 0:
            print(f"  Processing point {i + 1}/{len(odom_points)}...")
        
        # Add session_id to the data
        odom_data['session_id'] = session_id
        
        try:
            response = requests.post(
                f"{base_url}/local-position-odom/",
                json=odom_data,
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            if response.status_code == 201:
                results['local_position_odom']['success'] += 1
            else:
                error_data = response.json() if response.content else {'message': 'Unknown error'}
                results['local_position_odom']['errors'].append({
                    'index': i,
                    'status': response.status_code,
                    'error': error_data
                })
                if len(results['local_position_odom']['errors']) <= 5:  # Only print first 5 errors
                    print(f"\n  ⚠ Error at point {i + 1}: {error_data}")
        except Exception as e:
            results['local_position_odom']['errors'].append({
                'index': i,
                'error': str(e)
            })
            if len(results['local_position_odom']['errors']) <= 5:
                print(f"\n  ⚠ Exception at point {i + 1}: {e}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Extract GPS and Odometry data from rosbag2 and inject to DeepGIS via telemetry API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Extract both GPS and odometry (auto-detect topics)
  python {sys.argv[0]} /path/to/rosbag2_directory
  
  # Specify custom GPS topic
  python {sys.argv[0]} /path/to/rosbag2 --gps-topic /fmu/out/vehicle_gps_position
  
  # Specify custom odometry topic
  python {sys.argv[0]} /path/to/rosbag2 --odom-topic /fmu/out/vehicle_local_position
  
  # Only extract GPS data (skip odometry)
  python {sys.argv[0]} /path/to/rosbag2 --skip-odom
  
  # Only extract odometry data (skip GPS)
  python {sys.argv[0]} /path/to/rosbag2 --skip-gps
  
  # Custom API URL and session ID
  python {sys.argv[0]} /path/to/rosbag2 --url http://localhost:8000/api/telemetry --session-id my_session
        """
    )
    parser.add_argument(
        'rosbag_path',
        type=str,
        help='Path to rosbag2 directory'
    )
    parser.add_argument(
        '--url',
        type=str,
        default=API_BASE_URL,
        help=f'API base URL (default: {API_BASE_URL})'
    )
    parser.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Session ID to use. If not provided, will be generated from rosbag name.'
    )
    parser.add_argument(
        '--gps-topic',
        type=str,
        default=DEFAULT_GPS_TOPIC,
        help=f'GPS topic name (default: {DEFAULT_GPS_TOPIC})'
    )
    parser.add_argument(
        '--odom-topic',
        type=str,
        default=None,
        help=f'Odometry topic name. If not specified, will auto-detect from: {", ".join(DEFAULT_ODOM_TOPICS)}'
    )
    parser.add_argument(
        '--skip-gps',
        action='store_true',
        help='Skip GPS data extraction'
    )
    parser.add_argument(
        '--skip-odom',
        action='store_true',
        help='Skip odometry data extraction'
    )
    parser.add_argument(
        '--asset-name',
        type=str,
        default='RV Karin Valentine',
        help='Asset name for the session (default: RV Karin Valentine)'
    )
    parser.add_argument(
        '--skip-session-create',
        action='store_true',
        help='Skip session creation (assume session already exists)'
    )
    
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')  # Remove trailing slash if present
    rosbag_path = Path(args.rosbag_path)
    
    print("=" * 70)
    print("Rosbag2 GPS & Odometry Data Extraction and Injection")
    print("=" * 70)
    
    # Discover topics in rosbag
    print("\n1. Discovering topics in rosbag...")
    discovered = discover_topics_in_rosbag(rosbag_path)
    
    print(f"   Found {len(discovered['all'])} total topics")
    
    if discovered['gps']:
        print(f"   Found {len(discovered['gps'])} GPS-related topics:")
        for topic in discovered['gps'][:3]:  # Show top 3
            print(f"     • {topic['name']} ({topic['type']}) [score: {topic['score']}]")
        if len(discovered['gps']) > 3:
            print(f"     ... and {len(discovered['gps']) - 3} more")
    else:
        print("   ⚠ No GPS-related topics found")
    
    if discovered['odometry']:
        print(f"   Found {len(discovered['odometry'])} odometry-related topics:")
        for topic in discovered['odometry'][:3]:  # Show top 3
            print(f"     • {topic['name']} ({topic['type']}) [score: {topic['score']}]")
        if len(discovered['odometry']) > 3:
            print(f"     ... and {len(discovered['odometry']) - 3} more")
    else:
        print("   ⚠ No odometry-related topics found")
    
    # Determine which GPS topic to use
    gps_topic_to_use = None
    if not args.skip_gps:
        if args.gps_topic:
            gps_topic_to_use = args.gps_topic
            print(f"\n   Using user-specified GPS topic: {gps_topic_to_use}")
        elif discovered['gps']:
            gps_topic_to_use = discovered['gps'][0]['name']
            print(f"\n   Auto-selected GPS topic: {gps_topic_to_use} (highest score)")
        else:
            gps_topic_to_use = DEFAULT_GPS_TOPIC
            print(f"\n   Using default GPS topic: {gps_topic_to_use} (may not exist)")
    
    # Determine which odometry topic to use
    odom_topic_to_use = None
    if not args.skip_odom:
        if args.odom_topic:
            odom_topic_to_use = args.odom_topic
            print(f"   Using user-specified odometry topic: {odom_topic_to_use}")
        elif discovered['odometry']:
            odom_topic_to_use = discovered['odometry'][0]['name']
            print(f"   Auto-selected odometry topic: {odom_topic_to_use} (highest score)")
    
    # Generate or use provided session ID
    print("\n2. Setting up session...")
    if args.session_id:
        session_id = args.session_id
        print(f"   ✓ Using provided session ID: {session_id}")
    else:
        session_id = generate_session_id_from_rosbag(rosbag_path)
        print(f"   ✓ Generated session ID from rosbag: {session_id}")
    
    # Create session via API (unless skipped)
    if not args.skip_session_create:
        print(f"\n3. Creating session via API...")
        print(f"   URL: {base_url}/session/create/")
        session_created_success, session_was_created = create_session_via_api(
            base_url, session_id, args.asset_name, str(rosbag_path)
        )
        
        if session_created_success:
            if session_was_created:
                print(f"   ✓ Session created successfully via API")
            else:
                print(f"   ✓ Session already exists in database")
        else:
            print(f"   ✗ Failed to create session via API")
            print(f"   ⚠ Cannot proceed - session creation failed")
            print(f"   💡 Tip: Use --skip-session-create if session already exists")
            return
    else:
        print(f"\n3. Skipping session creation (--skip-session-create flag set)")
    
    # Read GPS data from rosbag
    gps_points = []
    if not args.skip_gps and gps_topic_to_use:
        print(f"\n4. Reading GPS data from rosbag2...")
        print(f"   Topic: {gps_topic_to_use}")
        try:
            gps_points = read_rosbag2_gps(rosbag_path, gps_topic_to_use)
            
            if len(gps_points) == 0:
                print("   ⚠ No GPS messages found in rosbag")
            else:
                print(f"   ✓ Found {len(gps_points)} GPS messages")
                
                # Show sample of first GPS point
                first_point = gps_points[0]
                print(f"\n   Sample GPS point:")
                print(f"     Timestamp: {first_point.get('timestamp', 'N/A')}")
                print(f"     Position: ({first_point.get('latitude', 'N/A')}, {first_point.get('longitude', 'N/A')})")
                print(f"     Altitude: {first_point.get('altitude', 'N/A')} m")
                print(f"     Fix Type: {first_point.get('fix_type', 'N/A')}")
                print(f"     Satellites: {first_point.get('satellites_visible', 'N/A')}")
            
        except Exception as e:
            print(f"   ⚠ Error reading GPS data from rosbag: {e}")
            import traceback
            if '--debug' in sys.argv or '-d' in sys.argv:
                print(f"   Traceback: {traceback.format_exc()}")
    elif args.skip_gps:
        print(f"\n4. Skipping GPS data (--skip-gps flag set)")
    else:
        print(f"\n4. Skipping GPS data (no GPS topic found)")
    
    # Read Odometry data from rosbag
    odom_points = []
    if not args.skip_odom and odom_topic_to_use:
        print(f"\n5. Reading odometry data from rosbag2...")
        print(f"   Topic: {odom_topic_to_use}")
        try:
            odom_points = read_rosbag2_odometry(rosbag_path, odom_topic_to_use)
            
            if len(odom_points) == 0:
                print("   ⚠ No odometry messages found in rosbag")
            else:
                print(f"   ✓ Found {len(odom_points)} odometry messages")
                
                # Show sample of first odometry point
                first_point = odom_points[0]
                print(f"\n   Sample odometry point:")
                print(f"     Timestamp: {first_point.get('timestamp', 'N/A')}")
                print(f"     Position (NED): x={first_point.get('x', 'N/A')}, y={first_point.get('y', 'N/A')}, z={first_point.get('z', 'N/A')}")
                print(f"     Velocity: vx={first_point.get('vx', 'N/A')}, vy={first_point.get('vy', 'N/A')}, vz={first_point.get('vz', 'N/A')}")
                if 'heading' in first_point:
                    print(f"     Heading: {first_point.get('heading', 'N/A')} rad")
            
        except Exception as e:
            print(f"   ⚠ Error reading odometry data from rosbag: {e}")
            import traceback
            if '--debug' in sys.argv or '-d' in sys.argv:
                print(f"   Traceback: {traceback.format_exc()}")
    elif args.skip_odom:
        print(f"\n5. Skipping odometry data (--skip-odom flag set)")
    else:
        print(f"\n5. Skipping odometry data (no odometry topic found)")
    
    # Check if we have any data to post
    if len(gps_points) == 0 and len(odom_points) == 0:
        print("\n✗ No telemetry data found in rosbag. Exiting.")
        return
    
    # Display API URL being used
    print(f"\n6. API Configuration:")
    print(f"   Base URL: {base_url}")
    if gps_points:
        print(f"   GPS Endpoint: {base_url}/gps-fix-raw/")
    if odom_points:
        print(f"   Odometry Endpoint: {base_url}/local-position-odom/")
    
    # Test connectivity
    print(f"\n7. Testing API connectivity...", end='', flush=True)
    try:
        # Try to reach the server
        test_response = requests.get(
            base_url.replace('/api/telemetry', '/admin/'),
            timeout=3.0,
            allow_redirects=True
        )
        print(f" ✓ Connected (Status: {test_response.status_code})")
    except requests.exceptions.ConnectionError:
        print(f" ⚠ Connection failed - will attempt to post anyway")
        print(f"   Make sure the server is running at: {base_url}")
    except Exception as e:
        print(f" ⚠ Could not verify connectivity: {e}")
    
    # Post data
    results = {}
    step = 8
    
    if gps_points:
        print(f"\n{step}. Posting GPS raw data via API...")
        gps_results = post_gps_raw_via_api(base_url, session_id, gps_points)
        results.update(gps_results)
        step += 1
    
    if odom_points:
        print(f"\n{step}. Posting odometry data via API...")
        odom_results = post_local_position_odom_via_api(base_url, session_id, odom_points)
        results.update(odom_results)
        step += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Rosbag Path: {rosbag_path}")
    print(f"Session ID: {session_id}")
    print(f"API Base URL: {base_url}")
    if gps_topic_to_use:
        print(f"GPS Topic: {gps_topic_to_use}")
    if odom_topic_to_use:
        print(f"Odometry Topic: {odom_topic_to_use}")
    
    print(f"\nRecords Posted:")
    total_success = 0
    
    if 'gps_raw' in results:
        gps_success = results['gps_raw']['success']
        gps_errors = len(results['gps_raw']['errors'])
        print(f"  • Raw GPS Fix: {gps_success} (errors: {gps_errors})")
        total_success += gps_success
    
    if 'local_position_odom' in results:
        odom_success = results['local_position_odom']['success']
        odom_errors = len(results['local_position_odom']['errors'])
        print(f"  • Local Position Odom: {odom_success} (errors: {odom_errors})")
        total_success += odom_success
    
    print(f"  • Total: {total_success}")
    
    # Show errors if any
    has_errors = False
    if 'gps_raw' in results and len(results['gps_raw']['errors']) > 0:
        has_errors = True
        print(f"\nGPS Errors encountered:")
        print(f"  GPS Raw: {len(results['gps_raw']['errors'])} errors")
        for err in results['gps_raw']['errors'][:5]:  # Show first 5 errors
            print(f"    - {err}")
        if len(results['gps_raw']['errors']) > 5:
            print(f"    ... and {len(results['gps_raw']['errors']) - 5} more errors")
    
    if 'local_position_odom' in results and len(results['local_position_odom']['errors']) > 0:
        has_errors = True
        print(f"\nOdometry Errors encountered:")
        print(f"  Local Position Odom: {len(results['local_position_odom']['errors'])} errors")
        for err in results['local_position_odom']['errors'][:5]:  # Show first 5 errors
            print(f"    - {err}")
        if len(results['local_position_odom']['errors']) > 5:
            print(f"    ... and {len(results['local_position_odom']['errors']) - 5} more errors")
    
    print("=" * 70)
    if has_errors:
        print("\n⚠ Telemetry injection completed with errors!")
    else:
        print("\n✓ Telemetry injection complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

