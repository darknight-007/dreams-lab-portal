#!/usr/bin/env python3
"""
DeepGIS Telemetry Publisher Node

Subscribes to MAVROS telemetry topics and publishes data to DeepGIS API
conforming to the DeepGIS telemetry API specification.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import PoseStamped

import requests
import json
import math
from datetime import datetime, timezone
from threading import Lock
import time


class DeepGISTelemetryPublisher(Node):
    """
    ROS2 Node that publishes vehicle telemetry data to DeepGIS API.
    Conforms to DeepGIS telemetry API specification.
    """

    def __init__(self):
        super().__init__('deepgis_telemetry_publisher')

        # Declare parameters
        self.declare_parameter('deepgis_api_url', 'http://192.168.0.186:8080')
        self.declare_parameter('api_key', '')  # Optional API key for authentication
        self.declare_parameter('asset_name', 'MAVROS Vehicle')
        self.declare_parameter('session_id', f'mavros_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.declare_parameter('project_title', 'MAVROS Data Collection')
        self.declare_parameter('flight_mode', 'AUTO')
        self.declare_parameter('mission_type', 'Telemetry Collection')
        self.declare_parameter('mavros_namespace', '/mavros')
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('batch_size', 10)  # Number of samples before batch upload
        self.declare_parameter('enable_batch_mode', True)
        
        # Get parameters
        self.api_url = self.get_parameter('deepgis_api_url').value.rstrip('/')
        self.api_key = self.get_parameter('api_key').value
        self.asset_name = self.get_parameter('asset_name').value
        self.session_id = self.get_parameter('session_id').value
        self.project_title = self.get_parameter('project_title').value
        self.flight_mode = self.get_parameter('flight_mode').value
        self.mission_type = self.get_parameter('mission_type').value
        self.mavros_ns = self.get_parameter('mavros_namespace').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.batch_size = self.get_parameter('batch_size').value
        self.enable_batch = self.get_parameter('enable_batch_mode').value

        # API endpoints (conforming to DeepGIS API spec)
        self.api_endpoints = {
            'create_session': f'{self.api_url}/api/telemetry/session/create/',
            'local_position_odom': f'{self.api_url}/api/telemetry/local-position-odom/',
            'gps_fix_raw': f'{self.api_url}/api/telemetry/gps-fix-raw/',
            'gps_fix_estimated': f'{self.api_url}/api/telemetry/gps-fix-estimated/',
            'batch': f'{self.api_url}/api/telemetry/batch/',
        }

        # Session management
        self.session_active = False

        # Data buffers for batch mode
        self.local_position_buffer = []
        self.gps_raw_buffer = []
        self.gps_estimated_buffer = []
        self.buffer_lock = Lock()

        # Latest data storage
        self.latest_odom = None
        self.latest_gps_raw = None
        self.latest_gps_estimated = None
        self.data_lock = Lock()
        
        # Reference position for local frame (will be set from first GPS fix)
        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = None
        self.ref_lock = Lock()

        # QoS profile for MAVROS topics (best effort, like MAVROS)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            f'{self.mavros_ns}/local_position/odom',
            self.odom_callback,
            qos_profile
        )

        self.gps_raw_sub = self.create_subscription(
            NavSatFix,
            f'{self.mavros_ns}/global_position/raw/fix',
            self.gps_raw_callback,
            qos_profile
        )

        self.gps_estimated_sub = self.create_subscription(
            NavSatFix,
            f'{self.mavros_ns}/global_position/global',
            self.gps_estimated_callback,
            qos_profile
        )

        # HTTP session with connection pooling
        self.http_session = requests.Session()
        if self.api_key:
            self.http_session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        self.http_session.headers.update({'Content-Type': 'application/json'})

        # Create telemetry session
        self.create_telemetry_session()

        # Publisher timer
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_telemetry
        )

        self.get_logger().info('DeepGIS Telemetry Publisher initialized')
        self.get_logger().info(f'API URL: {self.api_url}')
        self.get_logger().info(f'Asset Name: {self.asset_name}')
        self.get_logger().info(f'Session ID: {self.session_id}')
        self.get_logger().info(f'Batch Mode: {self.enable_batch}')

    def create_telemetry_session(self):
        """
        Create a new telemetry session with DeepGIS API.
        Conforms to DeepGIS session creation API spec.
        """
        try:
            payload = {
                'session_id': self.session_id,
                'asset_name': self.asset_name,
                'project_title': self.project_title,
                'flight_mode': self.flight_mode,
                'mission_type': self.mission_type,
                'notes': f'ROS2 MAVROS telemetry stream from {self.get_name()}'
            }

            response = self.http_session.post(
                self.api_endpoints['create_session'],
                json=payload,
                timeout=10.0
            )

            if response.status_code in [200, 201]:
                result = response.json()
                self.session_active = True
                created = result.get('created', True)
                if created:
                    self.get_logger().info(f'Created telemetry session: {self.session_id}')
                else:
                    self.get_logger().info(f'Using existing session: {self.session_id}')
            else:
                self.get_logger().error(
                    f'Failed to create session: {response.status_code} - {response.text}'
                )

        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'Error creating telemetry session: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error creating session: {str(e)}')

    def odom_callback(self, msg: Odometry):
        """Callback for local position odometry data."""
        with self.data_lock:
            self.latest_odom = msg

    def gps_raw_callback(self, msg: NavSatFix):
        """Callback for raw GPS fix data."""
        with self.data_lock:
            self.latest_gps_raw = msg
            
        # Set reference position from first valid GPS fix
        if msg.status.status >= NavSatStatus.STATUS_FIX:
            with self.ref_lock:
                if self.ref_lat is None and msg.latitude != 0.0 and msg.longitude != 0.0:
                    self.ref_lat = msg.latitude
                    self.ref_lon = msg.longitude
                    self.ref_alt = msg.altitude
                    self.get_logger().info(
                        f'Set reference position: ({self.ref_lat:.6f}, {self.ref_lon:.6f}, {self.ref_alt:.2f}m)'
                    )

    def gps_estimated_callback(self, msg: NavSatFix):
        """Callback for estimated GPS position."""
        with self.data_lock:
            self.latest_gps_estimated = msg

    @staticmethod
    def quaternion_to_heading(x, y, z, w):
        """
        Convert quaternion to heading (yaw) angle in radians.
        
        Returns heading in radians, where:
        - 0 is North
        - π/2 is East
        - π is South
        - -π/2 is West
        """
        # Yaw (heading) = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        heading = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        )
        return heading

    def format_odom_data(self, msg: Odometry):
        """
        Format odometry data for DeepGIS API.
        Conforms to LocalPositionOdom API spec.
        """
        # Convert ROS timestamp to ISO format and microseconds
        timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        timestamp_dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
        timestamp_usec = int(timestamp_sec * 1e6)
        
        # Extract position (NED frame for DeepGIS)
        # MAVROS local_position/odom is typically in ENU, need to check frame_id
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Extract velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        
        # Convert quaternion to heading
        heading = self.quaternion_to_heading(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        
        # Extract heading rate (angular velocity around z-axis)
        heading_rate = msg.twist.twist.angular.z
        
        # Extract covariances (convert from 6x6 to 3x3 for position and velocity)
        pose_cov = list(msg.pose.covariance)
        twist_cov = list(msg.twist.covariance)
        
        # Position covariance (extract x, y, z rows/cols from 6x6 matrix)
        position_covariance = [
            pose_cov[0], pose_cov[1], pose_cov[2],      # Row 0 (x)
            pose_cov[6], pose_cov[7], pose_cov[8],      # Row 1 (y)
            pose_cov[12], pose_cov[13], pose_cov[14]    # Row 2 (z)
        ]
        
        # Velocity covariance
        velocity_covariance = [
            twist_cov[0], twist_cov[1], twist_cov[2],   # Row 0 (vx)
            twist_cov[6], twist_cov[7], twist_cov[8],   # Row 1 (vy)
            twist_cov[12], twist_cov[13], twist_cov[14] # Row 2 (vz)
        ]
        
        data = {
            'session_id': self.session_id,
            'timestamp': timestamp_dt.isoformat(),
            'timestamp_usec': timestamp_usec,
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'vx': float(vx),
            'vy': float(vy),
            'vz': float(vz),
            'heading': float(heading),
            'heading_rate': float(heading_rate),
            'position_covariance': position_covariance,
            'velocity_covariance': velocity_covariance,
        }
        
        # Add reference position if available
        with self.ref_lock:
            if self.ref_lat is not None:
                data['ref_lat'] = self.ref_lat
                data['ref_lon'] = self.ref_lon
                data['ref_alt'] = self.ref_alt
        
        return data

    def format_gps_data(self, msg: NavSatFix, is_raw=True):
        """
        Format GPS fix data for DeepGIS API.
        Conforms to GPS Fix Raw/Estimated API spec.
        """
        # Convert ROS timestamp to ISO format and microseconds
        timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        timestamp_dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
        timestamp_usec = int(timestamp_sec * 1e6)
        
        # Map NavSatStatus to GPS fix type
        # NavSatStatus: -1=no fix, 0=fix, 1=SBAS fix, 2=GBAS fix
        # GPS fix type: 0=no fix, 1=dead reckoning, 2=2D fix, 3=3D fix, 4=GPS+dead reckoning, 5=time only
        status_to_fix_type = {
            -1: 0,  # No fix
            0: 3,   # Fix (assume 3D)
            1: 3,   # SBAS fix (3D)
            2: 3,   # GBAS fix (3D)
        }
        fix_type = status_to_fix_type.get(msg.status.status, 0)
        
        # Extract position covariance diagonal elements as accuracy estimates
        # NavSatFix covariance is 3x3 for (lat, lon, alt)
        cov = list(msg.position_covariance)
        
        # Estimate horizontal and vertical accuracy from covariance
        eph = None  # Horizontal position accuracy
        epv = None  # Vertical position accuracy
        
        if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            # eph ≈ sqrt(cov_lat^2 + cov_lon^2) in meters
            # Need to convert lat/lon covariance to meters (approximate)
            lat_std = math.sqrt(abs(cov[0]))  # degrees
            lon_std = math.sqrt(abs(cov[4]))  # degrees
            
            # Approximate conversion to meters (111km per degree at equator)
            lat_m = lat_std * 111000.0
            lon_m = lon_std * 111000.0 * math.cos(math.radians(msg.latitude))
            
            eph = math.sqrt(lat_m * lat_m + lon_m * lon_m)
            epv = math.sqrt(abs(cov[8]))  # Already in meters
        
        data = {
            'session_id': self.session_id,
            'timestamp': timestamp_dt.isoformat(),
            'timestamp_usec': timestamp_usec,
            'latitude': float(msg.latitude),
            'longitude': float(msg.longitude),
            'altitude': float(msg.altitude),
            'fix_type': fix_type,
        }
        
        # Add optional accuracy fields if available
        if eph is not None:
            data['eph'] = float(eph)
        if epv is not None:
            data['epv'] = float(epv)
        
        return data

    def publish_telemetry(self):
        """Publish telemetry data to DeepGIS API."""
        if not self.session_active:
            return

        with self.data_lock:
            odom = self.latest_odom
            gps_raw = self.latest_gps_raw
            gps_estimated = self.latest_gps_estimated

        if self.enable_batch:
            # Batch mode: accumulate data and send in batches
            with self.buffer_lock:
                if odom:
                    self.local_position_buffer.append(self.format_odom_data(odom))
                if gps_raw:
                    self.gps_raw_buffer.append(self.format_gps_data(gps_raw, is_raw=True))
                if gps_estimated:
                    self.gps_estimated_buffer.append(self.format_gps_data(gps_estimated, is_raw=False))

                # Check if we should send a batch
                total_samples = (len(self.local_position_buffer) +
                               len(self.gps_raw_buffer) +
                               len(self.gps_estimated_buffer))

                if total_samples >= self.batch_size:
                    self.send_batch()
        else:
            # Real-time mode: send data immediately
            if odom:
                self.send_local_position_odom(self.format_odom_data(odom))
            if gps_raw:
                self.send_gps_fix_raw(self.format_gps_data(gps_raw, is_raw=True))
            if gps_estimated:
                self.send_gps_fix_estimated(self.format_gps_data(gps_estimated, is_raw=False))

    def send_local_position_odom(self, data):
        """Send local position odometry data to API."""
        try:
            response = self.http_session.post(
                self.api_endpoints['local_position_odom'],
                json=data,
                timeout=5.0
            )
            if response.status_code == 201:
                self.get_logger().debug('Sent local position odometry')
            elif response.status_code not in [200, 201]:
                self.get_logger().warn(
                    f'Failed to send local position: {response.status_code} - {response.text}'
                )
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f'Error sending local position: {str(e)}')

    def send_gps_fix_raw(self, data):
        """Send raw GPS data to API."""
        try:
            response = self.http_session.post(
                self.api_endpoints['gps_fix_raw'],
                json=data,
                timeout=5.0
            )
            if response.status_code == 201:
                self.get_logger().debug('Sent GPS raw fix')
            elif response.status_code not in [200, 201]:
                self.get_logger().warn(
                    f'Failed to send GPS raw: {response.status_code} - {response.text}'
                )
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f'Error sending GPS raw: {str(e)}')

    def send_gps_fix_estimated(self, data):
        """Send estimated GPS data to API."""
        try:
            response = self.http_session.post(
                self.api_endpoints['gps_fix_estimated'],
                json=data,
                timeout=5.0
            )
            if response.status_code == 201:
                self.get_logger().debug('Sent GPS estimated fix')
            elif response.status_code not in [200, 201]:
                self.get_logger().warn(
                    f'Failed to send GPS estimated: {response.status_code} - {response.text}'
                )
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f'Error sending GPS estimated: {str(e)}')

    def send_batch(self):
        """
        Send accumulated data in batch.
        Conforms to DeepGIS batch API spec.
        """
        with self.buffer_lock:
            if not (self.local_position_buffer or self.gps_raw_buffer or self.gps_estimated_buffer):
                return

            batch_data = {}
            
            # Only include non-empty arrays (as per API spec)
            if self.local_position_buffer:
                batch_data['local_position_odom'] = self.local_position_buffer.copy()
            if self.gps_raw_buffer:
                batch_data['gps_fix_raw'] = self.gps_raw_buffer.copy()
            if self.gps_estimated_buffer:
                batch_data['gps_fix_estimated'] = self.gps_estimated_buffer.copy()

            # Clear buffers
            num_odom = len(self.local_position_buffer)
            num_gps_raw = len(self.gps_raw_buffer)
            num_gps_est = len(self.gps_estimated_buffer)
            
            self.local_position_buffer.clear()
            self.gps_raw_buffer.clear()
            self.gps_estimated_buffer.clear()

        try:
            response = self.http_session.post(
                self.api_endpoints['batch'],
                json=batch_data,
                timeout=10.0
            )
            if response.status_code in [200, 201, 207]:  # 207 = Multi-Status (partial success)
                total_items = num_odom + num_gps_raw + num_gps_est
                self.get_logger().info(
                    f'Sent batch: {num_odom} odom, {num_gps_raw} GPS raw, {num_gps_est} GPS est ({total_items} total)'
                )
                
                # Log any errors from batch response
                if response.status_code == 207:
                    result = response.json()
                    if 'results' in result:
                        for data_type, data_result in result['results'].items():
                            if data_result.get('errors'):
                                self.get_logger().warn(
                                    f'Batch {data_type} had {len(data_result["errors"])} errors'
                                )
            else:
                self.get_logger().warn(
                    f'Failed to send batch: {response.status_code} - {response.text}'
                )
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'Error sending batch: {str(e)}')

    def destroy_node(self):
        """Cleanup before node shutdown."""
        # Send any remaining buffered data
        if self.enable_batch:
            self.get_logger().info('Flushing remaining telemetry data...')
            self.send_batch()
        
        # Close HTTP session
        self.http_session.close()
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = DeepGISTelemetryPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

