# DeepGIS Telemetry Publisher for MAVROS

ROS2 node that streams telemetry data from MAVROS to the DeepGIS platform, conforming to the DeepGIS Telemetry API specification.

## Features

- ✅ **DeepGIS API Compliant**: Fully conforms to DeepGIS telemetry API specification
- ✅ **Real-time & Batch Modes**: Support for both real-time streaming and efficient batch uploads
- ✅ **Complete Telemetry**: Publishes GPS (raw & estimated) and odometry data
- ✅ **Automatic Session Management**: Creates and manages telemetry sessions
- ✅ **Reference Frame Handling**: Automatically sets reference position from first GPS fix
- ✅ **Quaternion to Heading Conversion**: Properly converts orientation to heading angle
- ✅ **Covariance Extraction**: Extracts and formats position/velocity covariance matrices
- ✅ **Error Handling**: Robust error handling with detailed logging

## Key Fixes & Improvements

### 1. Session Creation API Compliance
**Before:**
```python
{
    'vehicle_id': 'pixhawk_001',
    'session_name': 'flight_...',
    'start_time': '...',
    'metadata': {...}
}
```

**After (DeepGIS compliant):**
```python
{
    'session_id': 'mavros_20240101_120000',
    'asset_name': 'MAVROS Vehicle',
    'project_title': 'MAVROS Data Collection',
    'flight_mode': 'AUTO',
    'mission_type': 'Telemetry Collection',
    'notes': '...'
}
```

### 2. Local Position Odometry Format
**Before:** Nested ROS message structure with quaternions
```python
{
    'pose': {'position': {...}, 'orientation': {...}},
    'twist': {'linear': {...}, 'angular': {...}}
}
```

**After (DeepGIS compliant):** Flat structure with heading
```python
{
    'session_id': '...',
    'timestamp': '2024-01-01T12:00:00Z',
    'timestamp_usec': 1234567890,
    'x': 10.5,           # NED position (meters)
    'y': 5.2,
    'z': -2.1,
    'vx': 1.2,           # Velocity (m/s)
    'vy': 0.8,
    'vz': -0.1,
    'heading': 0.785,    # Converted from quaternion (radians)
    'heading_rate': 0.01,
    'position_covariance': [...],  # 3x3 matrix (9 elements)
    'velocity_covariance': [...],  # 3x3 matrix (9 elements)
    'ref_lat': 33.4255,  # Reference position
    'ref_lon': -111.9400,
    'ref_alt': 350.0
}
```

### 3. GPS Fix Format
**Before:** Nested structure with ROS-specific fields
```python
{
    'header': {...},
    'status': {'status': ..., 'service': ...},
    'position_covariance_type': ...
}
```

**After (DeepGIS compliant):** Flat structure with standard fields
```python
{
    'session_id': '...',
    'timestamp': '2024-01-01T12:00:00Z',
    'timestamp_usec': 1234567890,
    'latitude': 33.4255,
    'longitude': -111.9400,
    'altitude': 352.5,
    'fix_type': 3,       # Mapped from NavSatStatus
    'eph': 2.5,          # Horizontal accuracy (meters)
    'epv': 3.0           # Vertical accuracy (meters)
}
```

### 4. Batch API Compliance
**Before:**
```python
{
    'session_id': '...',
    'local_positions': [...],
    'gps_raw': [...],
    'gps_estimated': [...]
}
```

**After (DeepGIS compliant):**
```python
{
    'local_position_odom': [...],  # Correct endpoint name
    'gps_fix_raw': [...],          # Correct endpoint name
    'gps_fix_estimated': [...]     # Correct endpoint name
}
```

## Installation

### Prerequisites

- ROS2 (Humble, Iron, or later)
- MAVROS installed and configured
- Python 3.8+
- `requests` library

```bash
# Install Python dependencies
pip3 install requests

# Or if using ROS2 workspace
rosdep install --from-paths src --ignore-src -r -y
```

### Setup

1. Copy the node to your ROS2 workspace:
```bash
cp deepgis_telemetry_publisher.py ~/ros2_ws/src/your_package/scripts/
chmod +x ~/ros2_ws/src/your_package/scripts/deepgis_telemetry_publisher.py
```

2. Copy the launch file (optional):
```bash
cp deepgis_telemetry_publisher_launch.py ~/ros2_ws/src/your_package/launch/
```

3. Copy the config file (optional):
```bash
cp deepgis_telemetry_config.yaml ~/ros2_ws/src/your_package/config/
```

## Usage

### Method 1: Direct Execution

```bash
# Run with default parameters
ros2 run your_package deepgis_telemetry_publisher.py

# Run with custom API URL
ros2 run your_package deepgis_telemetry_publisher.py \
    --ros-args \
    -p deepgis_api_url:=http://192.168.0.186:8080 \
    -p asset_name:="My Drone" \
    -p session_id:=mission_001

# Run with config file
ros2 run your_package deepgis_telemetry_publisher.py \
    --ros-args --params-file deepgis_telemetry_config.yaml
```

### Method 2: Launch File

```bash
# Basic launch
ros2 launch your_package deepgis_telemetry_publisher_launch.py

# Launch with custom parameters
ros2 launch your_package deepgis_telemetry_publisher_launch.py \
    deepgis_api_url:=http://192.168.0.186:8080 \
    asset_name:="My Drone" \
    session_id:=mission_001 \
    publish_rate:=5.0 \
    enable_batch_mode:=true
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deepgis_api_url` | string | `http://192.168.0.186:8080` | DeepGIS API base URL |
| `api_key` | string | `""` | Optional API key for authentication |
| `asset_name` | string | `"MAVROS Vehicle"` | Name of the vehicle/asset |
| `session_id` | string | `mavros_YYYYMMDD_HHMMSS` | Unique session identifier |
| `project_title` | string | `"MAVROS Data Collection"` | Project title |
| `flight_mode` | string | `"AUTO"` | Flight mode (MANUAL, AUTO, GUIDED, etc.) |
| `mission_type` | string | `"Telemetry Collection"` | Mission type description |
| `mavros_namespace` | string | `"/mavros"` | MAVROS topic namespace |
| `publish_rate` | float | `1.0` | Publishing rate in Hz |
| `batch_size` | int | `10` | Samples to accumulate before batch upload |
| `enable_batch_mode` | bool | `true` | Enable batch mode |

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/mavros/local_position/odom` | `nav_msgs/Odometry` | Local position odometry (ENU frame) |
| `/mavros/global_position/raw/fix` | `sensor_msgs/NavSatFix` | Raw GPS fix from receiver |
| `/mavros/global_position/global` | `sensor_msgs/NavSatFix` | Estimated GPS position |

## DeepGIS API Endpoints Used

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/telemetry/session/create/` | POST | Create telemetry session |
| `/api/telemetry/local-position-odom/` | POST | Post local position odometry |
| `/api/telemetry/gps-fix-raw/` | POST | Post raw GPS fix |
| `/api/telemetry/gps-fix-estimated/` | POST | Post estimated GPS fix |
| `/api/telemetry/batch/` | POST | Post batch telemetry data |

## Examples

### Example 1: High-Frequency Real-Time Streaming

For real-time monitoring with high update rates:

```bash
ros2 run your_package deepgis_telemetry_publisher.py \
    --ros-args \
    -p deepgis_api_url:=http://192.168.0.186:8080 \
    -p asset_name:="Racing Drone" \
    -p publish_rate:=10.0 \
    -p enable_batch_mode:=false
```

### Example 2: Efficient Batch Logging

For efficient data logging with minimal network overhead:

```bash
ros2 run your_package deepgis_telemetry_publisher.py \
    --ros-args \
    -p deepgis_api_url:=http://192.168.0.186:8080 \
    -p asset_name:="Survey Drone" \
    -p publish_rate:=1.0 \
    -p batch_size:=100 \
    -p enable_batch_mode:=true
```

### Example 3: Custom Mission Configuration

```bash
ros2 run your_package deepgis_telemetry_publisher.py \
    --ros-args \
    -p deepgis_api_url:=http://192.168.0.186:8080 \
    -p asset_name:="Surveyor-1" \
    -p session_id:=survey_mission_20240101 \
    -p project_title:="Archaeological Survey" \
    -p flight_mode:=AUTO \
    -p mission_type:="Aerial Survey" \
    -p publish_rate:=2.0 \
    -p batch_size:=50
```

## Monitoring

The node provides detailed logging:

```
[INFO] [deepgis_telemetry_publisher]: DeepGIS Telemetry Publisher initialized
[INFO] [deepgis_telemetry_publisher]: API URL: http://192.168.0.186:8080
[INFO] [deepgis_telemetry_publisher]: Asset Name: MAVROS Vehicle
[INFO] [deepgis_telemetry_publisher]: Session ID: mavros_20240101_120000
[INFO] [deepgis_telemetry_publisher]: Batch Mode: True
[INFO] [deepgis_telemetry_publisher]: Created telemetry session: mavros_20240101_120000
[INFO] [deepgis_telemetry_publisher]: Set reference position: (33.425500, -111.940000, 350.00m)
[INFO] [deepgis_telemetry_publisher]: Sent batch: 10 odom, 10 GPS raw, 10 GPS est (30 total)
```

## Troubleshooting

### Session Creation Fails

**Problem:** `Failed to create session: 400 - Validation failed`

**Solution:** Check that all required session fields are provided:
- `session_id`
- `asset_name`
- `project_title`
- `flight_mode`
- `mission_type`

### API Connection Errors

**Problem:** `Error creating telemetry session: Connection refused`

**Solution:**
1. Check that DeepGIS server is running
2. Verify the `deepgis_api_url` parameter is correct
3. Check network connectivity: `curl http://192.168.0.186:8080/api/telemetry/`

### No Data Being Sent

**Problem:** Node runs but no data appears in DeepGIS

**Solution:**
1. Check that MAVROS is publishing data: `ros2 topic hz /mavros/local_position/odom`
2. Verify topics are being received: `ros2 topic echo /mavros/global_position/raw/fix`
3. Check node logs for errors
4. Verify session was created successfully

### Coordinate Frame Issues

**Problem:** Position data looks wrong in DeepGIS

**Solution:**
1. Check that reference position is set (look for "Set reference position" in logs)
2. Verify GPS fix quality is good (fix_type >= 3)
3. Check MAVROS coordinate frame configuration (ENU vs NED)

## Testing

### Test with Sample Data

```bash
# Terminal 1: Start MAVROS with simulator
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557

# Terminal 2: Start telemetry publisher
ros2 run your_package deepgis_telemetry_publisher.py \
    --ros-args \
    -p deepgis_api_url:=http://localhost:8000 \
    -p enable_batch_mode:=false

# Terminal 3: Monitor published data
ros2 topic echo /mavros/local_position/odom
```

### Verify API Data

Check that data appears in DeepGIS:

```bash
# List sessions
curl http://192.168.0.186:8080/api/telemetry/sessions/

# Get session path data
curl http://192.168.0.186:8080/api/telemetry/sessions/mavros_20240101_120000/path/
```

## Performance Notes

- **Batch Mode (Recommended)**: More efficient, reduces API calls, better for long missions
- **Real-Time Mode**: Lower latency, better for live monitoring, higher network usage
- **Publish Rate**: Balance between data resolution and system load
  - 1 Hz: Good for general logging
  - 5-10 Hz: Good for detailed analysis
  - >10 Hz: Only if needed for high-speed missions

## License

Same as DeepGIS project license.

## Support

For issues or questions:
- Check DeepGIS API documentation: `/api/telemetry/`
- Review logs for error messages
- Verify MAVROS is working correctly
- Test API connectivity independently

