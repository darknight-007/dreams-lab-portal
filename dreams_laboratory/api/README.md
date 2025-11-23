# Pixhawk Telemetry REST API

RESTful API endpoints for posting telemetry data from Pixhawk-based state estimators.

## Base URL

```
http://deepgis.org/api/telemetry/
```

## Endpoints

### 1. Local Position Odometry

**Endpoint:** `POST /api/telemetry/local-position-odom/`

**Description:** Post local position odometry data in NED (North-East-Down) frame.

**Request Body:**
```json
{
  "session_id": "flight_2024_11_23_001",
  "timestamp": "2024-11-23T01:00:00Z",
  "timestamp_usec": 1234567890,
  "x": 10.5,
  "y": 5.2,
  "z": -2.1,
  "vx": 1.2,
  "vy": 0.8,
  "vz": -0.1,
  "heading": 0.785,
  "heading_rate": 0.01,
  "xy_valid": true,
  "z_valid": true,
  "v_xy_valid": true,
  "v_z_valid": true,
  "heading_valid": true,
  "position_covariance": [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.02],
  "velocity_covariance": [0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.1],
  "ref_lat": 33.4255,
  "ref_lon": -111.9400,
  "ref_alt": 350.0,
  "eph": 0.5,
  "epv": 0.8,
  "evh": 0.2,
  "evv": 0.3
}
```

**Required Fields:**
- `session_id` (string): Telemetry session identifier
- `timestamp` (string/float): ISO 8601 timestamp or Unix timestamp
- `x` (float): North position in meters
- `y` (float): East position in meters
- `z` (float): Down position in meters (positive down)

**Optional Fields:**
- `timestamp_usec` (integer): Microseconds since system boot
- `vx`, `vy`, `vz` (float): Velocity components in m/s
- `heading` (float): Heading angle in radians
- `heading_rate` (float): Heading rate in rad/s
- `xy_valid`, `z_valid`, `v_xy_valid`, `v_z_valid`, `heading_valid` (boolean): Validity flags
- `position_covariance` (array): 9-element position covariance matrix
- `velocity_covariance` (array): 9-element velocity covariance matrix
- `ref_lat`, `ref_lon`, `ref_alt` (float): Reference frame origin
- `eph`, `epv`, `evh`, `evv` (float): Error estimates

**Response (201 Created):**
```json
{
  "success": true,
  "id": 123,
  "message": "Local position odometry data created successfully",
  "timestamp": "2024-11-23T01:00:00Z"
}
```

**Response (400 Bad Request):**
```json
{
  "error": "Validation failed",
  "errors": {
    "session_id": "Session 'invalid_session' not found.",
    "x": "Must be a number."
  }
}
```

---

### 2. Raw GPS Fix

**Endpoint:** `POST /api/telemetry/gps-fix-raw/`

**Description:** Post raw GPS fix data directly from GPS receiver.

**Request Body:**
```json
{
  "session_id": "flight_2024_11_23_001",
  "timestamp": "2024-11-23T01:00:00Z",
  "timestamp_usec": 1234567891,
  "latitude": 33.4255000,
  "longitude": -111.9400000,
  "altitude": 352.5,
  "fix_type": 3,
  "satellites_visible": 12,
  "satellites_used": 10,
  "hdop": 1.2,
  "vdop": 1.5,
  "pdop": 1.9,
  "eph": 2.5,
  "epv": 3.0,
  "s_variance_m_s": 0.5,
  "vel_n_m_s": 1.2,
  "vel_e_m_s": 0.8,
  "vel_d_m_s": -0.1,
  "vel_m_s": 1.44,
  "cog_rad": 0.588,
  "time_utc_usec": 1700707200000000,
  "noise_per_ms": 25,
  "jamming_indicator": 0,
  "jamming_state": 1,
  "device_id": 1
}
```

**Required Fields:**
- `session_id` (string): Telemetry session identifier
- `timestamp` (string/float): ISO 8601 timestamp or Unix timestamp
- `latitude` (float): Latitude in degrees (-90 to 90)
- `longitude` (float): Longitude in degrees (-180 to 180)
- `altitude` (float): Altitude above MSL in meters
- `fix_type` (integer): GPS fix type (0-5)
  - 0 = No fix
  - 1 = Dead reckoning
  - 2 = 2D fix
  - 3 = 3D fix
  - 4 = GPS + dead reckoning
  - 5 = Time only

**Optional Fields:**
- `satellites_visible` (integer): Number of visible satellites
- `satellites_used` (integer): Number of satellites used in solution
- `hdop`, `vdop`, `pdop` (float): Dilution of Precision values
- `eph`, `epv` (float): Position accuracy in meters
- `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s` (float): Velocity components
- `jamming_state` (integer): 0=unknown, 1=ok, 2=warning, 3=critical

**Response (201 Created):**
```json
{
  "success": true,
  "id": 456,
  "message": "Raw GPS fix data created successfully",
  "timestamp": "2024-11-23T01:00:00Z"
}
```

---

### 3. Estimated GPS Fix

**Endpoint:** `POST /api/telemetry/gps-fix-estimated/`

**Description:** Post estimated GPS position from state estimator (filtered/fused).

**Request Body:**
```json
{
  "session_id": "flight_2024_11_23_001",
  "timestamp": "2024-11-23T01:00:00Z",
  "timestamp_usec": 1234567892,
  "latitude": 33.4255010,
  "longitude": -111.9400010,
  "altitude": 352.3,
  "vel_n_m_s": 1.18,
  "vel_e_m_s": 0.82,
  "vel_d_m_s": -0.12,
  "position_covariance": [0.5, 0, 0, 0, 0.5, 0, 0, 0, 1.0],
  "velocity_covariance": [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.2],
  "eph": 1.8,
  "epv": 2.2,
  "evh": 0.15,
  "evv": 0.25,
  "estimator_type": "EKF2",
  "confidence": 0.95,
  "position_valid": true,
  "velocity_valid": true,
  "raw_gps_fix_id": 456,
  "local_position_id": 123
}
```

**Required Fields:**
- `session_id` (string): Telemetry session identifier
- `timestamp` (string/float): ISO 8601 timestamp or Unix timestamp
- `latitude` (float): Estimated latitude in degrees
- `longitude` (float): Estimated longitude in degrees
- `altitude` (float): Estimated altitude in meters

**Optional Fields:**
- `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s` (float): Estimated velocity components
- `position_covariance` (array): 9-element position covariance matrix
- `velocity_covariance` (array): 9-element velocity covariance matrix
- `eph`, `epv`, `evh`, `evv` (float): Error estimates
- `estimator_type` (string): State estimator type (e.g., "EKF2", "LPE")
- `confidence` (float): Estimation confidence (0-1)
- `position_valid`, `velocity_valid` (boolean): Validity flags
- `raw_gps_fix_id` (integer): ID of related raw GPS fix
- `local_position_id` (integer): ID of related local position odom

**Response (201 Created):**
```json
{
  "success": true,
  "id": 789,
  "message": "Estimated GPS fix data created successfully",
  "timestamp": "2024-11-23T01:00:00Z"
}
```

---

### 4. Batch Telemetry

**Endpoint:** `POST /api/telemetry/batch/`

**Description:** Post multiple telemetry records in a single request.

**Request Body:**
```json
{
  "local_position_odom": [
    {
      "session_id": "flight_2024_11_23_001",
      "timestamp": "2024-11-23T01:00:00Z",
      "x": 10.5,
      "y": 5.2,
      "z": -2.1
    },
    {
      "session_id": "flight_2024_11_23_001",
      "timestamp": "2024-11-23T01:00:01Z",
      "x": 10.6,
      "y": 5.3,
      "z": -2.0
    }
  ],
  "gps_fix_raw": [
    {
      "session_id": "flight_2024_11_23_001",
      "timestamp": "2024-11-23T01:00:00Z",
      "latitude": 33.4255,
      "longitude": -111.9400,
      "altitude": 352.5,
      "fix_type": 3
    }
  ],
  "gps_fix_estimated": [
    {
      "session_id": "flight_2024_11_23_001",
      "timestamp": "2024-11-23T01:00:00Z",
      "latitude": 33.4255,
      "longitude": -111.9400,
      "altitude": 352.3
    }
  ]
}
```

**Response (201 Created or 207 Multi-Status):**
```json
{
  "success": true,
  "message": "Batch processed: 3 records created",
  "results": {
    "local_position_odom": {
      "created": 2,
      "errors": []
    },
    "gps_fix_raw": {
      "created": 1,
      "errors": []
    },
    "gps_fix_estimated": {
      "created": 1,
      "errors": []
    }
  }
}
```

**Response with Errors (207 Multi-Status):**
```json
{
  "success": true,
  "message": "Batch processed: 2 records created",
  "results": {
    "local_position_odom": {
      "created": 1,
      "errors": [
        {
          "index": 1,
          "errors": {
            "session_id": "Session 'invalid' not found."
          }
        }
      ]
    },
    "gps_fix_raw": {
      "created": 1,
      "errors": []
    },
    "gps_fix_estimated": {
      "created": 0,
      "errors": []
    }
  }
}
```

---

## Usage Examples

### Python (using requests)

```python
import requests
import json
from datetime import datetime

BASE_URL = "http://deepgis.org/api/telemetry"

# Local Position Odometry
odom_data = {
    "session_id": "flight_2024_11_23_001",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "x": 10.5,
    "y": 5.2,
    "z": -2.1,
    "vx": 1.2,
    "vy": 0.8,
    "vz": -0.1,
    "heading": 0.785,
    "xy_valid": True,
    "z_valid": True
}

response = requests.post(
    f"{BASE_URL}/local-position-odom/",
    json=odom_data,
    headers={"Content-Type": "application/json"}
)

print(response.json())
```

### cURL

```bash
# Local Position Odometry
curl -X POST http://deepgis.org/api/telemetry/local-position-odom/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "flight_2024_11_23_001",
    "timestamp": "2024-11-23T01:00:00Z",
    "x": 10.5,
    "y": 5.2,
    "z": -2.1
  }'

# Raw GPS Fix
curl -X POST http://deepgis.org/api/telemetry/gps-fix-raw/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "flight_2024_11_23_001",
    "timestamp": "2024-11-23T01:00:00Z",
    "latitude": 33.4255,
    "longitude": -111.9400,
    "altitude": 352.5,
    "fix_type": 3,
    "satellites_visible": 12
  }'

# Estimated GPS Fix
curl -X POST http://deepgis.org/api/telemetry/gps-fix-estimated/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "flight_2024_11_23_001",
    "timestamp": "2024-11-23T01:00:00Z",
    "latitude": 33.4255,
    "longitude": -111.9400,
    "altitude": 352.3,
    "estimator_type": "EKF2"
  }'
```

### ROS2 Integration Example

```python
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleLocalPosition, VehicleGPSPosition
import requests
import json
from datetime import datetime

class TelemetryAPIBridge(Node):
    def __init__(self):
        super().__init__('telemetry_api_bridge')
        self.api_base_url = "http://deepgis.org/api/telemetry"
        self.session_id = "flight_2024_11_23_001"
        
        # Subscribe to PX4 topics
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.local_position_callback,
            10
        )
        
        self.gps_sub = self.create_subscription(
            VehicleGPSPosition,
            '/fmu/out/vehicle_gps_position',
            self.gps_callback,
            10
        )
    
    def local_position_callback(self, msg):
        """Convert VehicleLocalPosition to API format"""
        data = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "timestamp_usec": msg.timestamp,
            "x": float(msg.x),
            "y": float(msg.y),
            "z": float(msg.z),
            "vx": float(msg.vx) if msg.v_xy_valid else None,
            "vy": float(msg.vy) if msg.v_xy_valid else None,
            "vz": float(msg.vz) if msg.v_z_valid else None,
            "heading": float(msg.heading) if msg.heading_valid else None,
            "xy_valid": bool(msg.xy_valid),
            "z_valid": bool(msg.z_valid),
            "v_xy_valid": bool(msg.v_xy_valid),
            "v_z_valid": bool(msg.v_z_valid),
            "heading_valid": bool(msg.heading_valid),
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/local-position-odom/",
                json=data,
                timeout=1.0
            )
            if response.status_code == 201:
                self.get_logger().info("Posted local position odom")
        except Exception as e:
            self.get_logger().error(f"Failed to post telemetry: {e}")
    
    def gps_callback(self, msg):
        """Convert VehicleGPSPosition to API format"""
        data = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "timestamp_usec": msg.timestamp,
            "latitude": float(msg.lat) / 1e7,  # Convert from 1e7 degrees
            "longitude": float(msg.lon) / 1e7,
            "altitude": float(msg.alt) / 1000.0,  # Convert from mm to m
            "fix_type": int(msg.fix_type),
            "satellites_visible": int(msg.satellites_visible),
            "hdop": float(msg.hdop) / 100.0 if msg.hdop > 0 else None,
            "eph": float(msg.eph) / 100.0 if msg.eph > 0 else None,
            "epv": float(msg.epv) / 100.0 if msg.epv > 0 else None,
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/gps-fix-raw/",
                json=data,
                timeout=1.0
            )
            if response.status_code == 201:
                self.get_logger().info("Posted raw GPS fix")
        except Exception as e:
            self.get_logger().error(f"Failed to post GPS: {e}")

def main():
    rclpy.init()
    node = TelemetryAPIBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:

- **201 Created**: Successfully created
- **400 Bad Request**: Validation errors (missing/invalid fields)
- **500 Internal Server Error**: Server/database errors
- **207 Multi-Status**: Batch endpoint with partial success

Error responses include detailed error messages:

```json
{
  "error": "Validation failed",
  "errors": {
    "field_name": "Error message"
  }
}
```

---

## Timestamp Formats

The API accepts timestamps in multiple formats:

1. **ISO 8601 string**: `"2024-11-23T01:00:00Z"` or `"2024-11-23T01:00:00+00:00"`
2. **Unix timestamp (seconds)**: `1700707200`
3. **Unix timestamp (microseconds)**: `1700707200000000`
4. **Python datetime object**: Automatically converted

---

## Session Management

Before posting telemetry data, ensure a `DroneTelemetrySession` exists:

```python
from dreams_laboratory.models import DroneTelemetrySession, Asset
from django.utils import timezone

# Create a session
asset = Asset.objects.get(asset_name="PX4_Drone_01")
session = DroneTelemetrySession.objects.create(
    session_id="flight_2024_11_23_001",
    asset=asset,
    start_time=timezone.now(),
    flight_mode="POSCTL"
)
```

---

## Performance Considerations

- Use the **batch endpoint** for high-frequency data (50-100 Hz)
- Batch requests are processed in a single transaction
- Consider batching every 1-2 seconds for optimal performance
- The API uses `@csrf_exempt` for programmatic access (ensure proper authentication in production)

---

## Security Notes

⚠️ **Current Implementation**: The API uses `@csrf_exempt` for ease of integration. For production:

1. Implement API key authentication
2. Add rate limiting
3. Use HTTPS only
4. Consider IP whitelisting for known sources
5. Add request logging and monitoring

---

## Support

For issues or questions, contact the DREAMS Laboratory development team.

