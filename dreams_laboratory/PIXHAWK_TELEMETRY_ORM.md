# Pixhawk Telemetry ORM Design

## Overview

This document describes the ORM (Object-Relational Mapping) design for storing telemetry data from Pixhawk-based state estimators on drones. The design supports both raw sensor data and estimated/fused state data from the PX4 flight controller.

## Model Architecture

### 1. `DroneTelemetrySession`
**Purpose**: Represents a telemetry recording session or flight mission.

**Key Features**:
- Links telemetry data to a specific drone `Asset`
- Tracks session metadata (start/end time, duration, flight mode)
- Can be associated with a `Project`
- Maintains statistics (total telemetry points)

**Use Cases**:
- Organize telemetry data by flight/mission
- Query all data from a specific flight
- Track flight statistics and metadata

### 2. `LocalPositionOdom`
**Purpose**: Stores local position odometry data in NED (North-East-Down) frame.

**Data Source**: PX4 `VehicleLocalPosition` message

**Key Fields**:
- **Position**: `x`, `y`, `z` (meters in NED frame)
- **Velocity**: `vx`, `vy`, `vz` (m/s in NED frame)
- **Attitude**: `heading`, `heading_rate` (radians)
- **Covariance**: `position_covariance`, `velocity_covariance` (3x3 matrices as JSON)
- **Validity Flags**: `xy_valid`, `z_valid`, `v_xy_valid`, `v_z_valid`, `heading_valid`
- **Reference Frame**: `ref_lat`, `ref_lon`, `ref_alt` (origin of local frame)
- **Error Estimates**: `eph`, `epv`, `evh`, `evv` (position/velocity errors)

**Design Decisions**:
- Uses NED frame (standard for PX4)
- Stores covariance matrices as JSON arrays (9 elements for 3x3 matrix)
- Includes validity flags to indicate data quality
- Tracks reference frame origin for coordinate transformation

### 3. `GPSFixRaw`
**Purpose**: Stores raw GPS fix data directly from GPS receiver.

**Data Source**: PX4 `VehicleGPSPosition` message (raw GPS)

**Key Fields**:
- **Position**: `latitude`, `longitude`, `altitude` (WGS84)
- **Fix Quality**: `fix_type`, `satellites_visible`, `satellites_used`
- **DOP Values**: `hdop`, `vdop`, `pdop` (Dilution of Precision)
- **Accuracy**: `eph`, `epv`, `s_variance_m_s`
- **Velocity**: `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s`, `vel_m_s`, `cog_rad`
- **Jamming Detection**: `jamming_indicator`, `jamming_state`, `noise_per_ms`

**Design Decisions**:
- Separate model for raw GPS to distinguish from estimated
- Includes jamming detection for reliability assessment
- Stores all GPS quality metrics (DOP, accuracy, satellite count)

### 4. `GPSFixEstimated`
**Purpose**: Stores estimated GPS position from state estimator (filtered/fused).

**Data Source**: State estimator output (typically EKF2 or LPE)

**Key Fields**:
- **Estimated Position**: `latitude`, `longitude`, `altitude` (WGS84)
- **Estimated Velocity**: `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s`
- **Covariance**: `position_covariance`, `velocity_covariance` (uncertainty)
- **Quality Metrics**: `eph`, `epv`, `evh`, `evv` (error estimates)
- **Estimator Info**: `estimator_type`, `confidence`
- **References**: Links to `GPSFixRaw` and `LocalPositionOdom` for correlation

**Design Decisions**:
- Separate from raw GPS to enable comparison
- Includes covariance matrices for uncertainty quantification
- Links to raw GPS and local odom for data fusion analysis
- Tracks estimator type for algorithm comparison

## Database Design Considerations

### Indexing Strategy
All models include optimized indexes for:
- **Time-series queries**: `(session, timestamp)` composite indexes
- **Spatial queries**: `(latitude, longitude)` indexes on GPS models
- **Session lookups**: `(session, -timestamp)` for reverse chronological order

### Data Volume Management
For high-frequency telemetry (e.g., 50-100 Hz):
- Consider partitioning by time period (monthly/quarterly tables)
- Implement data retention policies
- Use database archiving for old sessions
- Consider time-series databases (TimescaleDB, InfluxDB) for very high-frequency data

### Performance Optimizations
1. **Batch Inserts**: Use `bulk_create()` for high-frequency data ingestion
2. **Selective Queries**: Use `select_related()` and `prefetch_related()` to avoid N+1 queries
3. **Query Optimization**: Use `only()` and `defer()` to limit field retrieval
4. **Connection Pooling**: Configure database connection pooling for concurrent writes

## Usage Examples

### Creating a Telemetry Session
```python
from dreams_laboratory.models import DroneTelemetrySession, Asset

# Get drone asset
drone = Asset.objects.get(asset_name="PX4_Drone_01")

# Create session
session = DroneTelemetrySession.objects.create(
    session_id="flight_2024_11_23_001",
    asset=drone,
    start_time=timezone.now(),
    flight_mode="POSCTL",
    mission_type="Mapping Survey"
)
```

### Storing Local Position Odometry
```python
from dreams_laboratory.models import LocalPositionOdom
import json

odom = LocalPositionOdom.objects.create(
    session=session,
    timestamp=timezone.now(),
    timestamp_usec=1234567890,
    x=10.5,  # North (m)
    y=5.2,   # East (m)
    z=-2.1,  # Down (m, positive down)
    vx=1.2,  # North velocity (m/s)
    vy=0.8,  # East velocity (m/s)
    vz=-0.1, # Down velocity (m/s)
    heading=0.785,  # 45 degrees (rad)
    xy_valid=True,
    z_valid=True,
    v_xy_valid=True,
    v_z_valid=True,
    position_covariance=[0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.02],  # 3x3 matrix
    velocity_covariance=[0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.1],
    ref_lat=33.4255,  # Reference frame origin
    ref_lon=-111.9400,
    ref_alt=350.0
)
```

### Storing Raw GPS Fix
```python
from dreams_laboratory.models import GPSFixRaw
from decimal import Decimal

gps_raw = GPSFixRaw.objects.create(
    session=session,
    timestamp=timezone.now(),
    timestamp_usec=1234567891,
    latitude=Decimal("33.4255000"),
    longitude=Decimal("-111.9400000"),
    altitude=352.5,
    fix_type=3,  # 3D fix
    satellites_visible=12,
    satellites_used=10,
    hdop=1.2,
    vdop=1.5,
    eph=2.5,  # Horizontal accuracy (m)
    epv=3.0,  # Vertical accuracy (m)
    vel_n_m_s=1.2,
    vel_e_m_s=0.8,
    vel_m_s=1.44,
    jamming_state=1  # OK
)
```

### Storing Estimated GPS Fix
```python
from dreams_laboratory.models import GPSFixEstimated

gps_est = GPSFixEstimated.objects.create(
    session=session,
    timestamp=timezone.now(),
    timestamp_usec=1234567892,
    latitude=Decimal("33.4255010"),  # Slightly different from raw
    longitude=Decimal("-111.9400010"),
    altitude=352.3,
    vel_n_m_s=1.18,  # Filtered velocity
    vel_e_m_s=0.82,
    position_covariance=[0.5, 0, 0, 0, 0.5, 0, 0, 0, 1.0],
    velocity_covariance=[0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.2],
    eph=1.8,  # Better than raw GPS
    epv=2.2,
    estimator_type="EKF2",
    confidence=0.95,
    position_valid=True,
    velocity_valid=True,
    raw_gps_fix=gps_raw,  # Link to raw GPS
    local_position=odom    # Link to local odom
)
```

### Querying Telemetry Data
```python
# Get all telemetry for a session
session = DroneTelemetrySession.objects.get(session_id="flight_2024_11_23_001")
odom_data = session.local_position_odom.all().order_by('timestamp')
gps_raw_data = session.gps_fixes_raw.all().order_by('timestamp')
gps_est_data = session.gps_fixes_estimated.all().order_by('timestamp')

# Get recent telemetry (last 5 minutes)
from datetime import timedelta
recent_time = timezone.now() - timedelta(minutes=5)
recent_odom = LocalPositionOdom.objects.filter(
    session=session,
    timestamp__gte=recent_time
).order_by('timestamp')

# Get GPS fixes with good quality
good_gps = GPSFixRaw.objects.filter(
    session=session,
    fix_type=3,  # 3D fix
    satellites_visible__gte=8,
    eph__lte=5.0  # Good horizontal accuracy
)

# Compare raw vs estimated GPS
gps_comparison = GPSFixEstimated.objects.filter(
    session=session,
    raw_gps_fix__isnull=False
).select_related('raw_gps_fix')
```

## Integration with PX4/ROS2

### ROS2 Message Mapping

**VehicleLocalPosition → LocalPositionOdom**:
- `msg.x` → `x`
- `msg.y` → `y`
- `msg.z` → `z`
- `msg.vx` → `vx`
- `msg.vy` → `vy`
- `msg.vz` → `vz`
- `msg.heading` → `heading`
- `msg.xy_valid` → `xy_valid`
- `msg.z_valid` → `z_valid`
- `msg.xy_global` → `ref_lat`, `ref_lon` (if available)

**VehicleGPSPosition → GPSFixRaw**:
- `msg.lat` → `latitude`
- `msg.lon` → `longitude`
- `msg.alt` → `altitude`
- `msg.fix_type` → `fix_type`
- `msg.satellites_visible` → `satellites_visible`
- `msg.hdop` → `hdop`
- `msg.eph` → `eph`
- `msg.vel_n_m_s` → `vel_n_m_s`

### Recommended ROS2 Bridge Implementation

Create a ROS2 node that:
1. Subscribes to `/fmu/out/vehicle_local_position` → Store in `LocalPositionOdom`
2. Subscribes to `/fmu/out/vehicle_gps_position` → Store in `GPSFixRaw`
3. Subscribes to state estimator output → Store in `GPSFixEstimated`
4. Uses `bulk_create()` for efficient batch inserts (e.g., every 1 second)
5. Handles timestamp conversion (PX4 microseconds to Django DateTime)

## Migration and Deployment

### Creating Migrations
```bash
python manage.py makemigrations dreams_laboratory
python manage.py migrate
```

### Admin Interface
All models are registered in Django admin with:
- List displays showing key fields
- Filters for common queries
- Search functionality
- Date hierarchies for time-based navigation
- Organized fieldsets for easy data entry

## Future Enhancements

1. **Time-Series Database Integration**: Consider TimescaleDB for better time-series performance
2. **Data Compression**: Implement compression for covariance matrices
3. **Real-Time Streaming**: Add WebSocket support for live telemetry visualization
4. **Analytics**: Add computed fields for trajectory analysis, error statistics
5. **Export Formats**: Add methods to export to common formats (CSV, KML, GeoJSON)
6. **Data Validation**: Add model-level validation for coordinate ranges, timestamp consistency
7. **Partitioning**: Implement table partitioning for large datasets

## Performance Benchmarks

For reference, expected performance:
- **Insert Rate**: ~1000 records/second (with bulk_create)
- **Query Time**: <100ms for 1-minute time range (with proper indexes)
- **Storage**: ~500 bytes per LocalPositionOdom record, ~400 bytes per GPSFixRaw record

## Related Models

- **Asset**: Represents the drone hardware
- **Project**: Can be associated with telemetry sessions
- **People**: Asset owners/operators

## Notes

- All timestamps use Django's timezone-aware DateTimeField
- GPS coordinates use DecimalField for precision (17 digits, 14 decimal places)
- Covariance matrices stored as JSON arrays (9 elements for 3x3 matrices)
- Validity flags help filter unreliable data
- Reference frame tracking enables coordinate transformations

