# DeepGIS Integration Guide

This guide explains how to integrate GPS telemetry session paths from the Dreams Laboratory API with the DeepGIS XR frontend at https://deepgis.org/label/3d/topology/legacy/.

## API Endpoints

### 1. List All Sessions
**GET** `/api/telemetry/sessions/`

Returns a list of all telemetry sessions with metadata.

**Query Parameters:**
- `asset_name` (optional): Filter by asset name
- `project_title` (optional): Filter by project title
- `has_gps` (optional): Set to `true` to only show sessions with GPS data

**Example Request:**
```javascript
fetch('https://deepgis.org/api/telemetry/sessions/?has_gps=true')
  .then(response => response.json())
  .then(data => console.log(data));
```

**Example Response:**
```json
{
  "sessions": [
    {
      "session_id": "test_session_20251123_170411_u39pa9jh",
      "asset": "RV Karin Valentine",
      "project": "Tempe Town Lake Survey",
      "start_time": "2025-11-23T17:04:11.318489Z",
      "end_time": null,
      "flight_mode": "AUTO",
      "mission_type": "Lake Survey",
      "gps_point_count": 100,
      "has_gps_data": true,
      "path_url": "/api/telemetry/sessions/test_session_20251123_170411_u39pa9jh/path/"
    }
  ],
  "count": 1
}
```

### 2. Get Session Path
**GET** `/api/telemetry/sessions/<session_id>/path/`

Returns GPS path data for a specific session in GeoJSON format.

**Query Parameters:**
- `format` (optional): `geojson` (default) or `points`
- `include_properties` (optional): `true` (default) or `false`

**Example Request:**
```javascript
fetch('https://deepgis.org/api/telemetry/sessions/test_session_20251123_170411_u39pa9jh/path/')
  .then(response => response.json())
  .then(data => console.log(data));
```

**Example Response (GeoJSON format):**
```json
{
  "session_id": "test_session_20251123_170411_u39pa9jh",
  "session_info": {
    "asset": "RV Karin Valentine",
    "project": "Tempe Town Lake Survey",
    "start_time": "2025-11-23T17:04:11.318489Z",
    "end_time": null,
    "flight_mode": "AUTO",
    "mission_type": "Lake Survey",
    "total_points": 100
  },
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [-111.939419941312, 33.4254790767595, 350.19366504993184],
            [-111.939419941312, 33.4254790767595, 350.19366504993184],
            ...
          ]
        },
        "properties": {
          "point_count": 100,
          "type": "path"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [-111.939419941312, 33.4254790767595, 350.19366504993184]
        },
        "properties": {
          "timestamp": "2025-11-23T17:05:00.000Z",
          "altitude": 350.19,
          "fix_type": 3,
          "satellites_visible": 12,
          "eph": 2.5,
          "epv": 3.1
        }
      },
      ...
    ]
  }
}
```

## Cesium Integration

The DeepGIS frontend uses Cesium for 3D visualization. Here's how to load and display GPS session paths:

### Basic Integration

```javascript
// Function to load and display a GPS session path
async function loadGPSSessionPath(sessionId, viewer) {
  try {
    // Fetch session path data
    const response = await fetch(
      `https://deepgis.org/api/telemetry/sessions/${sessionId}/path/`
    );
    const data = await response.json();
    
    if (!data.geojson || !data.geojson.features) {
      console.error('No GeoJSON data found');
      return;
    }
    
    // Find the LineString feature (path)
    const pathFeature = data.geojson.features.find(
      f => f.geometry.type === 'LineString'
    );
    
    if (pathFeature) {
      // Load the path as a Cesium entity
      const pathEntity = viewer.entities.add({
        name: `GPS Path: ${data.session_info.asset} - ${data.session_id}`,
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(
            pathFeature.geometry.coordinates.flat()
          ),
          width: 3,
          material: Cesium.Color.CYAN.withAlpha(0.8),
          clampToGround: false,
          heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND
        }
      });
      
      // Fly to the path
      viewer.flyTo(pathEntity);
      
      return pathEntity;
    }
  } catch (error) {
    console.error('Error loading GPS session path:', error);
  }
}

// Function to load GPS points as markers
async function loadGPSPoints(sessionId, viewer) {
  try {
    const response = await fetch(
      `https://deepgis.org/api/telemetry/sessions/${sessionId}/path/`
    );
    const data = await response.json();
    
    if (!data.geojson || !data.geojson.features) {
      return;
    }
    
    // Get all Point features
    const pointFeatures = data.geojson.features.filter(
      f => f.geometry.type === 'Point'
    );
    
    pointFeatures.forEach((feature, index) => {
      const [lon, lat, alt] = feature.geometry.coordinates;
      const props = feature.properties;
      
      viewer.entities.add({
        name: `GPS Point ${index + 1}`,
        position: Cesium.Cartesian3.fromDegrees(lon, lat, alt),
        point: {
          pixelSize: 5,
          color: getColorForFixType(props.fix_type),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 1,
          heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND
        },
        label: {
          text: `${index + 1}`,
          font: '10pt sans-serif',
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM
        },
        description: `
          <table>
            <tr><td>Timestamp:</td><td>${props.timestamp}</td></tr>
            <tr><td>Altitude:</td><td>${props.altitude?.toFixed(2)} m</td></tr>
            <tr><td>Fix Type:</td><td>${props.fix_type}</td></tr>
            <tr><td>Satellites:</td><td>${props.satellites_visible}</td></tr>
            <tr><td>Accuracy (H):</td><td>${props.eph?.toFixed(2)} m</td></tr>
            <tr><td>Accuracy (V):</td><td>${props.epv?.toFixed(2)} m</td></tr>
          </table>
        `
      });
    });
  } catch (error) {
    console.error('Error loading GPS points:', error);
  }
}

// Helper function to get color based on GPS fix type
function getColorForFixType(fixType) {
  const colors = {
    0: Cesium.Color.RED,      // No fix
    1: Cesium.Color.ORANGE,   // Dead reckoning
    2: Cesium.Color.YELLOW,    // 2D fix
    3: Cesium.Color.GREEN,     // 3D fix
    4: Cesium.Color.CYAN,      // GPS+DR
    5: Cesium.Color.MAGENTA    // Time only
  };
  return colors[fixType] || Cesium.Color.WHITE;
}
```

### Complete Example: Session Selector UI

```javascript
// Create a UI to select and load sessions
async function createSessionSelector(viewer) {
  // Fetch available sessions
  const response = await fetch(
    'https://deepgis.org/api/telemetry/sessions/?has_gps=true'
  );
  const data = await response.json();
  
  // Create dropdown or list UI
  const sessionList = document.createElement('div');
  sessionList.id = 'gps-session-selector';
  sessionList.style.cssText = `
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(42, 42, 42, 0.9);
    padding: 15px;
    border-radius: 5px;
    color: white;
    max-width: 300px;
    z-index: 1000;
  `;
  
  sessionList.innerHTML = `
    <h3 style="margin-top: 0;">GPS Sessions</h3>
    <select id="session-select" style="width: 100%; padding: 5px;">
      <option value="">Select a session...</option>
      ${data.sessions.map(s => `
        <option value="${s.session_id}">
          ${s.asset} - ${new Date(s.start_time).toLocaleString()}
          (${s.gps_point_count} points)
        </option>
      `).join('')}
    </select>
    <button id="load-path-btn" style="width: 100%; margin-top: 10px; padding: 8px;">
      Load Path
    </button>
    <button id="load-points-btn" style="width: 100%; margin-top: 5px; padding: 8px;">
      Load Points
    </button>
    <button id="clear-btn" style="width: 100%; margin-top: 5px; padding: 8px;">
      Clear
    </button>
  `;
  
  document.body.appendChild(sessionList);
  
  let currentEntities = [];
  
  // Load path button handler
  document.getElementById('load-path-btn').addEventListener('click', async () => {
    const sessionId = document.getElementById('session-select').value;
    if (!sessionId) return;
    
    // Clear previous entities
    currentEntities.forEach(e => viewer.entities.remove(e));
    currentEntities = [];
    
    const entity = await loadGPSSessionPath(sessionId, viewer);
    if (entity) {
      currentEntities.push(entity);
    }
  });
  
  // Load points button handler
  document.getElementById('load-points-btn').addEventListener('click', async () => {
    const sessionId = document.getElementById('session-select').value;
    if (!sessionId) return;
    
    await loadGPSPoints(sessionId, viewer);
    // Note: You may want to track these entities separately for clearing
  });
  
  // Clear button handler
  document.getElementById('clear-btn').addEventListener('click', () => {
    currentEntities.forEach(e => viewer.entities.remove(e));
    currentEntities = [];
    viewer.entities.removeAll(); // Or be more selective
  });
}

// Initialize when Cesium viewer is ready
if (typeof viewer !== 'undefined') {
  createSessionSelector(viewer);
}
```

## Advanced Features

### Time-Based Animation

You can animate the path based on timestamps:

```javascript
async function animateGPSSession(sessionId, viewer) {
  const response = await fetch(
    `https://deepgis.org/api/telemetry/sessions/${sessionId}/path/`
  );
  const data = await response.json();
  
  const points = data.geojson.features
    .filter(f => f.geometry.type === 'Point')
    .map(f => ({
      position: Cesium.Cartesian3.fromDegrees(
        f.geometry.coordinates[0],
        f.geometry.coordinates[1],
        f.geometry.coordinates[2]
      ),
      time: Cesium.JulianDate.fromIso8601(f.properties.timestamp)
    }));
  
  // Create a time-dynamic position
  const positionProperty = new Cesium.SampledPositionProperty();
  points.forEach(point => {
    positionProperty.addSample(point.time, point.position);
  });
  
  // Create entity with time-based position
  const entity = viewer.entities.add({
    name: 'GPS Path Animation',
    position: positionProperty,
    point: {
      pixelSize: 10,
      color: Cesium.Color.YELLOW,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2
    },
    path: {
      resolution: 1,
      material: new Cesium.PolylineGlowMaterialProperty({
        glowPower: 0.1,
        color: Cesium.Color.CYAN
      }),
      width: 3
    }
  });
  
  // Set timeline to session time range
  const startTime = Cesium.JulianDate.fromIso8601(data.session_info.start_time);
  const endTime = data.session_info.end_time 
    ? Cesium.JulianDate.fromIso8601(data.session_info.end_time)
    : points[points.length - 1].time;
  
  viewer.timeline.zoomTo(startTime, endTime);
  viewer.clock.startTime = startTime.clone();
  viewer.clock.stopTime = endTime.clone();
  viewer.clock.currentTime = startTime.clone();
  viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
  viewer.clock.multiplier = 1;
}
```

### Styling Based on GPS Quality

```javascript
function createQualityBasedPath(sessionId, viewer) {
  // Load data and create path segments colored by fix quality
  // Red for poor quality, green for good quality
  // Implementation similar to above but with dynamic material colors
}
```

## CORS Configuration

If the DeepGIS frontend is on a different domain, ensure CORS is properly configured in Django settings:

```python
# settings.py
CORS_ALLOWED_ORIGINS = [
    "https://deepgis.org",
    "http://localhost:8000",  # For development
]

CORS_ALLOW_METHODS = [
    'GET',
    'OPTIONS',
]
```

## Testing

Test the endpoints using curl:

```bash
# List all sessions
curl https://deepgis.org/api/telemetry/sessions/?has_gps=true

# Get a specific session path
curl https://deepgis.org/api/telemetry/sessions/test_session_20251123_170411_u39pa9jh/path/
```

## Notes

- GPS coordinates are in WGS84 (standard GPS format)
- Altitude is in meters above MSL (Mean Sea Level)
- Timestamps are in ISO 8601 format (UTC)
- The LineString feature represents the complete path
- Individual Point features represent each GPS fix with metadata
- Fix type: 0=no fix, 1=dead reckoning, 2=2D, 3=3D, 4=GPS+DR, 5=Time only

