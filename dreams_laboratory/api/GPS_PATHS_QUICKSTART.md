# GPS Paths on DeepGIS Frontend - Quick Start

## Overview

GPS telemetry session paths can now be displayed on the DeepGIS 3D viewer at `https://deepgis.org/label/3d/topology/legacy/`.

## How to Use

### 1. Access the DeepGIS Viewer

Navigate to: `https://deepgis.org/label/3d/topology/legacy/`

### 2. Find the GPS Telemetry Section

In the left sidebar, look for the **"GPS Telemetry Paths"** section (with a green border and satellite icon).

### 3. Load a Session Path

1. **Select a Session**: Use the dropdown to choose a GPS telemetry session
   - Sessions are automatically loaded from the API
   - Only sessions with GPS data are shown
   - Each session shows: Asset name, date/time, and number of GPS points

2. **Load Path**: Click the **"Load Path"** button to display the complete GPS track as a cyan line on the 3D globe

3. **Load Points** (optional): Click **"Load Points"** to show individual GPS fix locations as colored markers
   - Green = 3D fix (best quality)
   - Yellow = 2D fix
   - Orange = Dead reckoning
   - Red = No fix

4. **Fly To Path**: Click **"Fly To Path"** to automatically navigate the camera to view the entire path

5. **Clear All**: Click **"Clear All"** to remove all loaded GPS paths and points

### 4. Interact with the Path

- **Click on the path line** to see session information (asset, project, flight mode, etc.)
- **Click on individual points** to see detailed GPS information (timestamp, altitude, fix type, accuracy, etc.)
- **Use the 3D navigation controls** to zoom, pan, and rotate around the path

## Features

- **Automatic Session Loading**: Sessions are fetched from `/api/telemetry/sessions/?has_gps=true`
- **GeoJSON Format**: Paths are loaded in standard GeoJSON format for compatibility
- **Color-Coded Points**: GPS fix quality is indicated by point color
- **Detailed Information**: Click any path or point to see metadata
- **3D Visualization**: Paths are displayed in 3D space with proper altitude

## Troubleshooting

### No Sessions Appear

- Check that GPS data has been posted to the API
- Verify the API endpoint is accessible: `https://deepgis.org/api/telemetry/sessions/?has_gps=true`
- Check browser console for errors

### Path Doesn't Load

- Check browser console for error messages
- Verify the session has GPS data (check the point count in the dropdown)
- Ensure the API endpoint is working: `https://deepgis.org/api/telemetry/sessions/{session_id}/path/`

### Points Don't Show

- Make sure you clicked "Load Points" after loading the path
- Check that the session has GPS point data
- Verify browser console for any JavaScript errors

## API Endpoints Used

- `GET /api/telemetry/sessions/?has_gps=true` - List all sessions with GPS data
- `GET /api/telemetry/sessions/{session_id}/path/` - Get GeoJSON path data for a session

## Technical Details

The GPS path loader:
- Uses Cesium.js for 3D visualization
- Converts GeoJSON coordinates to Cesium Cartesian3 positions
- Creates polyline entities for paths
- Creates point entities with labels for individual GPS fixes
- Provides interactive information panels on click

## Next Steps

For advanced features like:
- Time-based animation
- Multiple session comparison
- Custom styling
- Export functionality

See `DEEPGIS_INTEGRATION.md` for detailed code examples.

