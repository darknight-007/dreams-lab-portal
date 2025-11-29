# DeepGIS-XR URL Pattern Analysis

## URL: https://deepgis.org/label/3d/topology/legacy/

**Analysis Date:** November 27, 2025  
**Analyst:** AI Code Assistant

---

## Executive Summary

The URL `https://deepgis.org/label/3d/topology/legacy/` points to the **DeepGIS Decision Support System (DSS)**, a sophisticated hybrid 2D/3D geospatial visualization platform built with CesiumJS. This is the "legacy" version of the topology viewer, representing the original implementation before refactoring to the SIGMA architecture.

---

## 1. URL Structure & Routing

### 1.1 URL Breakdown

```
https://deepgis.org/label/3d/topology/legacy/
│        │         │    │   │         │
│        │         │    │   │         └─ Endpoint: Legacy version
│        │         │    │   └─ Feature: Topology viewer
│        │         │    └─ Dimension: 3D capabilities
│        │         └─ Domain: Label/annotation tools
│        └─ Domain: DeepGIS.org
└─ Protocol: HTTPS
```

### 1.2 Nginx Reverse Proxy Configuration

**File:** `/home/jdas/dreams-lab-website-server/nginx.conf`

```nginx
server {
    server_name deepgis.org www.deepgis.org;
    listen 443 ssl;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/deepgis.org-0001/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/deepgis.org-0001/privkey.pem;
    
    # Route /label/* to deepgis-xr Django app on port 8060
    location /label/ {
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_pass http://localhost:8060;
        proxy_http_version 1.1;
    }
}
```

**Key Points:**
- HTTPS (port 443) with Let's Encrypt SSL certificates
- Proxies `/label/*` requests to Django app on `localhost:8060`
- Static files served from `/home/jdas/dreams-lab-website-server/staticfiles/`

---

## 2. Django Routing Architecture

### 2.1 URL Configuration Hierarchy

```
deepgis-xr/
├── deepgis_xr/
│   ├── urls.py                    # Root URL config
│   └── apps/
│       └── web/
│           ├── urls.py            # Web app URLs (defines /label/3d/topology/legacy/)
│           └── views.py           # View functions
```

### 2.2 URL Pattern Definition

**File:** `deepgis-xr/deepgis_xr/apps/web/urls.py` (Lines 13-15)

```python
urlpatterns = [
    # ... other routes ...
    
    # Line 13: Main topology route (now uses SIGMA/refactored)
    path('label/3d/topology/', views.label_topology_sigma, name='label_topology'),
    
    # Line 14: Legacy topology route (original version)
    path('label/3d/topology/legacy/', views.label_topology, name='label_topology_legacy'),
    
    # Line 15: Explicit SIGMA route
    path('label/3d/topology/sigma/', views.label_topology_sigma, name='label_topology_sigma'),
    
    # ... other routes ...
]
```

**Related Routes:**
- `/label/3d/` - Basic 3D labeling viewer
- `/label/3d/dev/` - Development version of 3D viewer
- `/label/3d/sigma/` - SIGMA 3D viewer
- `/label/3d/topology/` - Main topology viewer (SIGMA-based)
- `/label/3d/topology/legacy/` - **Legacy topology viewer** ⬅️ Our focus
- `/label/3d/topology/sigma/` - Explicit SIGMA topology viewer
- `/label/3d/search/` - DeepGIS search viewer
- `/label/3d/moon/` - Moon viewer

---

## 3. View Function Implementation

### 3.1 View Handler

**File:** `deepgis-xr/deepgis_xr/apps/web/views.py` (Lines 1040-1076)

```python
def label_topology(request):
    """
    Cesium hybrid 2D/3D viewer combining the best features from both 2D and 3D applications.
    This view provides:
        - 2D/3D/Columbus view modes
        - Advanced measurement tools
        - 3D terrain and models support
        - Temporal data layers
        - Real-time statistics
        - Performance monitoring
    """
    context = {}
    
    # Add viewer configuration information
    viewer_info = {
        'version': 'Hybrid',
        'engine': 'Cesium.js',
        'features': [
            'Multiple view modes (2D/3D/Columbus)',
            'Interactive measurement tools',
            '3D terrain visualization', 
            'GLTF/GLB model loading',
            'Temporal data layer support',
            'Real-time performance monitoring',
            'Responsive design with mobile support'
        ],
        'data_sources': [
            'Custom MBTiles layers',
            'OpenStreetMap',
            'Satellite imagery',
            'World terrain data',
            '3D models (GLTF/GLB)'
        ]
    }
    context['viewer_info'] = viewer_info
    
    return render(request, 'web/label_topology.html', context)
```

### 3.2 Template Rendering

**Template:** `deepgis-xr/deepgis_xr/apps/web/templates/web/label_topology.html`

**Key Characteristics:**
- Full HTML5 application with responsive design
- Mobile-friendly with viewport meta tags
- Server-side rendering with Django template tags
- Client-side JavaScript SPA (Single Page Application) behavior

---

## 4. Frontend Architecture

### 4.1 Technology Stack

| Layer | Technology | Version/Notes |
|-------|-----------|---------------|
| **3D Engine** | CesiumJS | v1.111 |
| **UI Framework** | Bootstrap | v5.3.0 |
| **Icons** | Font Awesome | v6.4.0 |
| **Charts** | Chart.js | v4.4.1 |
| **Module System** | ES6 Modules | Native browser support |
| **Build Tool** | Vite | For bundling |
| **CSS** | Custom CSS + Bootstrap | Modular architecture |

### 4.2 JavaScript Module Structure

**Entry Point:** `staticfiles/web/js/main.js`

```javascript
// Main imports
import { CONFIG } from './config.js';
import { AppState } from './state.js';
import { initializeCesium } from './core/cesium-init.js';
import { initializeAvailableLayers, loadBaseRasterLayer, toggleOverlayLayer } 
    from './core/layer-management.js';
import { toggleTerrain, changeBaseMap } from './core/base-map.js';
import { updateStatusIndicator, showSnackBar, logLayerOperation } 
    from './core/ui-helpers.js';
import { optimizeMemorySettings } from './core/memory-manager.js';
```

**Module Breakdown:**

```
staticfiles/web/
├── js/
│   ├── main.js                    # Entry point with lazy loading
│   ├── config.js                  # Application configuration
│   ├── state.js                   # Global state management
│   ├── gps-telemetry.js          # GPS telemetry path loader
│   ├── core/
│   │   ├── cesium-init.js        # Cesium viewer initialization
│   │   ├── layer-management.js   # Raster/vector layer management
│   │   ├── base-map.js           # Base map and terrain controls
│   │   ├── ui-helpers.js         # UI utility functions
│   │   └── memory-manager.js     # Memory optimization
│   ├── features/
│   │   ├── webxr.js              # WebXR/VR support (lazy loaded)
│   │   ├── models.js             # 3D model loading (lazy loaded)
│   │   ├── measurements.js       # Measurement tools (lazy loaded)
│   │   ├── statistics.js         # Statistical analysis (lazy loaded)
│   │   └── debug-console.js      # Debug console (lazy loaded)
│   ├── utils/
│   │   ├── astronomy.js          # Sun/moon calculations
│   │   ├── camera.js             # Camera utilities
│   │   ├── coordinates.js        # Coordinate transformations
│   │   ├── errors.js             # Error handling
│   │   ├── layers.js             # Layer utilities
│   │   ├── vector-tiles.js       # Vector tile processing
│   │   └── chunked-loading.js    # Progressive tile loading
│   └── widgets/
│       └── navigation.js          # Navigation widgets (heading dial, attitude indicator)
└── styles/
    ├── main.css                   # Main stylesheet
    ├── base.css                   # Base styles
    ├── components.css             # Component styles
    ├── sidebar.css                # Sidebar styles
    ├── widgets.css                # Widget styles
    ├── loading.css                # Loading animations
    ├── responsive.css             # Mobile responsiveness
    └── debug.css                  # Debug console styles
```

### 4.3 Lazy Loading Strategy

The application uses **dynamic ES6 imports** for code splitting:

```javascript
const lazyLoaders = {
  webxr: async () => import('./features/webxr.js'),
  models: async () => import('./features/models.js'),
  measurements: async () => import('./features/measurements.js'),
  debug: async () => import('./features/debug-console.js'),
  statistics: async () => import('./features/statistics.js'),
  navigation: async () => import('./widgets/navigation.js'),
  astronomy: async () => import('./utils/astronomy.js')
};
```

**Benefits:**
- Initial page load is faster
- Features loaded on-demand when user activates them
- Reduces memory footprint
- Better performance on mobile devices

---

## 5. Configuration & Settings

### 5.1 Application Configuration

**File:** `staticfiles/web/js/config.js`

```javascript
export const CONFIG = {
  SIDEBAR_WIDTH: 320,
  MAX_OVERLAY_LAYERS: 3,
  MAX_SAFE_CAMERA_HEIGHT: 5000000,      // 5,000 km
  MAX_2D_VIEW_HEIGHT: 10000000,         // 10,000 km
  DEFAULT_ZOOM_HEIGHT_BASE: 40000000,
  DEFAULT_ZOOM_LEVEL: 20,
  TILE_DIMENSIONS: { width: 256, height: 256 },
  
  MEMORY: {
    TILE_CACHE_SIZE: 300,
    MAX_SCREEN_SPACE_ERROR: 6,
    MAX_ZOOM_CAPS: {
      DEFAULT: 18,
      HIGH: 16,
      VERY_HIGH: 13
    },
    CHUNKED_LOADING: {
      ENABLED: false,  // Currently disabled due to freezing issues
      INITIAL_MAX_ZOOM: 12,
      ZOOM_INCREMENT: 2,
      DELAY_BETWEEN_CHUNKS: 500,
      MAX_TILES_PER_CHUNK: 50
    },
    MODE_SPECIFIC: {
      SCENE2D: {
        TILE_CACHE_SIZE: 150,
        MAX_SCREEN_SPACE_ERROR: 4,
        DESCRIPTION: '2D flat map view - optimized for memory and quality'
      },
      SCENE3D: {
        TILE_CACHE_SIZE: 300,
        MAX_SCREEN_SPACE_ERROR: 6,
        DESCRIPTION: '3D globe view - balanced performance and quality'
      },
      COLUMBUS_VIEW: {
        TILE_CACHE_SIZE: 225,
        MAX_SCREEN_SPACE_ERROR: 5,
        DESCRIPTION: 'Columbus view (2.5D) - hybrid between 2D and 3D'
      }
    }
  },
  
  SERVERS: {
    MBTILES_SERVER: 'https://mbtiles.deepgis.org',
    TOPOLOGY_SERVER: 'https://localhost:8092'
  },
  
  CESIUM_ION_TOKEN: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
  
  VECTOR_TILES: {
    ENABLED: true,
    SHOW_BOUNDING_BOX: false,
    MAX_ZOOM: 14,
    MAX_CACHED_TILES: 100,
    MAX_TILES_PER_DIMENSION: 4,
    DEFAULT_STYLE: {
      fillColor: { r: 0.4, g: 0.6, b: 0.8 },
      fillOpacity: 0.5,
      strokeColor: { r: 0.2, g: 0.3, b: 0.5 },
      strokeWidth: 2,
      pointSize: 8
    }
  },
  
  RASTER: {
    SHOW_BOUNDING_BOX: false
  }
};
```

### 5.2 External Services

| Service | URL | Purpose |
|---------|-----|---------|
| **MBTiles Server** | `https://mbtiles.deepgis.org` | Serves raster tile layers (MBTiles format) |
| **Static Tiles Server** | `https://statictiles.deepgis.org` | Alternative tile server |
| **Rock Tiles Server** | `https://rocks.deepgis.org` | Geological/rock sample tiles |
| **3D Models** | `https://deepgis.org/static/deepgis/models/` | GLTF/GLB 3D model files |
| **Cesium Ion** | `https://cesium.com` | Cesium terrain and 3D tiles |

---

## 6. Key Features & Capabilities

### 6.1 Viewer Modes

1. **2D Map View** - Traditional flat map interface
2. **3D Globe View** - Full 3D Earth visualization
3. **Columbus View** - 2.5D perspective (flat with perspective)

### 6.2 Layer Management

#### Raster Layers
- **Base Layers:** Ion Satellite, Ion Streets, Bing Satellite, Bing Streets, Mapbox Streets
- **Overlay Layers:** Custom MBTiles served from `mbtiles.deepgis.org`
- **Format:** MBTiles (SQLite-based tile storage)
- **Auto-discovery:** Fetches available layers from `/webclient/getTileserverLayers`

#### Vector Layers
- **Format:** GeoJSON, vector tiles
- **Styling:** Customizable per-layer (color, opacity, stroke)
- **Features:** Points, lines, polygons
- **Rendering:** Cesium entities (not primitive-based)

#### 3D Features
- **Terrain:** Cesium World Terrain (via Ion)
- **3D Models:** GLTF/GLB support
  - Featured model: Navagunjara Reborn Digital Twin (propane and solar-powered art installation)
  - Location options: Mount Everest, current view, custom coordinates
  - Scalable (0.1x - 10x)

### 6.3 Measurement Tools

- **Distance Measurement:** Click points to measure distances
- **Area Measurement:** Define polygons to calculate area
- **Height Measurement:** Measure terrain elevation changes
- **Real-time display:** Updates as you add points
- **Persistent storage:** Measurements saved during session

### 6.4 GPS Telemetry Integration

**File:** `staticfiles/web/js/gps-telemetry.js`

```javascript
class GPSTelemetryLoader {
    constructor(viewer) {
        this.viewer = viewer;
        this.apiBaseUrl = window.location.origin + '/api/telemetry';
        this.loadedEntities = [];
        this.currentSession = null;
    }
}
```

**Features:**
- Loads GPS session paths from Dreams Laboratory API (`/api/telemetry`)
- Displays as polylines on the 3D globe
- Session selection dropdown
- Fly-to path functionality
- Color-coded paths
- Metadata display (date, distance, duration)

**Integration Point:**
- Automatically initializes when Cesium viewer is ready
- UI injected into sidebar dynamically
- Non-intrusive (won't break if API unavailable)

### 6.5 WebXR/VR Support

- **Check VR Support:** Detects WebXR capabilities
- **Enter VR:** Launches immersive VR session
- **Exit VR:** Returns to desktop view
- **Status Display:** Shows VR availability and session state

### 6.6 Camera & Navigation

#### Real-time Camera Info Display
- Longitude, Latitude, Altitude
- Zoom level
- Ground height
- AGL (Above Ground Level)
- Heading, Pitch, Roll

#### Navigation Widgets
- **Heading Dial:** Compass-style orientation display
- **Attitude Indicator:** Pitch and roll visualization (aircraft-style)

#### Sun & Moon Information
- Sun azimuth and elevation
- Moon azimuth and elevation
- Moon phase

### 6.7 Performance Monitoring

- **FPS Counter:** Real-time frame rate display
- **Memory Management:** Automatic tile cache optimization
- **Mode-specific settings:** Different cache sizes for 2D/3D/Columbus
- **Status indicators:** Loading states and error messages

---

## 7. Data Flow Architecture

### 7.1 Layer Discovery Flow

```
Browser
   │
   ├─▶ GET /label/3d/topology/legacy/
   │        │
   │        └─▶ Django view: label_topology()
   │                │
   │                └─▶ Renders: label_topology.html
   │                        │
   │                        └─▶ Loads: main.js
   │
   ├─▶ GET /webclient/getTileserverLayers
   │        │
   │        └─▶ Django API returns available MBTiles layers
   │
   └─▶ GET https://mbtiles.deepgis.org/{layer}/{z}/{x}/{y}.png
            │
            └─▶ Tile server returns raster tile
```

### 7.2 Tile Loading Flow

```
User selects layer
      │
      ├─▶ Fetch metadata: /webclient/getRasterInfo?id={layer_id}
      │        │
      │        └─▶ Returns: { bounds, minzoom, maxzoom, center }
      │
      ├─▶ Create Cesium ImageryLayer
      │        │
      │        └─▶ UrlTemplateImageryProvider
      │                 │
      │                 └─▶ Requests tiles:
      │                     https://mbtiles.deepgis.org/{layer}/{z}/{x}/{y}.png
      │
      └─▶ Add to viewer.imageryLayers collection
```

### 7.3 GPS Telemetry Flow

```
Page loads
    │
    └─▶ GPS Telemetry Loader initializes
            │
            ├─▶ GET /api/telemetry/sessions/
            │        │
            │        └─▶ Returns: [{ id, date, metadata }]
            │
            └─▶ User selects session
                    │
                    └─▶ GET /api/telemetry/sessions/{id}/paths/
                            │
                            └─▶ Returns: { coordinates: [[lon,lat,alt],...] }
                                    │
                                    └─▶ Creates Cesium PolylineCollection
                                            │
                                            └─▶ Displays on globe
```

---

## 8. Static File Organization

### 8.1 Static File Routing

**Nginx Configuration:**

```nginx
location /static/deepgis/ {
    alias /home/jdas/dreams-lab-website-server/deepgis-xr/staticfiles/;
    expires 30d;
    add_header Cache-Control "public, no-transform";
    add_header Access-Control-Allow-Origin *;
}

location /static/deepgis/models/ {
    alias /home/jdas/dreams-lab-website-server/deepgis-xr/staticfiles/models/;
    expires 7d;
    add_header Cache-Control "public, no-transform, immutable";
    
    # CORS headers for 3D models
    add_header Access-Control-Allow-Origin "*" always;
    add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
    
    # Enable range requests for large files
    add_header Accept-Ranges bytes always;
    
    # GLB MIME type
    location ~* \.glb$ {
        add_header Content-Type "model/gltf-binary" always;
    }
}
```

### 8.2 Django Static Settings

**File:** `deepgis-xr/deepgis_xr/settings.py`

```python
STATIC_URL = '/static/deepgis/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'deepgis_xr', 'static'),
]

MEDIA_URL = '/media/deepgis/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

---

## 9. Security & Performance

### 9.1 Security Features

- **HTTPS Only:** Enforced via nginx (HTTP → HTTPS redirect)
- **SSL Certificates:** Let's Encrypt automated certificates
- **CSRF Protection:** Django CSRF middleware enabled
- **CORS Headers:** Configured for cross-origin tile requests
- **Proxy Headers:** X-Real-IP, X-Forwarded-For, X-Forwarded-Proto

### 9.2 Performance Optimizations

#### Nginx Level
- **Gzip Compression:** Enabled for text/JS/CSS/JSON
- **Static File Caching:** 30-day cache for static assets, 7-day for models
- **sendfile:** Enabled for efficient file transfer
- **tcp_nopush/tcp_nodelay:** Optimized TCP settings
- **Large file support:** 200MB max body size

#### Application Level
- **Lazy Loading:** Features loaded on-demand
- **Tile Cache Management:** Adaptive cache sizes (150-300 tiles)
- **Memory Profiling:** Mode-specific optimizations (2D/3D/Columbus)
- **Chunked Loading:** Progressive tile loading (currently disabled)
- **Error Suppression:** Source map 404s suppressed in console

#### Browser Level
- **Preload/Prefetch:** DNS prefetch for cesium.com
- **Viewport Optimization:** Mobile viewport meta tags
- **Module Imports:** ES6 modules for tree-shaking

---

## 10. Related URLs & Routes

### 10.1 DeepGIS Family URLs

| URL | Description | View Function |
|-----|-------------|---------------|
| `/label/` | Main labeling interface | `views.label` |
| `/label/3d/` | Basic 3D viewer | `views.label_3d` |
| `/label/3d/dev/` | Development 3D viewer | `views.label_3d_dev` |
| `/label/3d/sigma/` | SIGMA 3D viewer | `views.label_3d_sigma` |
| `/label/3d/topology/` | **Main topology viewer (SIGMA)** | `views.label_topology_sigma` |
| `/label/3d/topology/legacy/` | **Legacy topology viewer** ⬅️ | `views.label_topology` |
| `/label/3d/topology/sigma/` | Explicit SIGMA topology | `views.label_topology_sigma` |
| `/label/3d/search/` | DeepGIS search | `views.label_search` |
| `/label/3d/moon/` | Moon viewer | `views.label_moon_viewer` |
| `/stl-viewer/` | STL model viewer | `views.stl_viewer` |
| `/map-label/` | 2D map labeling | `views.map_label` |

### 10.2 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/webclient/getCategoryInfo` | GET | Fetch labeling categories |
| `/webclient/getNewImage` | GET | Get new image for labeling |
| `/webclient/getAllImages` | GET | List all images |
| `/webclient/saveLabel` | POST | Save label data |
| `/webclient/createCategory` | POST | Create new category |
| `/webclient/getRasterInfo` | GET | Get raster layer metadata |
| `/webclient/getTileserverLayers` | GET | List available tile layers |
| `/webclient/save-labels` | POST | Save map labels |
| `/webclient/export-shapefile` | GET | Export shapefile |
| `/webclient/detect-grid` | POST | Grid detection |
| `/webclient/get-3d-model` | GET | Fetch 3D model |
| `/webclient/list-stl-models` | GET | List STL models |
| `/api/elevation-proxy` | GET | Elevation data proxy |

### 10.3 World Sampler API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/webclient/sampler/initialize` | POST | Initialize sampler |
| `/webclient/sampler/sample` | POST | Sample locations |
| `/webclient/sampler/update` | POST | Update distribution |
| `/webclient/sampler/query` | POST | Query region |
| `/webclient/sampler/statistics` | GET | Get statistics |
| `/webclient/sampler/reset` | POST | Reset sampler |
| `/webclient/sampler/history` | GET | Sample history |
| `/webclient/sampler/scored` | GET | Scored locations |

---

## 11. Legacy vs SIGMA Comparison

### 11.1 What is "Legacy"?

The **legacy** version refers to the **original implementation** of the topology viewer before refactoring to the SIGMA architecture. It maintains backward compatibility while the SIGMA version introduces new features and architectural improvements.

### 11.2 Key Differences

| Aspect | Legacy (`/topology/legacy/`) | SIGMA (`/topology/` or `/topology/sigma/`) |
|--------|------------------------------|-------------------------------------------|
| **Template** | `label_topology.html` | `label_topology_refactored.html` (likely) |
| **Architecture** | Original monolithic JS | Modular, refactored architecture |
| **View Function** | `views.label_topology` | `views.label_topology_sigma` |
| **URL Route Name** | `label_topology_legacy` | `label_topology` or `label_topology_sigma` |
| **Default Route** | No (requires `/legacy/`) | Yes (`/topology/` redirects here) |
| **Purpose** | Backward compatibility | Active development |

### 11.3 Why Keep Legacy?

1. **User Familiarity:** Some users may prefer the original interface
2. **Testing:** Compare old vs new implementations
3. **Gradual Migration:** Allows users to transition at their own pace
4. **Bug Fallback:** If SIGMA has issues, legacy provides a stable alternative
5. **Documentation:** Preserves historical implementation

---

## 12. Integration Points

### 12.1 GPS Telemetry Integration

**API Endpoint:** `/api/telemetry/sessions/`  
**Documentation:** `dreams_laboratory/api/GPS_PATHS_QUICKSTART.md`

The legacy topology viewer integrates with the Dreams Laboratory telemetry API to display GPS session paths on the 3D globe. This is documented in the quickstart guide which explicitly mentions the legacy URL:

> "Navigate to: `https://deepgis.org/label/3d/topology/legacy/`"

### 12.2 Earth Innovation Hub

The DeepGIS-XR platform is referenced in the Earth Innovation Hub homepage:

**File:** `templates/earthinnovationhub/home.html`

```html
<strong><a href="https://deepgis.org/label/3d/topology/" target="_blank">DeepGIS-XR</a></strong>
```

(Note: This links to the main `/topology/` route, not the legacy version)

---

## 13. Development & Deployment

### 13.1 Build Process

**Build Tool:** Vite  
**Config:** `deepgis-xr/vite.config.js`

```bash
# Development
cd deepgis-xr
npm run dev

# Production build
npm run build

# Collect static files
python manage.py collectstatic --noinput
```

### 13.2 Directory Structure

```
/home/jdas/dreams-lab-website-server/
├── deepgis-xr/                          # DeepGIS-XR Django app
│   ├── deepgis_xr/
│   │   ├── settings.py                  # Django settings
│   │   ├── urls.py                      # Root URL config
│   │   └── apps/
│   │       └── web/
│   │           ├── urls.py              # Web URLs (defines /label/3d/topology/legacy/)
│   │           ├── views.py             # View functions
│   │           └── templates/
│   │               └── web/
│   │                   └── label_topology.html  # Legacy template
│   ├── staticfiles/                     # Collected static files
│   │   └── web/
│   │       ├── js/                      # JavaScript modules
│   │       └── styles/                  # CSS files
│   └── manage.py                        # Django management
├── dreams_laboratory/                   # Main Django app
│   ├── settings.py                      # Main settings
│   └── urls.py                          # Main URL config
├── nginx.conf                           # Nginx configuration
├── staticfiles/                         # Global static files
└── media/                               # User uploads
```

### 13.3 Running the Application

```bash
# Terminal 1: Django (DeepGIS-XR)
cd /home/jdas/dreams-lab-website-server/deepgis-xr
source venv/bin/activate
python manage.py runserver 8060

# Terminal 2: Main Django app
cd /home/jdas/dreams-lab-website-server
source venv/bin/activate
python manage.py runserver 8080

# Terminal 3: MBTiles tile server
cd /home/jdas/dreams-lab-website-server
./tileserver-gl-light --config tileserver-config.json --port 8091

# Nginx is already running as a system service
sudo systemctl status nginx
```

---

## 14. Troubleshooting & Known Issues

### 14.1 Common Issues

#### Tiles Not Loading
- **Symptom:** White/blank tiles, console errors
- **Causes:**
  - MBTiles server not running
  - CORS headers misconfigured
  - Tile URL malformed
- **Solution:** Check `mbtiles.deepgis.org` is accessible, verify nginx CORS headers

#### Memory Issues
- **Symptom:** Browser slows down, tab crashes
- **Causes:**
  - Too many high-resolution tiles loaded
  - Large 3D models
  - Multiple overlay layers
- **Solution:** Reduce `TILE_CACHE_SIZE`, close unused layers, use 2D view for memory-constrained devices

#### GPS Telemetry Not Loading
- **Symptom:** "Loading sessions..." never completes
- **Causes:**
  - API endpoint not available
  - CSRF token issues
  - Network timeout
- **Solution:** Check `/api/telemetry/sessions/` returns valid JSON, verify CSRF middleware

### 14.2 Debug Mode

Enable debug console by adding to URL:
```
https://deepgis.org/label/3d/topology/legacy/?debug=true
```

Or in browser console:
```javascript
// Load debug module
const debug = await import('/static/deepgis/web/features/debug-console.js');
debug.default.init(window.DeepGISTopology.viewer);
```

---

## 15. Future Roadmap

### 15.1 Planned Features (Based on Codebase Analysis)

- **World Sampler Integration:** Intelligent location sampling for data collection
- **Enhanced Vector Tile Support:** Improved performance and styling
- **WebXR Improvements:** Better VR/AR integration
- **Chunked Loading Re-enablement:** Progressive tile loading (currently disabled)
- **Moon Viewer Enhancements:** Improved lunar data visualization

### 15.2 Migration Path: Legacy → SIGMA

1. **Phase 1 (Current):** Both legacy and SIGMA versions available
2. **Phase 2:** SIGMA becomes default (`/topology/` route)
3. **Phase 3:** Legacy available as fallback (`/topology/legacy/`)
4. **Phase 4:** Legacy deprecated (redirects to SIGMA with notice)
5. **Phase 5:** Legacy removed (SIGMA only)

**Current Status:** Phase 2 (SIGMA is default, legacy available)

---

## 16. Conclusion

The URL `https://deepgis.org/label/3d/topology/legacy/` represents a sophisticated geospatial visualization platform with the following characteristics:

✅ **Mature Technology Stack:** CesiumJS + Django + PostgreSQL/SQLite  
✅ **Modular Architecture:** ES6 modules with lazy loading  
✅ **Performance Optimized:** Adaptive memory management, tile caching  
✅ **Feature-Rich:** 2D/3D/Columbus views, measurements, WebXR, GPS integration  
✅ **Production-Ready:** HTTPS, SSL, caching, CORS, error handling  
✅ **Well-Documented:** Inline comments, configuration files, README docs  

The "legacy" designation indicates this is the **original, stable implementation** maintained for backward compatibility while the newer SIGMA architecture is actively developed. Users can choose between the familiar legacy interface and the cutting-edge SIGMA version.

---

## 17. References

### 17.1 Internal Documentation
- `deepgis-xr/COMPLETE_DOCUMENTATION.md`
- `dreams_laboratory/api/GPS_PATHS_QUICKSTART.md`
- `dreams_laboratory/api/DEEPGIS_INTEGRATION.md`
- `deepgis-xr/WORLD_SAMPLER_ARCHITECTURE.md`

### 17.2 External Resources
- [CesiumJS Documentation](https://cesium.com/learn/)
- [Django Documentation](https://docs.djangoproject.com/)
- [MBTiles Specification](https://github.com/mapbox/mbtiles-spec)
- [WebXR API](https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API)

### 17.3 Related Projects
- **Dreams Laboratory:** Main research platform at ASU
- **OpenUAV:** UAV management system (`openuav.deepgis.org`)
- **Earth Innovation Hub:** Knowledge sharing platform (`/earthinnovationhub/`)

---

**Document Version:** 1.0  
**Last Updated:** November 27, 2025  
**Maintainer:** DeepGIS Development Team  
**Contact:** https://deepgis.org  

---

*This document was generated through comprehensive codebase analysis and is accurate as of the analysis date. For the most current information, refer to the live codebase.*

