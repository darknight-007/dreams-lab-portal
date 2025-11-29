# DeepGIS-XR Architecture Diagram

## URL: https://deepgis.org/label/3d/topology/legacy/

---

## 1. Request Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                  │
│                                                                       │
│  https://deepgis.org/label/3d/topology/legacy/                      │
│         │                                                             │
│         │ HTTPS Request (Port 443)                                   │
│         ▼                                                             │
└─────────┼─────────────────────────────────────────────────────────────┘
          │
          │
┌─────────▼─────────────────────────────────────────────────────────────┐
│                         NGINX (Port 443)                              │
│  Location: /etc/nginx/nginx.conf                                      │
│                                                                        │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ SSL Termination                                       │            │
│  │ - Let's Encrypt Certificate                           │            │
│  │ - deepgis.org-0001/fullchain.pem                     │            │
│  └──────────────────────────────────────────────────────┘            │
│                           │                                            │
│  ┌────────────────────────▼──────────────────────────────┐            │
│  │ URL Pattern Matching                                  │            │
│  │                                                        │            │
│  │  location /label/ {                                   │            │
│  │    proxy_pass http://localhost:8060;                 │            │
│  │  }                                                     │            │
│  │                                                        │            │
│  │  location /static/deepgis/ {                          │            │
│  │    alias /path/to/staticfiles/;                      │            │
│  │  }                                                     │            │
│  └────────────────────────────────────────────────────────┘            │
│                           │                                            │
└───────────────────────────┼──────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                │ /label/               │ /static/deepgis/
                ▼                       ▼
┌───────────────────────────┐   ┌──────────────────────┐
│   Django App (Port 8060)  │   │   Static Files       │
│   deepgis-xr/             │   │   Nginx Direct Serve │
│                           │   │                      │
│  ┌─────────────────────┐ │   │  - CSS files         │
│  │ URL Router          │ │   │  - JavaScript files  │
│  │ deepgis_xr/urls.py  │ │   │  - Images            │
│  └─────────┬───────────┘ │   │  - 3D models (.glb)  │
│            │             │   └──────────────────────┘
│            ▼             │
│  ┌─────────────────────┐ │
│  │ Web App URLs        │ │
│  │ apps/web/urls.py    │ │
│  │                     │ │
│  │ Pattern Match:      │ │
│  │ 'label/3d/         │ │
│  │  topology/legacy/'  │ │
│  └─────────┬───────────┘ │
│            │             │
│            ▼             │
│  ┌─────────────────────┐ │
│  │ View Function       │ │
│  │ views.py:           │ │
│  │ label_topology()    │ │
│  └─────────┬───────────┘ │
│            │             │
│            ▼             │
│  ┌─────────────────────┐ │
│  │ Template Render     │ │
│  │ label_topology.html │ │
│  └─────────┬───────────┘ │
│            │             │
└────────────┼─────────────┘
             │
             │ HTML Response
             ▼
┌──────────────────────────────────────────────────────────────┐
│                    BROWSER RENDERS                            │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ HTML + CSS Loaded                                       │ │
│  └────────────────────────────┬───────────────────────────┘ │
│                                │                              │
│  ┌─────────────────────────────▼──────────────────────────┐ │
│  │ JavaScript Modules Load (ES6)                          │ │
│  │                                                         │ │
│  │  <script type="module" src="/static/deepgis/         │ │
│  │          web/js/main.js"></script>                     │ │
│  │                                                         │ │
│  │  Imports:                                               │ │
│  │   - config.js                                          │ │
│  │   - state.js                                           │ │
│  │   - core/cesium-init.js                               │ │
│  │   - core/layer-management.js                          │ │
│  │   - features/* (lazy loaded)                          │ │
│  └──────────────────────────┬──────────────────────────────┘ │
│                             │                                 │
│  ┌──────────────────────────▼─────────────────────────────┐ │
│  │ Cesium Viewer Initialization                           │ │
│  │  - Creates 3D globe                                     │ │
│  │  - Loads base imagery                                   │ │
│  │  - Sets up camera controls                             │ │
│  └──────────────────────────┬─────────────────────────────┘ │
│                             │                                 │
│  ┌──────────────────────────▼─────────────────────────────┐ │
│  │ API Calls to Backend                                    │ │
│  │                                                         │ │
│  │  GET /webclient/getTileserverLayers                    │ │
│  │  → Returns available tile layers                       │ │
│  │                                                         │ │
│  │  GET /webclient/getRasterInfo?id={layer}              │ │
│  │  → Returns layer metadata (bounds, zoom)               │ │
│  │                                                         │ │
│  │  GET /api/telemetry/sessions/                         │ │
│  │  → Returns GPS telemetry sessions                      │ │
│  └──────────────────────────┬─────────────────────────────┘ │
│                             │                                 │
│  ┌──────────────────────────▼─────────────────────────────┐ │
│  │ External Tile Requests                                  │ │
│  │                                                         │ │
│  │  https://mbtiles.deepgis.org/{layer}/{z}/{x}/{y}.png │ │
│  │  → MBTiles tile server (Port 8091)                    │ │
│  │                                                         │ │
│  │  https://cesium.com/...                                │ │
│  │  → Cesium Ion terrain and imagery                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Interactive 3D Viewer Ready                             │ │
│  │  ✓ Layers loaded                                        │ │
│  │  ✓ Controls active                                      │ │
│  │  ✓ Camera positioned                                    │ │
│  │  ✓ GPS paths displayed                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        DeepGIS-XR System                           │
└────────────────────────────────────────────────────────────────────┘

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│   Presentation   │  │   Application    │  │     Data Layer       │
│      Layer       │  │      Layer       │  │                      │
└──────────────────┘  └──────────────────┘  └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (Browser)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    HTML Template                            │   │
│  │              label_topology.html                            │   │
│  │                                                             │   │
│  │  - UI Structure (sidebar, viewer container)                │   │
│  │  - Django template tags ({% static %}, {% csrf_token %})   │   │
│  │  - External dependencies (Cesium, Bootstrap, Chart.js)     │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    CSS Modules                              │   │
│  │                                                             │   │
│  │  - main.css        (imports all other CSS)                 │   │
│  │  - base.css        (colors, typography, layout)            │   │
│  │  - components.css  (buttons, controls, panels)             │   │
│  │  - sidebar.css     (sidebar-specific styles)               │   │
│  │  - widgets.css     (navigation widgets, indicators)        │   │
│  │  - responsive.css  (mobile breakpoints)                    │   │
│  │  - loading.css     (loading animations)                    │   │
│  │  - debug.css       (debug console)                         │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                JavaScript Architecture                      │   │
│  │                 (ES6 Modules)                               │   │
│  │                                                             │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │               main.js (Entry Point)                   │ │   │
│  │  │                                                        │ │   │
│  │  │  - Bootstraps application                             │ │   │
│  │  │  - Lazy loading coordinator                           │ │   │
│  │  │  - Event listener setup                               │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  │                         │                                   │   │
│  │           ┌─────────────┴─────────────┬──────────────────┐│   │
│  │           │                           │                  ││   │
│  │  ┌────────▼────────┐  ┌──────────────▼───────┐  ┌──────▼┴──┐│
│  │  │   config.js     │  │   state.js           │  │  utils/   ││
│  │  │                 │  │                      │  │           ││
│  │  │ - SIDEBAR_WIDTH │  │ - viewer (Cesium)    │  │ - coords  ││
│  │  │ - MEMORY limits │  │ - currentLayers      │  │ - camera  ││
│  │  │ - SERVER URLs   │  │ - measurements       │  │ - layers  ││
│  │  │ - CESIUM token  │  │ - histogram_chart    │  │ - errors  ││
│  │  │ - VECTOR config │  │ - webxr state        │  │ - astro   ││
│  │  └─────────────────┘  └──────────────────────┘  └───────────┘│
│  │                                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │                    core/ (Core Modules)                   │ │   │
│  │  │                                                            │ │   │
│  │  │  - cesium-init.js        Initialize Cesium viewer         │ │   │
│  │  │  - layer-management.js   Raster/vector layer loading      │ │   │
│  │  │  - base-map.js           Base map and terrain controls    │ │   │
│  │  │  - ui-helpers.js         UI utility functions             │ │   │
│  │  │  - memory-manager.js     Memory optimization              │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  │                                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │              features/ (Lazy Loaded Features)             │ │   │
│  │  │                                                            │ │   │
│  │  │  - webxr.js              WebXR/VR support                 │ │   │
│  │  │  - models.js             3D model loading (GLTF/GLB)      │ │   │
│  │  │  - measurements.js       Distance/area/height tools       │ │   │
│  │  │  - statistics.js         Statistical analysis & charts    │ │   │
│  │  │  - debug-console.js      Debug console & logging          │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  │                                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │               widgets/ (UI Widgets)                       │ │   │
│  │  │                                                            │ │   │
│  │  │  - navigation.js         Heading dial, attitude indicator │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  │                                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │              gps-telemetry.js                             │ │   │
│  │  │                                                            │ │   │
│  │  │  - GPSTelemetryLoader class                               │ │   │
│  │  │  - Session loading and display                            │ │   │
│  │  │  - Path visualization on globe                            │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                  External Libraries                         │   │
│  │                                                             │   │
│  │  - Cesium.js 1.111      3D geospatial engine               │   │
│  │  - Bootstrap 5.3.0      UI framework                        │   │
│  │  - Font Awesome 6.4.0   Icons                              │   │
│  │  - Chart.js 4.4.1       Statistical charts                 │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        BACKEND (Django)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    URL Routing                              │   │
│  │                                                             │   │
│  │  deepgis_xr/urls.py                                        │   │
│  │    │                                                        │   │
│  │    └─▶ include('deepgis_xr.apps.web.urls')                │   │
│  │                                                             │   │
│  │  deepgis_xr/apps/web/urls.py                              │   │
│  │    │                                                        │   │
│  │    ├─▶ path('label/3d/topology/',                         │   │
│  │    │        views.label_topology_sigma)                    │   │
│  │    │                                                        │   │
│  │    ├─▶ path('label/3d/topology/legacy/',                  │   │
│  │    │        views.label_topology)  ◀── THIS URL            │   │
│  │    │                                                        │   │
│  │    ├─▶ path('label/3d/topology/sigma/',                   │   │
│  │    │        views.label_topology_sigma)                    │   │
│  │    │                                                        │   │
│  │    └─▶ path('webclient/*', ...)  (API endpoints)          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    View Functions                           │   │
│  │              (deepgis_xr/apps/web/views.py)                │   │
│  │                                                             │   │
│  │  def label_topology(request):                              │   │
│  │      context = {                                            │   │
│  │          'viewer_info': {                                   │   │
│  │              'version': 'Hybrid',                           │   │
│  │              'engine': 'Cesium.js',                         │   │
│  │              'features': [...],                             │   │
│  │              'data_sources': [...]                          │   │
│  │          }                                                   │   │
│  │      }                                                       │   │
│  │      return render(request,                                 │   │
│  │                    'web/label_topology.html',               │   │
│  │                    context)                                 │   │
│  │                                                             │   │
│  │  def get_tileserver_layers(request):                       │   │
│  │      # Returns available MBTiles layers                     │   │
│  │                                                             │   │
│  │  def get_raster_info(request):                             │   │
│  │      # Returns layer metadata (bounds, zoom levels)         │   │
│  │                                                             │   │
│  │  # ... other API endpoints ...                             │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    Django Settings                          │   │
│  │              (deepgis_xr/settings.py)                      │   │
│  │                                                             │   │
│  │  STATIC_URL = '/static/deepgis/'                           │   │
│  │  STATIC_ROOT = 'staticfiles/'                              │   │
│  │  MEDIA_URL = '/media/deepgis/'                             │   │
│  │  DATABASES = { 'default': ... }                            │   │
│  │  INSTALLED_APPS = [                                         │   │
│  │      'deepgis_xr.apps.web',                                │   │
│  │      'deepgis_xr.apps.api',                                │   │
│  │      'deepgis_xr.apps.ml',                                 │   │
│  │      ...                                                    │   │
│  │  ]                                                          │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              MBTiles Tile Server (Port 8091)                │   │
│  │         https://mbtiles.deepgis.org                         │   │
│  │                                                             │   │
│  │  - Serves raster tiles from MBTiles files                  │   │
│  │  - URL pattern: /{layer}/{z}/{x}/{y}.png                   │   │
│  │  - CORS enabled for cross-origin requests                  │   │
│  │  - TileJSON metadata endpoints                             │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │               Static Tile Server (Port 8092)                │   │
│  │        https://statictiles.deepgis.org                      │   │
│  │                                                             │   │
│  │  - Alternative tile server for static files                │   │
│  │  - Pre-rendered tile pyramids                              │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                  Cesium Ion Services                        │   │
│  │              https://cesium.com                             │   │
│  │                                                             │   │
│  │  - World Terrain (high-resolution elevation)               │   │
│  │  - Satellite imagery (Ion Satellite, Ion Streets)          │   │
│  │  - 3D tilesets                                             │   │
│  │  - Authentication via Ion token                            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              Dreams Laboratory API                          │   │
│  │         /api/telemetry/                                     │   │
│  │                                                             │   │
│  │  - GET /sessions/                                          │   │
│  │    Returns: [{ id, date, metadata }]                       │   │
│  │                                                             │   │
│  │  - GET /sessions/{id}/paths/                               │   │
│  │    Returns: { coordinates: [[lon,lat,alt], ...] }         │   │
│  │                                                             │   │
│  │  - POST /sessions/                                         │   │
│  │    Creates new GPS telemetry session                       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    3D Model Storage                         │   │
│  │      /static/deepgis/models/gltf/                          │   │
│  │                                                             │   │
│  │  - navagunjara-reborn-digital-twin-propane-and-solar-v4.glb│   │
│  │  - Other GLTF/GLB models                                   │   │
│  │  - Served with CORS headers                                │   │
│  │  - Range request support for large files                   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    Database (SQLite)                        │   │
│  │                  db.sqlite3                                 │   │
│  │                                                             │   │
│  │  - User authentication                                      │   │
│  │  - Labeling categories                                      │   │
│  │  - Saved annotations                                        │   │
│  │  - Session data                                             │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Sequence

```
User Action: Load page
│
├─▶ 1. HTTPS Request
│      https://deepgis.org/label/3d/topology/legacy/
│
├─▶ 2. Nginx SSL Termination
│      └─▶ Proxy to http://localhost:8060/label/3d/topology/legacy/
│
├─▶ 3. Django URL Routing
│      └─▶ Match: path('label/3d/topology/legacy/', views.label_topology)
│
├─▶ 4. View Function Execution
│      └─▶ views.label_topology(request)
│           └─▶ Prepare context with viewer_info
│
├─▶ 5. Template Rendering
│      └─▶ render('web/label_topology.html', context)
│           └─▶ Returns HTML response
│
├─▶ 6. Browser Receives HTML
│      └─▶ Parse HTML
│           ├─▶ Load CSS (main.css, Bootstrap)
│           ├─▶ Load external JS (Cesium, Chart.js)
│           └─▶ Load ES6 module (main.js)
│
├─▶ 7. JavaScript Initialization
│      └─▶ main.js executes
│           ├─▶ Import CONFIG from config.js
│           ├─▶ Import AppState from state.js
│           ├─▶ Import initializeCesium from core/cesium-init.js
│           └─▶ Set Cesium.Ion.defaultAccessToken
│
├─▶ 8. Cesium Viewer Creation
│      └─▶ initializeCesium()
│           ├─▶ Create Cesium.Viewer('#cesiumContainer')
│           ├─▶ Set scene mode (2D/3D/Columbus)
│           ├─▶ Configure camera position
│           └─▶ Add base imagery layer
│
├─▶ 9. Layer Discovery
│      └─▶ AJAX: GET /webclient/getTileserverLayers
│           └─▶ Django returns: { layers: [...] }
│                └─▶ Populate layer selection UI
│
├─▶ 10. GPS Telemetry Init
│       └─▶ gps-telemetry.js loads
│            └─▶ new GPSTelemetryLoader(viewer)
│                 └─▶ AJAX: GET /api/telemetry/sessions/
│                      └─▶ Populate session dropdown
│
└─▶ 11. Application Ready
       └─▶ User can interact with viewer


User Action: Load a raster layer
│
├─▶ 1. User clicks checkbox for layer "bf_aug_2020"
│
├─▶ 2. Event Handler: toggleOverlayLayer('bf_aug_2020')
│
├─▶ 3. Fetch Layer Metadata
│      └─▶ AJAX: GET /webclient/getRasterInfo?id=bf_aug_2020
│           └─▶ Returns: {
│                 bounds: [lon_min, lat_min, lon_max, lat_max],
│                 minzoom: 0,
│                 maxzoom: 18,
│                 center: [lon, lat, zoom]
│               }
│
├─▶ 4. Create ImageryLayer
│      └─▶ new Cesium.UrlTemplateImageryProvider({
│           url: 'https://mbtiles.deepgis.org/bf_aug_2020/{z}/{x}/{y}.png',
│           rectangle: Cesium.Rectangle.fromDegrees(...bounds),
│           minimumLevel: 0,
│           maximumLevel: 18
│         })
│
├─▶ 5. Add to Viewer
│      └─▶ viewer.imageryLayers.add(imageryLayer)
│
├─▶ 6. Cesium Requests Tiles
│      └─▶ For each visible tile at current zoom:
│           GET https://mbtiles.deepgis.org/bf_aug_2020/{z}/{x}/{y}.png
│
├─▶ 7. MBTiles Server Response
│      └─▶ Nginx proxies to tileserver-gl (Port 8091)
│           └─▶ Tileserver reads from bf_aug_2020.mbtiles (SQLite)
│                └─▶ Returns PNG tile with CORS headers
│
└─▶ 8. Tiles Rendered
       └─▶ Cesium composites tiles onto globe
            └─▶ User sees imagery overlay


User Action: Load GPS telemetry path
│
├─▶ 1. User selects session from dropdown
│
├─▶ 2. User clicks "Load Path" button
│
├─▶ 3. Event Handler: loadGPSPath(sessionId)
│
├─▶ 4. Fetch Path Data
│      └─▶ AJAX: GET /api/telemetry/sessions/{sessionId}/paths/
│           └─▶ Returns: {
│                 session_id: 123,
│                 coordinates: [
│                   [lon1, lat1, alt1],
│                   [lon2, lat2, alt2],
│                   ...
│                 ],
│                 metadata: { ... }
│               }
│
├─▶ 5. Convert to Cesium Positions
│      └─▶ coordinates.map(coord => 
│           Cesium.Cartesian3.fromDegrees(coord[0], coord[1], coord[2])
│         )
│
├─▶ 6. Create Polyline Entity
│      └─▶ viewer.entities.add({
│           polyline: {
│             positions: cesiumPositions,
│             width: 3,
│             material: Cesium.Color.RED,
│             clampToGround: true
│           }
│         })
│
├─▶ 7. Fly to Path
│      └─▶ viewer.flyTo(entity, {
│           duration: 2.0,
│           offset: new Cesium.HeadingPitchRange(0, -Math.PI / 4, distance)
│         })
│
└─▶ 8. Path Displayed
       └─▶ User sees GPS track on globe
```

---

## 4. Memory Management Strategy

```
┌──────────────────────────────────────────────────────────────┐
│               Memory Management Architecture                  │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Scene Mode Detection                       │
│                                                              │
│  viewer.scene.mode === Cesium.SceneMode.SCENE2D?           │
│                           │                                  │
│        ┌──────────────────┴──────────────────┬──────────┐  │
│        │                  │                   │          │  │
│     SCENE2D         SCENE3D          COLUMBUS_VIEW      │  │
│        │                  │                   │          │  │
└────────┼──────────────────┼───────────────────┼──────────┘  │
         │                  │                   │              │
         ▼                  ▼                   ▼              │
┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│   2D Settings  │  │  3D Settings   │  │  Columbus Settings│ │
│                │  │                │  │                  │ │
│ Cache: 150     │  │ Cache: 300     │  │ Cache: 225       │ │
│ Error: 4       │  │ Error: 6       │  │ Error: 5         │ │
│ Preload: OFF   │  │ Preload: ON    │  │ Preload: ON      │ │
└────────────────┘  └────────────────┘  └──────────────────┘ │
         │                  │                   │              │
         └──────────────────┴───────────────────┘              │
                            │                                  │
                            ▼                                  │
            ┌───────────────────────────────┐                 │
            │   Apply to Cesium Viewer      │                 │
            │                               │                 │
            │  viewer.scene.globe           │                 │
            │    .tileCacheSize = X         │                 │
            │                               │                 │
            │  viewer.scene.globe           │                 │
            │    .maximumScreenSpaceError   │                 │
            │                               │                 │
            │  imageryProvider              │                 │
            │    .maximumLevel = capped     │                 │
            └───────────────────────────────┘                 │

┌─────────────────────────────────────────────────────────────┐
│                    Tile Cache Lifecycle                      │
│                                                              │
│  1. Request tile at (z, x, y)                               │
│     │                                                        │
│  2. Check cache                                             │
│     ├─▶ Hit: Return cached tile                            │
│     └─▶ Miss: Fetch from server                            │
│          │                                                   │
│  3. Fetch tile from mbtiles.deepgis.org                    │
│     │                                                        │
│  4. Add to cache                                            │
│     │                                                        │
│  5. Cache full?                                             │
│     ├─▶ No: Keep tile                                      │
│     └─▶ Yes: Evict LRU tile                                │
│                                                              │
│  Cache Size Limits:                                         │
│  - 2D Mode: 150 tiles (~38 MB @ 256x256 RGBA)             │
│  - 3D Mode: 300 tiles (~75 MB)                             │
│  - Columbus: 225 tiles (~56 MB)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. File System Layout

```
/home/jdas/dreams-lab-website-server/
│
├── nginx.conf                              # Main nginx configuration
│
├── deepgis-xr/                             # DeepGIS-XR Django application
│   │
│   ├── manage.py                           # Django management script
│   ├── db.sqlite3                          # SQLite database
│   ├── vite.config.js                      # Vite build configuration
│   ├── package.json                        # npm dependencies
│   │
│   ├── deepgis_xr/                         # Main Django package
│   │   ├── __init__.py
│   │   ├── settings.py                     # Django settings
│   │   ├── urls.py                         # Root URL configuration
│   │   ├── wsgi.py                         # WSGI entry point
│   │   │
│   │   └── apps/
│   │       ├── web/                        # Web application
│   │       │   ├── urls.py                 # ⬅️ Defines /label/3d/topology/legacy/
│   │       │   ├── views.py                # ⬅️ Contains label_topology() function
│   │       │   │
│   │       │   └── templates/
│   │       │       └── web/
│   │       │           └── label_topology.html  # ⬅️ Legacy template
│   │       │
│   │       ├── api/                        # API endpoints
│   │       ├── auth/                       # Authentication
│   │       ├── core/                       # Core models
│   │       └── ml/                         # Machine learning
│   │
│   ├── staticfiles/                        # Collected static files (production)
│   │   └── web/
│   │       ├── js/
│   │       │   ├── main.js                 # ⬅️ Entry point
│   │       │   ├── config.js               # ⬅️ Configuration
│   │       │   ├── state.js                # ⬅️ Global state
│   │       │   ├── gps-telemetry.js        # ⬅️ GPS integration
│   │       │   │
│   │       │   ├── core/                   # Core modules
│   │       │   │   ├── cesium-init.js
│   │       │   │   ├── layer-management.js
│   │       │   │   ├── base-map.js
│   │       │   │   ├── ui-helpers.js
│   │       │   │   └── memory-manager.js
│   │       │   │
│   │       │   ├── features/               # Lazy-loaded features
│   │       │   │   ├── webxr.js
│   │       │   │   ├── models.js
│   │       │   │   ├── measurements.js
│   │       │   │   ├── statistics.js
│   │       │   │   └── debug-console.js
│   │       │   │
│   │       │   ├── utils/                  # Utilities
│   │       │   │   ├── astronomy.js
│   │       │   │   ├── camera.js
│   │       │   │   ├── coordinates.js
│   │       │   │   ├── errors.js
│   │       │   │   ├── layers.js
│   │       │   │   └── vector-tiles.js
│   │       │   │
│   │       │   └── widgets/                # UI widgets
│   │       │       └── navigation.js
│   │       │
│   │       └── styles/                     # CSS files
│   │           ├── main.css
│   │           ├── base.css
│   │           ├── components.css
│   │           ├── sidebar.css
│   │           ├── widgets.css
│   │           ├── responsive.css
│   │           ├── loading.css
│   │           └── debug.css
│   │
│   └── staticfiles/models/                 # 3D models
│       └── gltf/
│           └── navagunjara-reborn-digital-twin-propane-and-solar-v4.glb
│
├── dreams_laboratory/                      # Main Dreams Lab application
│   ├── settings.py                         # Main Django settings
│   ├── urls.py                             # Main URL configuration
│   └── api/                                # Telemetry API
│       ├── urls.py                         # API routes
│       └── views.py                        # API views (GPS telemetry)
│
├── staticfiles/                            # Global static files
│   ├── deepgis/                            # DeepGIS static files (collected)
│   └── dreams/                             # Dreams Lab static files
│
└── media/                                  # User-uploaded media
    └── render_outputs/
```

---

## 6. Port Mapping

```
┌─────────────────────────────────────────────────────────────┐
│                        Port Allocation                       │
└─────────────────────────────────────────────────────────────┘

External (Public Internet)
    │
    └─▶ Port 443 (HTTPS)
    └─▶ Port 80 (HTTP → redirect to 443)
         │
         └─▶ Nginx (System Service)
              │
              ├─▶ deepgis.org
              │    │
              │    ├─▶ /label/*       → localhost:8060 (DeepGIS-XR Django)
              │    ├─▶ /webclient/*   → localhost:8060 (DeepGIS-XR API)
              │    ├─▶ /*             → localhost:8080 (Main Django)
              │    └─▶ /static/*      → Direct file serve
              │
              ├─▶ mbtiles.deepgis.org  → localhost:8091 (Tileserver GL)
              ├─▶ statictiles.deepgis.org → localhost:8092 (Static Tiles)
              ├─▶ rocks.deepgis.org    → /mnt/.../rock-tiles/raw/
              └─▶ openuav.deepgis.org  → localhost:6080 (OpenUAV noVNC)

Internal (Localhost)
    │
    ├─▶ 8060: DeepGIS-XR Django App
    │         python manage.py runserver 8060
    │         Location: /home/jdas/dreams-lab-website-server/deepgis-xr/
    │
    ├─▶ 8080: Main Django App (Dreams Laboratory)
    │         python manage.py runserver 8080
    │         Location: /home/jdas/dreams-lab-website-server/
    │
    ├─▶ 8091: MBTiles Tileserver (tileserver-gl)
    │         serves: *.mbtiles files
    │         URL: https://mbtiles.deepgis.org
    │
    ├─▶ 8092: Static Tiles Server
    │         URL: https://statictiles.deepgis.org
    │
    └─▶ 6080: OpenUAV noVNC Server
              URL: https://openuav.deepgis.org
```

---

## 7. Security Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Security Layers                          │
└──────────────────────────────────────────────────────────────┘

Layer 1: Transport Security
    │
    ├─▶ HTTPS Enforcement
    │    └─▶ HTTP (Port 80) → 301 Redirect → HTTPS (Port 443)
    │
    ├─▶ SSL/TLS Certificates
    │    └─▶ Let's Encrypt (Auto-renewal)
    │         └─▶ deepgis.org-0001/fullchain.pem
    │         └─▶ deepgis.org-0001/privkey.pem
    │
    └─▶ SSL Configuration
         └─▶ options-ssl-nginx.conf
         └─▶ dhparams.pem (Diffie-Hellman parameters)

Layer 2: Application Security
    │
    ├─▶ Django Security Middleware
    │    ├─▶ SecurityMiddleware
    │    ├─▶ CsrfViewMiddleware (CSRF tokens)
    │    ├─▶ AuthenticationMiddleware
    │    └─▶ ClickjackingMiddleware (X-Frame-Options)
    │
    ├─▶ CORS Headers
    │    └─▶ corsheaders.middleware.CorsMiddleware
    │         └─▶ Allows cross-origin tile requests
    │
    └─▶ Proxy Security Headers
         ├─▶ X-Real-IP
         ├─▶ X-Forwarded-For
         └─▶ X-Forwarded-Proto

Layer 3: Access Control
    │
    ├─▶ Authentication
    │    └─▶ Django Auth System
    │         └─▶ User login required for certain endpoints
    │
    ├─▶ Rate Limiting
    │    └─▶ Nginx rate limiting (not explicitly configured)
    │
    └─▶ Static File Permissions
         └─▶ Read-only access to /static/ and /media/

Layer 4: Data Security
    │
    ├─▶ Database
    │    └─▶ SQLite file permissions (owner-only read/write)
    │
    ├─▶ CSRF Tokens
    │    └─▶ {% csrf_token %} in templates
    │         └─▶ Required for POST requests
    │
    └─▶ Input Validation
         └─▶ Django form validation
         └─▶ SQL injection prevention (ORM)
         └─▶ XSS prevention (template auto-escaping)
```

---

## 8. Monitoring & Debugging

```
┌──────────────────────────────────────────────────────────────┐
│                   Monitoring Architecture                     │
└──────────────────────────────────────────────────────────────┘

Frontend Monitoring
    │
    ├─▶ Performance Monitoring
    │    ├─▶ FPS Counter (real-time)
    │    │    └─▶ Displayed in top-right corner
    │    │
    │    ├─▶ Memory Usage
    │    │    └─▶ Tile cache size tracking
    │    │    └─▶ Entity count
    │    │
    │    └─▶ Network Requests
    │         └─▶ Tile load times
    │         └─▶ Failed requests
    │
    ├─▶ Error Tracking
    │    ├─▶ Console Logging
    │    │    └─▶ Categorized by severity
    │    │    └─▶ Source map errors suppressed
    │    │
    │    ├─▶ Status Indicators
    │    │    └─▶ "Initializing Cesium..."
    │    │    └─▶ "Loading tiles..."
    │    │    └─▶ Error messages
    │    │
    │    └─▶ Snackbar Notifications
    │         └─▶ Success/error messages
    │         └─▶ Auto-dismiss after 3s
    │
    └─▶ Debug Console
         └─▶ Lazy-loaded module
         └─▶ Press '~' to toggle
         └─▶ Shows:
              ├─▶ Camera position
              ├─▶ Loaded layers
              ├─▶ Entity count
              └─▶ Performance stats

Backend Monitoring
    │
    ├─▶ Django Logging
    │    └─▶ django_debug.log
    │         └─▶ Request/response logs
    │         └─▶ Error stack traces
    │
    ├─▶ Nginx Access Logs
    │    └─▶ /var/log/nginx/access.log
    │         └─▶ All HTTP requests
    │
    ├─▶ Nginx Error Logs
    │    └─▶ /var/log/nginx/error.log
    │         └─▶ Proxy errors
    │         └─▶ SSL errors
    │
    └─▶ System Monitoring
         ├─▶ systemctl status nginx
         ├─▶ ps aux | grep django
         └─▶ netstat -tulpn (port usage)

Browser DevTools
    │
    ├─▶ Network Tab
    │    └─▶ Tile requests
    │    └─▶ API calls
    │    └─▶ Response times
    │
    ├─▶ Console Tab
    │    └─▶ JavaScript logs
    │    └─▶ Cesium warnings
    │    └─▶ Application logs
    │
    ├─▶ Performance Tab
    │    └─▶ Frame rate
    │    └─▶ Memory heap
    │    └─▶ CPU usage
    │
    └─▶ Application Tab
         └─▶ Local storage
         └─▶ Session storage
         └─▶ Cookies
```

---

## 9. Deployment Checklist

```
┌──────────────────────────────────────────────────────────────┐
│                     Deployment Steps                          │
└──────────────────────────────────────────────────────────────┘

Pre-Deployment
  ☐ Update code from git
  ☐ Review changes in urls.py and views.py
  ☐ Test locally (runserver)
  ☐ Check for linter errors

Build Static Files
  ☐ cd deepgis-xr/
  ☐ npm run build (Vite production build)
  ☐ python manage.py collectstatic --noinput
  ☐ Verify staticfiles/ directory updated

Database Migrations
  ☐ python manage.py makemigrations
  ☐ python manage.py migrate
  ☐ Test database integrity

Service Restart
  ☐ sudo systemctl restart nginx
  ☐ Kill Django processes:
      pkill -f "manage.py runserver"
  ☐ Restart Django apps:
      cd /home/jdas/dreams-lab-website-server/deepgis-xr
      python manage.py runserver 8060 &
      
      cd /home/jdas/dreams-lab-website-server
      python manage.py runserver 8080 &
  ☐ Check tileserver running (Port 8091)

Verification
  ☐ Visit https://deepgis.org/label/3d/topology/legacy/
  ☐ Check console for errors
  ☐ Test layer loading
  ☐ Test GPS telemetry
  ☐ Test measurements
  ☐ Test 3D model loading
  ☐ Test WebXR/VR
  ☐ Check mobile responsiveness
  ☐ Verify HTTPS (no mixed content)

Monitoring
  ☐ tail -f /var/log/nginx/access.log
  ☐ tail -f /var/log/nginx/error.log
  ☐ tail -f deepgis-xr/django_debug.log
  ☐ Check FPS counter
  ☐ Monitor memory usage

Rollback Plan
  ☐ Keep backup of staticfiles/
  ☐ Keep backup of db.sqlite3
  ☐ Git tag release version
  ☐ Document known issues
```

---

**End of Architecture Diagram**

