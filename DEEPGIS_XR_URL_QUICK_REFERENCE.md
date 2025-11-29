# DeepGIS-XR URL Quick Reference

## URL: https://deepgis.org/label/3d/topology/legacy/

---

## TL;DR (Executive Summary)

**What is it?**  
The **Legacy Topology Viewer** - a Cesium-based 3D geospatial visualization platform for hybrid 2D/3D mapping, measurement tools, and GPS telemetry visualization.

**Status:** ✅ Production - Stable  
**Version:** Legacy (original implementation)  
**Alternative:** `/label/3d/topology/` (SIGMA refactored version)

---

## Quick Facts

| Aspect | Details |
|--------|---------|
| **Protocol** | HTTPS (Port 443) |
| **Web Server** | Nginx → Reverse Proxy |
| **Backend** | Django (Python) on Port 8060 |
| **Frontend** | CesiumJS 1.111 + ES6 Modules |
| **Template** | `label_topology.html` |
| **View Function** | `views.label_topology()` |
| **URL Route Name** | `label_topology_legacy` |

---

## URL Routing Path

```
User Browser
    ↓
https://deepgis.org/label/3d/topology/legacy/
    ↓
Nginx (Port 443) [SSL Termination]
    ↓
Proxy to: http://localhost:8060/label/3d/topology/legacy/
    ↓
Django URL Router: deepgis_xr/apps/web/urls.py
    ↓
Match: path('label/3d/topology/legacy/', views.label_topology)
    ↓
View: deepgis_xr/apps/web/views.py → label_topology(request)
    ↓
Template: deepgis_xr/apps/web/templates/web/label_topology.html
    ↓
JavaScript: staticfiles/web/js/main.js (ES6 Module)
    ↓
3D Viewer Ready
```

---

## File Locations

```
/home/jdas/dreams-lab-website-server/
├── nginx.conf                                    # Nginx config (Port 443 → 8060)
├── deepgis-xr/
│   ├── deepgis_xr/
│   │   ├── urls.py                              # Root URL config
│   │   └── apps/web/
│   │       ├── urls.py                          # Defines /label/3d/topology/legacy/
│   │       ├── views.py                         # View: label_topology()
│   │       └── templates/web/
│   │           └── label_topology.html          # Template
│   └── staticfiles/web/
│       ├── js/
│       │   ├── main.js                          # Entry point
│       │   ├── config.js                        # Configuration
│       │   ├── gps-telemetry.js                 # GPS integration
│       │   └── core/, features/, utils/         # Modules
│       └── styles/
│           └── main.css                         # Styles
```

---

## Key Technologies

| Layer | Technology | Version |
|-------|-----------|---------|
| **3D Engine** | CesiumJS | 1.111 |
| **Backend** | Django | 4.x |
| **Web Server** | Nginx | Latest |
| **UI Framework** | Bootstrap | 5.3.0 |
| **Charts** | Chart.js | 4.4.1 |
| **Icons** | Font Awesome | 6.4.0 |
| **Module System** | ES6 Modules | Native |
| **Build Tool** | Vite | Latest |

---

## Main Features

✅ **Multiple View Modes:** 2D Map, 3D Globe, Columbus View  
✅ **Layer Management:** Raster overlays, Vector layers, 3D terrain  
✅ **Measurement Tools:** Distance, Area, Height  
✅ **GPS Telemetry:** Load and visualize GPS session paths  
✅ **3D Models:** Load GLTF/GLB models (e.g., Navagunjara Digital Twin)  
✅ **WebXR/VR:** Virtual reality support  
✅ **Performance Monitoring:** FPS counter, memory management  
✅ **Real-time Stats:** Camera pose, sun/moon position  
✅ **Navigation Widgets:** Heading dial, attitude indicator  
✅ **Mobile Responsive:** Works on tablets and phones  

---

## Configuration Highlights

**File:** `staticfiles/web/js/config.js`

```javascript
CONFIG = {
  SERVERS: {
    MBTILES_SERVER: 'https://mbtiles.deepgis.org',   // Tile server
    TOPOLOGY_SERVER: 'https://localhost:8092'        // Alternative tiles
  },
  MEMORY: {
    TILE_CACHE_SIZE: 300,                            // 3D mode
    MAX_SCREEN_SPACE_ERROR: 6,                       // Tile detail
    MODE_SPECIFIC: {
      SCENE2D: { TILE_CACHE_SIZE: 150 },             // 2D optimization
      SCENE3D: { TILE_CACHE_SIZE: 300 },             // 3D optimization
      COLUMBUS_VIEW: { TILE_CACHE_SIZE: 225 }        // Hybrid
    }
  },
  VECTOR_TILES: {
    ENABLED: true,
    MAX_ZOOM: 14,
    MAX_CACHED_TILES: 100
  }
}
```

---

## API Endpoints Used by Viewer

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/webclient/getTileserverLayers` | GET | List available tile layers |
| `/webclient/getRasterInfo?id={layer}` | GET | Get layer metadata (bounds, zoom) |
| `/api/telemetry/sessions/` | GET | List GPS telemetry sessions |
| `/api/telemetry/sessions/{id}/paths/` | GET | Get GPS path coordinates |
| `https://mbtiles.deepgis.org/{layer}/{z}/{x}/{y}.png` | GET | Raster tiles |

---

## Related URLs

| URL | Description | Status |
|-----|-------------|--------|
| `/label/3d/topology/` | **Main topology viewer (SIGMA)** | ✅ Default |
| `/label/3d/topology/legacy/` | **Legacy topology viewer** | ✅ Active |
| `/label/3d/topology/sigma/` | Explicit SIGMA route | ✅ Active |
| `/label/3d/` | Basic 3D viewer | ✅ Active |
| `/label/3d/dev/` | Development 3D viewer | ✅ Active |
| `/label/3d/sigma/` | SIGMA 3D viewer | ✅ Active |
| `/label/3d/search/` | DeepGIS search viewer | ✅ Active |
| `/label/3d/moon/` | Moon viewer | ✅ Active |

---

## Port Mapping

| Port | Service | URL |
|------|---------|-----|
| **443** | Nginx (HTTPS) | https://deepgis.org |
| **8060** | DeepGIS-XR Django | Internal (proxied) |
| **8080** | Main Django (Dreams Lab) | Internal (proxied) |
| **8091** | MBTiles Tileserver | https://mbtiles.deepgis.org |
| **8092** | Static Tiles Server | https://statictiles.deepgis.org |
| **6080** | OpenUAV noVNC | https://openuav.deepgis.org |

---

## How to Run Locally

```bash
# Terminal 1: DeepGIS-XR Django
cd /home/jdas/dreams-lab-website-server/deepgis-xr
source venv/bin/activate
python manage.py runserver 8060

# Terminal 2: Main Django
cd /home/jdas/dreams-lab-website-server
source venv/bin/activate
python manage.py runserver 8080

# Terminal 3: MBTiles Server
cd /home/jdas/dreams-lab-website-server
./tileserver-gl-light --config tileserver-config.json --port 8091

# Nginx should already be running
sudo systemctl status nginx

# Access at:
https://deepgis.org/label/3d/topology/legacy/
```

---

## Troubleshooting

### Tiles Not Loading
- Check `https://mbtiles.deepgis.org` is accessible
- Verify nginx CORS headers
- Check tileserver running on port 8091

### GPS Telemetry Not Loading
- Verify `/api/telemetry/sessions/` returns JSON
- Check CSRF token in request
- Check Django telemetry app is running (port 8080)

### Memory Issues (Browser Slows Down)
- Reduce `TILE_CACHE_SIZE` in config.js
- Close unused overlay layers
- Switch to 2D view
- Reduce browser zoom level

### Console Errors
- **Source map 404s:** Suppressed (normal, ignored)
- **CORS errors:** Check nginx config for Access-Control-Allow-Origin
- **Cesium Ion warnings:** Verify Ion token is valid

---

## Debug Mode

**Enable in URL:**
```
https://deepgis.org/label/3d/topology/legacy/?debug=true
```

**Or in Browser Console:**
```javascript
// Load debug module
const debug = await import('/static/deepgis/web/features/debug-console.js');
debug.default.init(window.DeepGISTopology.viewer);

// Toggle with '~' key
```

---

## Legacy vs SIGMA

| Aspect | Legacy | SIGMA |
|--------|--------|-------|
| **URL** | `/topology/legacy/` | `/topology/` or `/topology/sigma/` |
| **Template** | `label_topology.html` | `label_topology_refactored.html` (likely) |
| **Architecture** | Original monolithic | Refactored modular |
| **Default** | No | Yes |
| **Stability** | Very stable | Active development |
| **Purpose** | Backward compatibility | New features |

---

## Performance Benchmarks

| Metric | 2D Mode | 3D Mode | Columbus View |
|--------|---------|---------|---------------|
| **Tile Cache** | 150 tiles | 300 tiles | 225 tiles |
| **Memory** | ~38 MB | ~75 MB | ~56 MB |
| **Screen Space Error** | 4 (sharp) | 6 (balanced) | 5 (hybrid) |
| **FPS Target** | 60 FPS | 30-60 FPS | 45-60 FPS |

---

## Security

✅ **HTTPS Enforced:** HTTP → HTTPS redirect  
✅ **SSL Certificates:** Let's Encrypt (auto-renewed)  
✅ **CSRF Protection:** Django middleware enabled  
✅ **CORS Headers:** Configured for tile requests  
✅ **Proxy Security:** X-Real-IP, X-Forwarded-For headers  
✅ **Static File Security:** Read-only access, 30-day cache  
✅ **Input Validation:** Django ORM prevents SQL injection  
✅ **XSS Prevention:** Template auto-escaping  

---

## JavaScript Module Structure

```
main.js (Entry Point)
  │
  ├─▶ config.js                 Configuration
  ├─▶ state.js                  Global state
  │
  ├─▶ core/
  │    ├─▶ cesium-init.js       Viewer initialization
  │    ├─▶ layer-management.js  Layer loading
  │    ├─▶ base-map.js          Base map controls
  │    ├─▶ ui-helpers.js        UI utilities
  │    └─▶ memory-manager.js    Memory optimization
  │
  ├─▶ features/ (Lazy Loaded)
  │    ├─▶ webxr.js             WebXR/VR support
  │    ├─▶ models.js            3D model loading
  │    ├─▶ measurements.js      Measurement tools
  │    ├─▶ statistics.js        Statistical charts
  │    └─▶ debug-console.js     Debug console
  │
  ├─▶ utils/
  │    ├─▶ astronomy.js         Sun/moon calculations
  │    ├─▶ camera.js            Camera utilities
  │    ├─▶ coordinates.js       Coordinate transforms
  │    ├─▶ errors.js            Error handling
  │    ├─▶ layers.js            Layer utilities
  │    └─▶ vector-tiles.js      Vector tile processing
  │
  ├─▶ widgets/
  │    └─▶ navigation.js        Navigation widgets
  │
  └─▶ gps-telemetry.js          GPS path loader
```

---

## Useful Commands

### Check Services
```bash
sudo systemctl status nginx
ps aux | grep "manage.py runserver"
netstat -tulpn | grep -E "443|8060|8080|8091"
```

### View Logs
```bash
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
tail -f /home/jdas/dreams-lab-website-server/deepgis-xr/django_debug.log
```

### Restart Services
```bash
sudo systemctl restart nginx
pkill -f "manage.py runserver"
# Then re-run Django apps
```

### Build Static Files
```bash
cd /home/jdas/dreams-lab-website-server/deepgis-xr
npm run build
python manage.py collectstatic --noinput
```

---

## External Resources

- **CesiumJS Docs:** https://cesium.com/learn/
- **Django Docs:** https://docs.djangoproject.com/
- **MBTiles Spec:** https://github.com/mapbox/mbtiles-spec
- **WebXR API:** https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API

---

## Contact & Support

- **Documentation:** See `DEEPGIS_XR_URL_PATTERN_ANALYSIS.md` for full details
- **Architecture Diagram:** See `DEEPGIS_XR_ARCHITECTURE_DIAGRAM.md`
- **GPS Integration:** See `dreams_laboratory/api/GPS_PATHS_QUICKSTART.md`
- **Website:** https://deepgis.org

---

**Last Updated:** November 27, 2025  
**Version:** 1.0  
**Status:** ✅ Production Ready

