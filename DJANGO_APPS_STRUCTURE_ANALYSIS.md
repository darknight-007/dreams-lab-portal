# Django Apps Structure Analysis

## Overview

This document provides a comprehensive analysis of the two main Django applications in the Dreams Lab Portal:
1. **dreams_laboratory** - Main Django project at the root level
2. **deepgis-xr** - Separate Django project in the `deepgis-xr/` subdirectory

Both apps are part of the same workspace but operate as separate Django projects with their own databases, settings, and URL configurations.

---

## 1. dreams_laboratory App

### 1.1 Project Structure

```
dreams_laboratory/
├── __init__.py
├── apps.py                    # DreamsLaboratoryConfig
├── settings.py                # Main Django settings
├── urls.py                    # Root URL configuration
├── wsgi.py / asgi.py          # WSGI/ASGI application
├── models.py                  # All data models
├── views.py                   # Main view functions
├── quiz_views.py              # Quiz-specific views
├── tutorials.py               # Tutorial management
├── authentication.py          # Custom auth backend (phone number)
├── db_routers.py              # Database routing for multi-DB setup
│
├── api/                       # REST API sub-app
│   ├── urls.py
│   ├── views.py
│   ├── serializers.py
│   └── README.md
│
├── management/                # Custom management commands
│   └── commands/
│
├── migrations/                # Database migrations
│
├── utils/                     # Utility modules
│   └── twilio_verify.py
│
├── templatetags/              # Custom template tags/filters
│   └── custom_filters.py
│
└── static/                    # Static files
    └── dreams_laboratory/
```

### 1.2 Key Features

#### Models (`models.py`)
- **User Management:**
  - `CustomUser` - Extends AbstractUser with phone number verification
  - `People`, `Role` - Lab member management
  - `Project`, `Research`, `Publication`, `FundingSource` - Research tracking
  - `Asset`, `Photo` - Asset management

- **Quiz System:**
  - `QuizSubmission` - Stores quiz answers and scores
  - `QuizProgress` - Tracks user progress through quiz components

- **Telemetry (Pixhawk Integration):**
  - `DroneTelemetrySession` - Flight session tracking
  - `LocalPositionOdom` - Local NED position/velocity data
  - `GPSFixRaw` - Raw GPS data from receiver
  - `GPSFixEstimated` - Filtered GPS estimates from state estimator

- **World Sampler (Geospatial Sampling):**
  - `SampledLocation` - Sampled locations with scores
  - `SamplingSession` - Session tracking
  - `DistributionUpdate` - Distribution update logs
  - **Note:** These models are routed to `deepgis_xr` database via `db_routers.py`

#### Views (`views.py`)
- **Tutorial Views:** Multiple "buddy" tutorial interfaces:
  - `stereo_buddy_view`, `slam_buddy_view`, `bundle_adjustment_buddy_view`
  - `param_estimation_buddy_view`, `ransac_buddy_view`
  - `gaussian_processes_buddy_view`, `sampling_buddy_view`
  - `gp_ucb_buddy_view`, `path_planning_buddy_view`
  - `cart_pole_lqr_buddy_view`, `particle_filter_buddy_view`
  - `point_cloud_buddy`, `sensor_fusion_buddy`
  - `image_buddy_view`, `drone_buddy_view`
  - `multi_armed_bandit_buddy_view`

- **Course/Quiz Views:**
  - `ses598_course_view`, `ses598_quiz`, `ses598_quiz_part2`
  - `generate_certificate`, `reset_quiz`, `quiz_admin_view`
  - `ses598_2025_retrospective`

- **Semi-Supervised Labeling:**
  - `semi_supervised_label_view`
  - `generate_assisted_labels`, `save_assisted_labels`, `get_label_images`

- **Earth Innovation Hub:**
  - `earthinnovationhub_home`, `earthinnovationhub_journal`
  - `earthinnovationhub_article_mcp`, `earthinnovationhub_article_welcome`
  - `earthinnovationhub_navagunjara`

#### URL Patterns (`urls.py`)
- Root paths: `/`, `/dreamslab/`
- Tutorial paths: `/tutorials/*`
- Quiz paths: `/dreamslab/ses598/quiz/*`
- API paths: `/api/*` (includes `dreams_laboratory.api.urls`)
- Label paths: `/label/semi-supervised/*`
- Earth Innovation Hub: `/earthinnovationhub/*`

#### API (`api/`)
- **Endpoints:** Defined in `api/urls.py`
- **Views:** Telemetry, GPS paths, and other API endpoints
- **Integration:** Connects to DeepGIS-XR for geospatial features

#### Settings (`settings.py`)
- **Database Configuration:**
  - `default`: SQLite at root (`db.sqlite3`)
  - `deepgis_xr`: SQLite at `deepgis-xr/db.sqlite3`
- **Database Router:** `db_routers.WorldSamplerRouter` routes World Sampler models to `deepgis_xr` DB
- **Custom User Model:** `AUTH_USER_MODEL = 'dreams_laboratory.CustomUser'`
- **Authentication Backends:**
  - `dreams_laboratory.authentication.PhoneNumberBackend`
  - `django.contrib.auth.backends.ModelBackend`
- **Installed Apps:**
  - `dreams_laboratory.apps.DreamsLaboratoryConfig`
  - `openuav_manager`

---

## 2. deepgis-xr App

### 2.1 Project Structure

```
deepgis-xr/
├── manage.py                  # Separate Django management script
├── deepgis_xr/
│   ├── __init__.py
│   ├── settings.py            # DeepGIS-XR Django settings
│   ├── urls.py                # Root URL configuration
│   ├── wsgi.py / celery.py    # WSGI and Celery config
│   │
│   └── apps/
│       ├── auth/              # Authentication app
│       │   ├── apps.py
│       │   ├── models.py      # Custom User with phone verification
│       │   ├── views.py
│       │   ├── urls.py
│       │   ├── forms.py
│       │   └── templates/
│       │
│       ├── core/              # Core functionality
│       │   ├── apps.py
│       │   ├── models.py      # Image, Category, Label models
│       │   ├── admin.py
│       │   ├── middleware.py
│       │   ├── exceptions.py
│       │   ├── image_processing/
│       │   ├── management/commands/
│       │   ├── utils/
│       │   └── tests/
│       │
│       ├── web/               # Web interface app
│       │   ├── models.py      # World Sampler models (duplicate of dreams_lab)
│       │   ├── views.py        # Main web views
│       │   ├── urls.py         # Web URL patterns
│       │   ├── admin.py
│       │   ├── world_sampler_api.py  # World Sampler API endpoints
│       │   ├── world_sampler.py      # World Sampler logic
│       │   ├── templates/web/
│       │   ├── static/web/
│       │   └── management/commands/
│       │
│       ├── api/               # REST API app
│       │   └── v1/
│       │       ├── urls.py
│       │       ├── serializers.py
│       │       └── views/
│       │
│       └── ml/                # Machine Learning app
│           └── services/
│               ├── predictor.py
│               ├── trainer.py
│               └── learning_tools.py
│
├── staticfiles/               # Collected static files
├── static/                   # Source static files
├── templates/                # Global templates
├── media/                    # User-uploaded files
└── data/                     # Data files
```

### 2.2 Key Features

#### Apps Structure

**1. auth App (`deepgis_xr.apps.auth`)**
- **Models:**
  - `User` - Custom user model with phone number (extends AbstractUser)
  - `VerificationCode` - Phone verification codes
- **Features:** Phone number authentication, verification workflow

**2. core App (`deepgis_xr.apps.core`)**
- **Models:**
  - `Color`, `CategoryType` - Labeling categories
  - `Image`, `ImageSourceType` - Image management
  - `ImageLabel`, `CategoryLabel`, `Labeler` - Labeling system
  - `ImageWindow`, `ImageFilter` - Image processing
  - `RasterImage`, `TiledGISLabel` - GIS raster/label data
  - `VehicleType`, `Vehicle`, `VehiclePosition` - Vehicle tracking
  - `VehicleGeofence`, `VehicleAlert` - Vehicle monitoring
- **Features:** Core labeling infrastructure, vehicle tracking

**3. web App (`deepgis_xr.apps.web`)**
- **Models:**
  - `SampledLocation`, `SamplingSession`, `DistributionUpdate` - World Sampler models
  - **Note:** These are duplicates of models in `dreams_laboratory.models` but stored in `deepgis_xr` database
- **Views:**
  - `index`, `label`, `label_3d`, `label_3d_dev`, `label_3d_sigma`
  - `label_topology`, `label_topology_sigma` (legacy and new versions)
  - `label_search`, `label_moon_viewer`
  - `stl_viewer`, `map_label`, `view_label`, `results`
  - `label_semi_supervised` - Semi-supervised labeling interface
  - `ai_analysis_report` - AI analysis reporting
  - Webclient API endpoints for labeling operations
  - World Sampler API endpoints (in `world_sampler_api.py`)
- **Features:** Main web interface, 3D labeling, geospatial visualization

**4. api App (`deepgis_xr.apps.api.v1`)**
- **Endpoints:**
  - `/api/v1/predict/tile/` - Tile prediction
  - `/api/v1/predict/save/` - Save predictions
  - `/api/v1/train/start/` - Start training
  - `/api/v1/train/status/<task_id>/` - Training status

**5. ml App (`deepgis_xr.apps.ml`)**
- **Services:**
  - `predictor.py` - ML prediction services
  - `trainer.py` - Model training services
  - `learning_tools.py` - ML utilities

#### URL Patterns (`deepgis_xr/urls.py`)
- Root: `/` (includes `deepgis_xr.apps.web.urls`)
- Auth: `/auth/*` (includes `deepgis_xr.apps.auth.urls`)
- API: `/api/v1/*` (includes `deepgis_xr.apps.api.v1.urls`)

#### Web URL Patterns (`deepgis_xr/apps/web/urls.py`)
- Main pages: `/`, `/label/`, `/label/3d/`, `/label/3d/dev/`, `/label/3d/sigma/`
- Topology: `/label/3d/topology/` (SIGMA), `/label/3d/topology/legacy/` (original)
- Search: `/label/3d/search/`
- Moon viewer: `/label/3d/moon/`
- STL viewer: `/stl-viewer/`
- Map label: `/map-label/`, `/view-label/`, `/results/`
- Webclient API: `/webclient/*` (various endpoints)
- Semi-supervised: `/label/semi-supervised/*`
- World Sampler: `/webclient/sampler/*`
- AI Analysis: `/ai-analysis/report/<session_id>/`

#### Settings (`deepgis_xr/settings.py`)
- **Database:** Single SQLite database at `deepgis-xr/db.sqlite3`
- **Python Path:** Adds `dreams_laboratory/scripts` to path for ML scripts (SAM, etc.)
- **Installed Apps:**
  - `deepgis_xr.apps.auth.apps.AuthConfig`
  - `deepgis_xr.apps.core`
  - `deepgis_xr.apps.api`
  - `deepgis_xr.apps.ml`
  - `deepgis_xr.apps.web`
- **Third-party:** `rest_framework`, `corsheaders`, `phonenumber_field`, `django.contrib.gis`

---

## 3. Integration Points

### 3.1 Database Integration

**Multi-Database Setup:**
- `dreams_laboratory` uses two databases:
  - `default`: Main database for most models
  - `deepgis_xr`: For World Sampler models (routed via `db_routers.py`)

**World Sampler Models:**
- Defined in both `dreams_laboratory.models` and `deepgis_xr.apps.web.models`
- Routed to `deepgis_xr` database via `WorldSamplerRouter`
- Models: `SampledLocation`, `SamplingSession`, `DistributionUpdate`

### 3.2 Script Integration

**ML Scripts Sharing:**
- `deepgis-xr` settings add `dreams_laboratory/scripts` to Python path
- Allows DeepGIS-XR to use SAM and other ML scripts from dreams_laboratory
- Docker volume mount: `../dreams_laboratory/scripts:/app/dreams_laboratory_scripts:ro`

### 3.3 API Integration

**Telemetry API:**
- `dreams_laboratory/api/` provides telemetry endpoints
- DeepGIS-XR topology viewer can display GPS session paths
- Documentation: `dreams_laboratory/api/GPS_PATHS_QUICKSTART.md`

### 3.4 URL Patterns

**Shared Patterns:**
- Both apps have `/label/semi-supervised/` endpoints
- Similar patterns but separate implementations
- `dreams_laboratory`: `/label/semi-supervised/`
- `deepgis-xr`: `/label/semi-supervised/`

---

## 4. Key Differences

| Aspect | dreams_laboratory | deepgis-xr |
|--------|------------------|------------|
| **Project Type** | Main Django project | Separate Django project |
| **Database** | Multi-DB (default + deepgis_xr) | Single DB (deepgis_xr) |
| **User Model** | `CustomUser` (phone number) | `User` (phone number, different app) |
| **Focus** | Lab management, tutorials, quizzes | Geospatial, labeling, XR |
| **URL Prefix** | Root (`/`) | Root (`/`) - separate server/port |
| **Static Files** | `staticfiles/` at root | `staticfiles/` in deepgis-xr/ |
| **Templates** | `templates/` at root | `templates/` in deepgis-xr/ |

---

## 5. Deployment Architecture

### 5.1 Separate Servers
- **dreams_laboratory:** Main server (likely port 8000 or similar)
- **deepgis-xr:** Separate server (port 8060 → 8090 in Docker)

### 5.2 Docker Setup
- `deepgis-xr` has its own `docker-compose.yml`
- Mounts `dreams_laboratory/scripts` for ML functionality
- Includes tileserver service for map tiles

### 5.3 Nginx Configuration
- Likely reverse proxy setup to route requests
- Separate domains or subdomains for each app

---

## 6. Model Duplication

### 6.1 World Sampler Models

**Location:**
- `dreams_laboratory/models.py` (lines 450-564)
- `deepgis_xr/apps/web/models.py` (lines 12-177)

**Purpose:**
- Models defined in `dreams_laboratory` but routed to `deepgis_xr` database
- Models also defined in `deepgis_xr` for direct access
- Allows both apps to work with same data

**Models:**
- `SampledLocation`
- `SamplingSession`
- `DistributionUpdate`

---

## 7. Ready for Changes

### 7.1 Current State Summary

✅ **dreams_laboratory:**
- Main Django project with lab management features
- Tutorial system, quiz system, telemetry tracking
- Multi-database setup with router
- Custom authentication with phone numbers

✅ **deepgis-xr:**
- Separate Django project for geospatial/XR features
- Labeling system, 3D visualization, vehicle tracking
- World Sampler functionality
- ML integration (SAM, Mask2Former)

### 7.2 Integration Points to Consider

1. **Database Routing:** World Sampler models use router
2. **Script Sharing:** ML scripts shared via Python path
3. **API Endpoints:** Telemetry API integration
4. **Model Duplication:** World Sampler models in both apps
5. **User Models:** Two separate custom user models

### 7.3 Potential Refactoring Areas

- Consolidate World Sampler models (remove duplication)
- Unify user models or create shared auth app
- Standardize API patterns between apps
- Consider monorepo vs. separate projects strategy

---

## 8. File Locations Reference

### dreams_laboratory
- Settings: `/home/jdas/dreams-lab-website-server/dreams_laboratory/settings.py`
- URLs: `/home/jdas/dreams-lab-website-server/dreams_laboratory/urls.py`
- Models: `/home/jdas/dreams-lab-website-server/dreams_laboratory/models.py`
- Views: `/home/jdas/dreams-lab-website-server/dreams_laboratory/views.py`
- DB Router: `/home/jdas/dreams-lab-website-server/dreams_laboratory/db_routers.py`

### deepgis-xr
- Settings: `/home/jdas/dreams-lab-website-server/deepgis-xr/deepgis_xr/settings.py`
- URLs: `/home/jdas/dreams-lab-website-server/deepgis-xr/deepgis_xr/urls.py`
- Web URLs: `/home/jdas/dreams-lab-website-server/deepgis-xr/deepgis_xr/apps/web/urls.py`
- Web Views: `/home/jdas/dreams-lab-website-server/deepgis-xr/deepgis_xr/apps/web/views.py`
- Web Models: `/home/jdas/dreams-lab-website-server/deepgis-xr/deepgis_xr/apps/web/models.py`
- Core Models: `/home/jdas/dreams-lab-website-server/deepgis-xr/deepgis_xr/apps/core/models.py`
- Auth Models: `/home/jdas/dreams-lab-website-server/deepgis-xr/deepgis_xr/apps/auth/models.py`

---

**Analysis Complete - Ready for Changes!**

