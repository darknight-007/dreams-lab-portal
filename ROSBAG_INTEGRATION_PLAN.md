# ROS Manager Integration Plan

## 🎯 Overview

Leverage existing digital-twin container ecosystem to provide on-demand ROS2 rosbag visualization and ROS tools in DeepGIS.

## 🏗️ Architecture (Based on openuav_manager)

```
User Request → Django View → Launch ROS2 Container → Mount Rosbag → Visualize
                                      ↓
                            Track in Database (Container model)
                                      ↓
                         Provide VNC/Web Access to Visualization
```

## 📁 Rosbag Location

```bash
/mnt/tesseract-store/trike-backup/rosbag2_2023_11_17-09_54_09/
├── rosbag2_2023_11_17-09_54_09_0.db3  (31GB)
└── metadata.yaml                       (14KB)
```

## 🔧 Implementation Components

### 1. New Django App: `ros_manager`

Similar to `openuav_manager`, but for ROS2 rosbag visualization and tools.

```python
# ros_manager/models.py
class RosbagSession(models.Model):
    """Track active rosbag visualization sessions"""
    container_id = models.CharField(max_length=128, unique=True)
    short_id = models.CharField(max_length=12)
    name = models.CharField(max_length=255)
    rosbag_path = models.CharField(max_length=512)
    status = models.CharField(max_length=50)
    user = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(auto_now=True)
    vnc_url = models.URLField(max_length=512)
    visualization_type = models.CharField(max_length=50)  # rviz2, foxglove, plotjuggler
```

### 2. Container Launch Pattern (from openuav_manager)

```python
# ros_manager/views.py
def launch_rosbag_viewer(request):
    """Launch ROS2 container with rosbag mounted"""
    rosbag_path = request.POST.get('rosbag_path')
    username = request.POST.get('username')
    viz_type = request.POST.get('viz_type', 'rviz2')  # rviz2, foxglove, plotjuggler
    
    container_name = f'rosbag-viewer-{username}'
    
    # Launch command (following openuav pattern)
    launch_cmd = f"""docker run --init --runtime=nvidia \
        --network=dreamslab \
        --privileged \
        --name={container_name} \
        --hostname={container_name} \
        -d \
        -e DISPLAY=:1.0 \
        -v {rosbag_path}:/rosbags:ro \
        -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw \
        digital-twin-ras-ses-598:5.0 \
        /bin/bash -c "source /opt/ros/humble/setup.bash && {get_viz_command(viz_type)}"
    """
    
    # Execute and track (same pattern as openuav_manager)
    stdout, stderr, code = run_command(launch_cmd)
    
    if code == 0:
        container_id = stdout.strip()
        # Create session record
        RosbagSession.objects.create(
            container_id=container_id,
            short_id=container_id[:12],
            name=container_name,
            rosbag_path=rosbag_path,
            status='running',
            user=username,
            vnc_url=f"https://{container_name}.deepgis.org/vnc.html",
            visualization_type=viz_type
        )
        return JsonResponse({
            'status': 'success',
            'vnc_url': f"https://{container_name}.deepgis.org/vnc.html"
        })
```

### 3. Visualization Commands

```python
def get_viz_command(viz_type):
    """Get command to run inside container"""
    commands = {
        'rviz2': 'rviz2',
        'foxglove': 'foxglove-studio',
        'plotjuggler': 'ros2 run plotjuggler plotjuggler',
        'rqt_bag': 'ros2 run rqt_bag rqt_bag',
    }
    return commands.get(viz_type, 'rviz2')
```

### 4. Rosbag Playback Integration

```python
def play_rosbag(container_id, rosbag_path, rate=1.0, loop=False):
    """Play rosbag inside running container"""
    loop_flag = '--loop' if loop else ''
    play_cmd = f"""docker exec {container_id} /bin/bash -c \
        "source /opt/ros/humble/setup.bash && \
         ros2 bag play /rosbags/{os.path.basename(rosbag_path)} \
         --rate {rate} {loop_flag}
    " """
    
    return run_command(play_cmd)
```

### 5. DeepGIS Integration - New View

```python
# dreams_laboratory/views.py or new rosbag_views.py

def rosbag_browser(request):
    """Browse available rosbags"""
    rosbag_base = Path('/mnt/tesseract-store/trike-backup')
    
    # Find all rosbags
    rosbags = []
    for rosbag_dir in rosbag_base.glob('rosbag2_*'):
        metadata_file = rosbag_dir / 'metadata.yaml'
        if metadata_file.exists():
            # Parse metadata
            import yaml
            with open(metadata_file) as f:
                metadata = yaml.safe_load(f)
            
            # Get rosbag info
            rosbags.append({
                'path': str(rosbag_dir),
                'name': rosbag_dir.name,
                'size': get_dir_size(rosbag_dir),
                'metadata': metadata,
                'topics': metadata.get('rosbag2_bagfile_information', {}).get('topics_with_message_count', [])
            })
    
    return render(request, 'rosbag_browser.html', {
        'rosbags': rosbags
    })

def rosbag_visualize(request, rosbag_id):
    """Launch visualization for specific rosbag"""
    # Get rosbag path
    rosbag_path = get_rosbag_path(rosbag_id)
    
    # Check if already have active session
    active_session = RosbagSession.objects.filter(
        user=request.user.username,
        rosbag_path=rosbag_path,
        status='running'
    ).first()
    
    if active_session:
        return JsonResponse({
            'status': 'exists',
            'vnc_url': active_session.vnc_url
        })
    
    # Launch new session
    return launch_rosbag_viewer(request)
```

## 🎨 UI Components

### 1. Rosbag Browser Page

```html
<!-- templates/rosbag_browser.html -->
<div class="rosbag-list">
    {% for rosbag in rosbags %}
    <div class="rosbag-card">
        <h3>{{ rosbag.name }}</h3>
        <p>Size: {{ rosbag.size|filesizeformat }}</p>
        <p>Topics: {{ rosbag.topics|length }}</p>
        
        <div class="topics-preview">
            {% for topic in rosbag.topics|slice:":5" %}
            <span class="topic-tag">{{ topic.topic_metadata.name }}</span>
            {% endfor %}
        </div>
        
        <div class="actions">
            <button onclick="visualizeRosbag('{{ rosbag.path }}', 'rviz2')">
                View in RViz2
            </button>
            <button onclick="visualizeRosbag('{{ rosbag.path }}', 'foxglove')">
                View in Foxglove
            </button>
            <button onclick="visualizeRosbag('{{ rosbag.path }}', 'plotjuggler')">
                View in PlotJuggler
            </button>
        </div>
    </div>
    {% endfor %}
</div>
```

### 2. Viewer Interface (Embedded VNC)

```html
<!-- templates/rosbag_viewer.html -->
<div class="rosbag-viewer">
    <div class="controls">
        <button id="play-btn">▶ Play</button>
        <button id="pause-btn">⏸ Pause</button>
        <button id="stop-btn">⏹ Stop</button>
        <input type="range" id="rate-slider" min="0.1" max="2" step="0.1" value="1.0">
        <span id="rate-display">1.0x</span>
        <label>
            <input type="checkbox" id="loop-checkbox"> Loop
        </label>
    </div>
    
    <iframe id="vnc-frame" src="{{ vnc_url }}" width="100%" height="800px"></iframe>
</div>

<script>
function visualizeRosbag(rosbagPath, vizType) {
    fetch('/rosbag/visualize/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({
            rosbag_path: rosbagPath,
            viz_type: vizType
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            window.open(data.vnc_url, '_blank');
        }
    });
}
</script>
```

## 🔌 URL Routes

```python
# ros_manager/urls.py
from django.urls import path
from . import views

app_name = 'ros_manager'

urlpatterns = [
    path('', views.rosbag_list, name='rosbag_list'),
    path('browse/', views.rosbag_browser, name='rosbag_browser'),
    path('visualize/', views.launch_rosbag_viewer, name='visualize'),
    path('play/<str:container_id>/', views.play_rosbag, name='play'),
    path('stop/<str:container_id>/', views.stop_rosbag, name='stop'),
    path('sessions/', views.session_list, name='session_list'),
    path('session/<str:container_id>/stop/', views.stop_session, name='stop_session'),
]

# Main urls.py
urlpatterns = [
    ...
    path('ros/', include('ros_manager.urls')),
]
```

## 📊 Database Schema

```python
# ros_manager/models.py
from django.db import models

class RosbagFile(models.Model):
    """Catalog of available rosbags"""
    path = models.CharField(max_length=512, unique=True)
    name = models.CharField(max_length=255)
    size = models.BigIntegerField()  # bytes
    duration = models.FloatField(null=True)  # seconds
    topics = models.JSONField(default=list)
    metadata = models.JSONField(default=dict)
    created = models.DateTimeField()
    indexed = models.DateTimeField(auto_now_add=True)

class RosbagSession(models.Model):
    """Active visualization sessions"""
    container_id = models.CharField(max_length=128, unique=True)
    short_id = models.CharField(max_length=12)
    name = models.CharField(max_length=255)
    rosbag = models.ForeignKey(RosbagFile, on_delete=models.CASCADE)
    status = models.CharField(max_length=50)
    user = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(auto_now=True)
    vnc_url = models.URLField(max_length=512)
    visualization_type = models.CharField(max_length=50)
    playback_rate = models.FloatField(default=1.0)
    is_looping = models.BooleanField(default=False)
```

## 🚀 Implementation Steps

### Phase 1: Basic Container Launch (Day 1)
1. Create `ros_manager` Django app
2. Copy pattern from `openuav_manager/views.py`
3. Modify launch command to mount rosbag directory
4. Test basic RViz2 visualization

### Phase 2: Rosbag Catalog (Day 2)
1. Scan `/mnt/tesseract-store/trike-backup/` for rosbags
2. Parse metadata.yaml files
3. Create browse interface
4. Add topic filtering

### Phase 3: Playback Controls (Day 3)
1. Implement play/pause/stop via docker exec
2. Add rate control
3. Add loop functionality
4. Add topic selection

### Phase 4: DeepGIS Integration (Day 4)
1. Embed in DeepGIS UI
2. Add geospatial context (if rosbag has GPS)
3. Sync timeline with map
4. Export frames/data

## 💻 Quick Start Commands

```bash
# 1. Create Django app
cd /home/jdas/dreams-lab-website-server
python manage.py startapp ros_manager

# 2. Test rosbag in container manually
docker run --rm -it \
    --runtime=nvidia \
    --network=dreamslab \
    -v /mnt/tesseract-store/trike-backup/rosbag2_2023_11_17-09_54_09:/rosbags:ro \
    -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw \
    -e DISPLAY=:1.0 \
    digital-twin-ras-ses-598:5.0 \
    /bin/bash

# Inside container:
source /opt/ros/humble/setup.bash
ros2 bag info /rosbags
ros2 bag play /rosbags --rate 0.5

# In another terminal:
rviz2
```

## 🎯 Benefits of This Approach

1. **Reuses Existing Infrastructure**
   - Same container management pattern as openuav_manager
   - Same networking (dreamslab)
   - Same VNC access pattern
   - Same database tracking

2. **Scalable**
   - Multiple users can view different rosbags simultaneously
   - Each gets their own container
   - Auto-cleanup on timeout

3. **Secure**
   - Rosbags mounted read-only
   - Isolated containers
   - Existing authentication

4. **Flexible**
   - Support multiple visualization tools
   - Easy to add new tools
   - Can extend to rosbag editing/filtering

## 📝 Next Steps

Would you like me to:
1. ✅ Create the `ros_manager` Django app structure
2. ✅ Implement the container launch view (following openuav pattern)
3. ✅ Create the rosbag browser UI
4. ✅ Add playback controls
5. ✅ Integrate with DeepGIS

Ready to implement! Shall I start with creating the Django app?

