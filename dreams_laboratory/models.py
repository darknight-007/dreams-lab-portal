from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser, User
from django.conf import settings


class CustomUser(AbstractUser):
    phone_number = models.CharField(max_length=15, unique=True, null=True, blank=True)
    is_phone_verified = models.BooleanField(default=False)


class People(models.Model):
  first_name = models.CharField(max_length=100)
  last_name = models.CharField(max_length=100)
  email = models.EmailField()
  profile_pic = models.URLField(max_length=300)
  bio = models.TextField(blank=True)
  role = models.ForeignKey('Role', on_delete=models.SET_NULL, null=True, blank=True)
  affiliation = models.CharField(max_length=200, blank=True, null=True)  # New field for affiliation

  def __str__(self):
      return f"{self.first_name} {self.last_name}"

# Other models remain unchanged

class Role(models.Model):
  position = models.CharField(max_length=100)
  start_date = models.DateField(default=timezone.now)
  stop_date = models.DateField(null=True, blank=True)
  email = models.EmailField()
  # Removed the affiliation field

  def __str__(self):
      return self.position
    


class Research(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    start_date = models.DateField()
    end_date = models.DateField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    effort = models.IntegerField(default=0, choices=[(i, i) for i in range(101)])
    style = models.CharField(max_length=200, default="'background-color: rgba(255,0,0,0.3);'")  # For example, could be 'background-color: rgba(255,0,0,0.3);'
    def __str__(self):
        return self.title





class Publication(models.Model):
  title = models.CharField(max_length=200)
  authors = models.TextField(blank=True, null=True)  # Changed to TextField in case there are many authors
  publication_date = models.DateField()
  abstract = models.TextField(blank=True, null=True)
  link = models.URLField(blank=True)
  doi = models.CharField(max_length=200, blank=True, null=True)

  def __str__(self):
      return self.title


class FundingSource(models.Model):
  source_name = models.CharField(max_length=255)
  photo_url = models.URLField(max_length=300, blank=True, null=True)  # Added photo URL field

  def __str__(self):
      return self.source_name

class Project(models.Model):
    title = models.CharField(max_length=200, default='Untitled Project')  # Added title field with default value
    funding_source = models.ForeignKey(FundingSource, on_delete=models.SET_NULL, null=True, blank=True, related_name='funded_projects')
    team_members = models.ManyToManyField(People, related_name='projects')
    publications = models.ManyToManyField(Publication, related_name='projects')
    research_areas = models.ManyToManyField(Research, related_name='projects')
    website_url = models.URLField(max_length=300, blank=True, null=True)  # Added URL field for project website

    def __str__(self):
        return self.title



class Asset(models.Model):
  asset_name = models.CharField(max_length=200,blank=True, null=True)
  description = models.TextField(blank=True, null=True)
  person = models.ForeignKey(People, on_delete=models.CASCADE, related_name='assets')
  project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='assets')

  def __str__(self):
    return self.asset_name

class Photo(models.Model):
  projects = models.ManyToManyField(Project, blank=True)
  researches = models.ManyToManyField(Research, blank=True)
  assets = models.ManyToManyField(Asset, blank=True)
  photo_url = models.URLField(max_length=300)
  caption = models.TextField(blank=True, default=' ')

  def __str__(self):
      return self.photo_url  # Or any other string representation you prefer.

class QuizSubmission(models.Model):
    quiz_id = models.CharField(max_length=8)
    session_id = models.CharField(max_length=64, null=True, blank=True)
    email = models.EmailField(help_text='Preferably your ASU email')  # Changed from asu_id to email
    submission_date = models.DateTimeField(auto_now_add=True)
    cv_score = models.FloatField(null=True, default=0)
    slam_score = models.FloatField(null=True, default=0)
    estimation_score = models.FloatField(null=True, default=0)
    sensing_score = models.FloatField(null=True, default=0)
    motion_score = models.FloatField(null=True, default=0)
    neural_score = models.FloatField(null=True, default=0)
    total_score = models.FloatField()
    
    # Question responses
    q1 = models.CharField(max_length=1, null=True, blank=True)
    q2 = models.CharField(max_length=1, null=True, blank=True)
    q3 = models.CharField(max_length=1, null=True, blank=True)
    q4 = models.CharField(max_length=1, null=True, blank=True)
    q5 = models.CharField(max_length=1, null=True, blank=True)
    q6 = models.CharField(max_length=1, null=True, blank=True)
    q7 = models.CharField(max_length=1, null=True, blank=True)
    q8 = models.CharField(max_length=1, null=True, blank=True)
    q9 = models.CharField(max_length=1, null=True, blank=True)
    q10 = models.CharField(max_length=1, null=True, blank=True)
    q11 = models.CharField(max_length=1, null=True, blank=True)
    q12 = models.CharField(max_length=1, null=True, blank=True)
    q13 = models.CharField(max_length=1, null=True, blank=True)
    q14 = models.CharField(max_length=1, null=True, blank=True)
    q15 = models.CharField(max_length=1, null=True, blank=True)

    class Meta:
        db_table = 'dreams_laboratory_quizsubmission'

    def __str__(self):
        return f"Quiz {self.quiz_id} - {self.email} - Score: {self.total_score}%"

class QuizProgress(models.Model):
    """Model to store user progress in quizzes"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    component_id = models.CharField(max_length=100)
    is_correct = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'component_id')
        ordering = ['created_at']

    def __str__(self):
        return f"{self.user.username} - {self.component_id} - {'Correct' if self.is_correct else 'Incorrect'}"


# ============================================================================
# Pixhawk Telemetry Models
# ============================================================================

class DroneTelemetrySession(models.Model):
    """
    Represents a telemetry recording session/flight for a drone.
    Links telemetry data to a specific drone asset and flight mission.
    """
    session_id = models.CharField(max_length=100, unique=True, help_text="Unique session identifier")
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name='telemetry_sessions', 
                              help_text="Drone asset this telemetry belongs to")
    project = models.ForeignKey(Project, on_delete=models.SET_NULL, null=True, blank=True,
                               related_name='telemetry_sessions', help_text="Associated project")
    
    # Session metadata
    start_time = models.DateTimeField(help_text="Session start timestamp")
    end_time = models.DateTimeField(null=True, blank=True, help_text="Session end timestamp")
    duration_seconds = models.FloatField(null=True, blank=True, help_text="Session duration in seconds")
    
    # Flight information
    flight_mode = models.CharField(max_length=50, blank=True, null=True, 
                                   help_text="PX4 flight mode (e.g., MANUAL, POSCTL, AUTO)")
    mission_type = models.CharField(max_length=100, blank=True, null=True,
                                   help_text="Type of mission/flight")
    notes = models.TextField(blank=True, null=True, help_text="Additional session notes")
    
    # Statistics (computed fields)
    total_telemetry_points = models.PositiveIntegerField(default=0, 
                                                         help_text="Total number of telemetry records")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-start_time']
        indexes = [
            models.Index(fields=['asset', '-start_time']),
            models.Index(fields=['session_id']),
            models.Index(fields=['start_time']),
        ]
    
    def __str__(self):
        return f"Session {self.session_id} - {self.asset.asset_name} - {self.start_time}"


class LocalPositionOdom(models.Model):
    """
    Local position odometry data from Pixhawk state estimator.
    Represents position and velocity in local NED (North-East-Down) frame.
    Maps to PX4 VehicleLocalPosition message.
    """
    session = models.ForeignKey(DroneTelemetrySession, on_delete=models.CASCADE, 
                               related_name='local_position_odom', 
                               help_text="Telemetry session this data belongs to")
    
    # Timestamp
    timestamp = models.DateTimeField(help_text="PX4 timestamp (microseconds since boot)")
    timestamp_usec = models.BigIntegerField(help_text="Microseconds since system boot")
    received_at = models.DateTimeField(auto_now_add=True, help_text="When record was received by server")
    
    # Position in local NED frame (meters)
    x = models.FloatField(help_text="North position (m)")
    y = models.FloatField(help_text="East position (m)")
    z = models.FloatField(help_text="Down position (m, positive down)")
    
    # Velocity in local NED frame (m/s)
    vx = models.FloatField(null=True, blank=True, help_text="North velocity (m/s)")
    vy = models.FloatField(null=True, blank=True, help_text="East velocity (m/s)")
    vz = models.FloatField(null=True, blank=True, help_text="Down velocity (m/s)")
    
    # Attitude
    heading = models.FloatField(null=True, blank=True, help_text="Heading angle (rad, 0 = North)")
    heading_rate = models.FloatField(null=True, blank=True, help_text="Heading rate (rad/s)")
    
    # Position covariance matrix (3x3, stored as JSON array)
    # Format: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    position_covariance = models.JSONField(default=list, blank=True, 
                                    help_text="Position covariance matrix (9 elements)")
    
    # Velocity covariance matrix (3x3, stored as JSON array)
    velocity_covariance = models.JSONField(default=list, blank=True,
                                    help_text="Velocity covariance matrix (9 elements)")
    
    # Validity flags (from PX4)
    xy_valid = models.BooleanField(default=False, help_text="XY position valid")
    z_valid = models.BooleanField(default=False, help_text="Z position valid")
    v_xy_valid = models.BooleanField(default=False, help_text="XY velocity valid")
    v_z_valid = models.BooleanField(default=False, help_text="Z velocity valid")
    heading_valid = models.BooleanField(default=False, help_text="Heading valid")
    
    # Reference frame information
    ref_timestamp = models.BigIntegerField(null=True, blank=True, 
                                          help_text="Reference timestamp for local frame")
    ref_lat = models.FloatField(null=True, blank=True, 
                               help_text="Reference latitude (deg) for local frame origin")
    ref_lon = models.FloatField(null=True, blank=True,
                               help_text="Reference longitude (deg) for local frame origin")
    ref_alt = models.FloatField(null=True, blank=True,
                               help_text="Reference altitude (m) for local frame origin")
    
    # Additional state estimator data
    dist_bottom = models.FloatField(null=True, blank=True, 
                                   help_text="Distance to bottom surface (m)")
    dist_bottom_valid = models.BooleanField(default=False, 
                                           help_text="Distance to bottom valid")
    dist_bottom_sensor_bitfield = models.IntegerField(null=True, blank=True,
                                                      help_text="Sensors used for bottom distance")
    
    # Euler angles (if available)
    eph = models.FloatField(null=True, blank=True, 
                           help_text="Estimated horizontal position error (m)")
    epv = models.FloatField(null=True, blank=True,
                           help_text="Estimated vertical position error (m)")
    evh = models.FloatField(null=True, blank=True,
                           help_text="Estimated horizontal velocity error (m/s)")
    evv = models.FloatField(null=True, blank=True,
                           help_text="Estimated vertical velocity error (m/s)")
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['session', '-timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['timestamp_usec']),
            # Composite index for time-range queries
            models.Index(fields=['session', 'timestamp']),
        ]
    
    def __str__(self):
        return f"LocalOdom {self.session.session_id} - ({self.x:.2f}, {self.y:.2f}, {self.z:.2f}) - {self.timestamp}"


class GPSFixRaw(models.Model):
    """
    Raw GPS fix data directly from GPS receiver.
    Maps to PX4 VehicleGPSPosition message (raw GPS data).
    """
    session = models.ForeignKey(DroneTelemetrySession, on_delete=models.CASCADE,
                               related_name='gps_fixes_raw',
                               help_text="Telemetry session this data belongs to")
    
    # Timestamp
    timestamp = models.DateTimeField(help_text="GPS timestamp")
    timestamp_usec = models.BigIntegerField(help_text="Microseconds since system boot")
    received_at = models.DateTimeField(auto_now_add=True, help_text="When record was received by server")
    
    # GPS position (WGS84)
    latitude = models.DecimalField(max_digits=17, decimal_places=14, 
                                  help_text="Latitude (deg)")
    longitude = models.DecimalField(max_digits=17, decimal_places=14,
                                   help_text="Longitude (deg)")
    altitude = models.FloatField(help_text="Altitude above MSL (m)")
    
    # GPS fix quality
    fix_type = models.IntegerField(help_text="GPS fix type: 0=no fix, 1=dead reckoning, 2=2D, 3=3D, 4=GPS+DR, 5=Time only")
    satellites_visible = models.PositiveIntegerField(help_text="Number of satellites visible")
    satellites_used = models.PositiveIntegerField(null=True, blank=True,
                                                  help_text="Number of satellites used in solution")
    
    # Dilution of Precision (DOP)
    hdop = models.FloatField(null=True, blank=True, help_text="Horizontal dilution of precision")
    vdop = models.FloatField(null=True, blank=True, help_text="Vertical dilution of precision")
    pdop = models.FloatField(null=True, blank=True, help_text="Position dilution of precision")
    
    # GPS accuracy
    eph = models.FloatField(null=True, blank=True, 
                           help_text="GPS horizontal position accuracy (m)")
    epv = models.FloatField(null=True, blank=True,
                           help_text="GPS vertical position accuracy (m)")
    s_variance_m_s = models.FloatField(null=True, blank=True,
                                      help_text="GPS speed accuracy (m/s)")
    
    # Velocity (if available from GPS)
    vel_n_m_s = models.FloatField(null=True, blank=True, help_text="North velocity (m/s)")
    vel_e_m_s = models.FloatField(null=True, blank=True, help_text="East velocity (m/s)")
    vel_d_m_s = models.FloatField(null=True, blank=True, help_text="Down velocity (m/s)")
    vel_m_s = models.FloatField(null=True, blank=True, help_text="Ground speed (m/s)")
    cog_rad = models.FloatField(null=True, blank=True, help_text="Course over ground (rad)")
    
    # Time information
    time_utc_usec = models.BigIntegerField(null=True, blank=True,
                                          help_text="UTC timestamp (microseconds)")
    
    # Noise and jamming
    noise_per_ms = models.IntegerField(null=True, blank=True,
                                      help_text="Noise level per millisecond")
    jamming_indicator = models.IntegerField(null=True, blank=True,
                                          help_text="Jamming indicator (0-100)")
    jamming_state = models.IntegerField(null=True, blank=True,
                                       help_text="Jamming state: 0=unknown, 1=ok, 2=warning, 3=critical")
    
    # Additional metadata
    device_id = models.IntegerField(null=True, blank=True, help_text="GPS device ID")
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['session', '-timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['latitude', 'longitude']),
            models.Index(fields=['fix_type']),
            # Composite index for time-range queries
            models.Index(fields=['session', 'timestamp']),
        ]
    
    def __str__(self):
        return f"GPSRaw {self.session.session_id} - ({self.latitude}, {self.longitude}) - Fix:{self.fix_type} - {self.timestamp}"


class GPSFixEstimated(models.Model):
    """
    Estimated GPS position from state estimator (filtered/fused).
    Represents the state estimator's best estimate of global position,
    typically from fusing GPS, IMU, and other sensors.
    """
    session = models.ForeignKey(DroneTelemetrySession, on_delete=models.CASCADE,
                               related_name='gps_fixes_estimated',
                               help_text="Telemetry session this data belongs to")
    
    # Timestamp
    timestamp = models.DateTimeField(help_text="Estimate timestamp")
    timestamp_usec = models.BigIntegerField(help_text="Microseconds since system boot")
    received_at = models.DateTimeField(auto_now_add=True, help_text="When record was received by server")
    
    # Estimated position (WGS84)
    latitude = models.DecimalField(max_digits=17, decimal_places=14,
                                  help_text="Estimated latitude (deg)")
    longitude = models.DecimalField(max_digits=17, decimal_places=14,
                                   help_text="Estimated longitude (deg)")
    altitude = models.FloatField(help_text="Estimated altitude above MSL (m)")
    
    # Position covariance/uncertainty
    # Format: [lat_lat, lat_lon, lat_alt, lon_lat, lon_lon, lon_alt, alt_lat, alt_lon, alt_alt]
    position_covariance = models.JSONField(default=list, blank=True,
                                   help_text="Position covariance matrix (9 elements)")
    
    # Estimated velocity
    vel_n_m_s = models.FloatField(null=True, blank=True, help_text="Estimated North velocity (m/s)")
    vel_e_m_s = models.FloatField(null=True, blank=True, help_text="Estimated East velocity (m/s)")
    vel_d_m_s = models.FloatField(null=True, blank=True, help_text="Estimated Down velocity (m/s)")
    
    # Velocity covariance
    velocity_covariance = models.JSONField(default=list, blank=True,
                                   help_text="Velocity covariance matrix (9 elements)")
    
    # Estimation quality metrics
    eph = models.FloatField(null=True, blank=True,
                           help_text="Estimated horizontal position error (m)")
    epv = models.FloatField(null=True, blank=True,
                           help_text="Estimated vertical position error (m)")
    evh = models.FloatField(null=True, blank=True,
                           help_text="Estimated horizontal velocity error (m/s)")
    evv = models.FloatField(null=True, blank=True,
                           help_text="Estimated vertical velocity error (m/s)")
    
    # Estimation source/confidence
    estimator_type = models.CharField(max_length=50, blank=True, null=True,
                                     help_text="State estimator type (e.g., EKF2, LPE)")
    confidence = models.FloatField(null=True, blank=True,
                                  help_text="Estimation confidence (0-1)")
    
    # Validity flags
    position_valid = models.BooleanField(default=False, help_text="Estimated position valid")
    velocity_valid = models.BooleanField(default=False, help_text="Estimated velocity valid")
    
    # Reference to raw GPS (if available)
    raw_gps_fix = models.ForeignKey(GPSFixRaw, on_delete=models.SET_NULL, null=True, blank=True,
                                   related_name='estimated_fixes',
                                   help_text="Corresponding raw GPS fix (if available)")
    
    # Reference to local position odom (if available)
    local_position = models.ForeignKey(LocalPositionOdom, on_delete=models.SET_NULL, null=True, blank=True,
                                     related_name='estimated_gps_fixes',
                                     help_text="Corresponding local position odometry (if available)")
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['session', '-timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['latitude', 'longitude']),
            # Composite index for time-range queries
            models.Index(fields=['session', 'timestamp']),
        ]
    
    def __str__(self):
        return f"GPSEst {self.session.session_id} - ({self.latitude}, {self.longitude}) - {self.timestamp}"