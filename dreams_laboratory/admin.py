from django.contrib import admin
from .models import (
    People, Research, Publication, Project, Role, Photo, Asset, FundingSource, 
    QuizSubmission, QuizProgress,
    DroneTelemetrySession, LocalPositionOdom, GPSFixRaw, GPSFixEstimated,
    SampledLocation, SamplingSession, DistributionUpdate,
    Mission, MissionWaypoint, Vehicle, VehicleType
)
from django.utils.html import format_html
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    pass


# Register your models here.
admin.site.register(People)
admin.site.register(Research)
admin.site.register(Publication)
admin.site.register(Project)

admin.site.register(Role)
@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ('thumbnail', 'photo_url',)  # Add other fields to display if needed

    def thumbnail(self, obj):
        return format_html('<img src="{}" style="width: 50px; height:auto;">', obj.photo_url)
    
    thumbnail.short_description = 'Thumbnail'  # Optional: Sets column name
  
@admin.register(Asset)
class AssetAdmin(admin.ModelAdmin):
    list_display = ('asset_name',)  # Add other fields if you want them to show up in the list as well


@admin.register(FundingSource)
class AssetAdmin(admin.ModelAdmin):
    list_display = ('source_name',)  # Add other fields if you want them to show up in the list as well

@admin.register(QuizSubmission)
class QuizSubmissionAdmin(admin.ModelAdmin):
    list_display = ('get_submission_title', 'quiz_type', 'submission_date', 'total_score')
    list_filter = ('submission_date', 'quiz_id')
    search_fields = ('email', 'quiz_id', 'session_id')
    readonly_fields = ('quiz_id', 'submission_date', 'total_score', 
                      'cv_score', 'slam_score', 'estimation_score', 
                      'sensing_score', 'motion_score', 'neural_score',
                      'q1', 'q2', 'q3', 'q4', 'q5', 
                      'q6', 'q7', 'q8', 'q9', 'q10',
                      'q11', 'q12', 'q13', 'q14', 'q15')
    
    def get_submission_title(self, obj):
        return f"{obj.email} - {obj.submission_date.strftime('%Y-%m-%d %H:%M:%S')}"
    get_submission_title.short_description = 'Submission Info'
    
    def quiz_type(self, obj):
        quiz_types = {
            'SES598': 'Enrollment Quiz P1',
            'SES598_Part2': 'Enrollment Quiz P2',
            'SES598_2025_RETRO_P1': 'Retrospective Q1-15',
            'SES598_2025_RETRO_P2': 'Retrospective Q16-25'
        }
        return quiz_types.get(obj.quiz_id, obj.quiz_id)
    quiz_type.short_description = 'Quiz Type'
    
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return True


@admin.register(QuizProgress)
class QuizProgressAdmin(admin.ModelAdmin):
    list_display = ('user', 'component_id', 'is_correct', 'created_at', 'updated_at')
    list_filter = ('is_correct', 'created_at', 'updated_at')
    search_fields = ('user__username', 'user__email', 'component_id')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('User & Component', {
            'fields': ('user', 'component_id')
        }),
        ('Progress', {
            'fields': ('is_correct',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


# ============================================================================
# Pixhawk Telemetry Admin
# ============================================================================

@admin.register(DroneTelemetrySession)
class DroneTelemetrySessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'asset', 'start_time', 'end_time', 'duration_seconds', 
                   'flight_mode', 'total_telemetry_points')
    list_filter = ('start_time', 'flight_mode', 'asset', 'project')
    search_fields = ('session_id', 'asset__asset_name', 'mission_type', 'notes')
    readonly_fields = ('total_telemetry_points', 'created_at', 'updated_at')
    date_hierarchy = 'start_time'
    
    fieldsets = (
        ('Session Information', {
            'fields': ('session_id', 'asset', 'project', 'start_time', 'end_time', 'duration_seconds')
        }),
        ('Flight Information', {
            'fields': ('flight_mode', 'mission_type', 'notes')
        }),
        ('Statistics', {
            'fields': ('total_telemetry_points',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(LocalPositionOdom)
class LocalPositionOdomAdmin(admin.ModelAdmin):
    list_display = ('session', 'timestamp', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'heading', 
                   'xy_valid', 'z_valid')
    list_filter = ('timestamp', 'xy_valid', 'z_valid', 'v_xy_valid', 'v_z_valid', 'heading_valid')
    search_fields = ('session__session_id', 'session__asset__asset_name')
    readonly_fields = ('received_at',)
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Session & Timestamp', {
            'fields': ('session', 'timestamp', 'timestamp_usec', 'received_at')
        }),
        ('Position (NED frame)', {
            'fields': ('x', 'y', 'z', 'xy_valid', 'z_valid')
        }),
        ('Velocity (NED frame)', {
            'fields': ('vx', 'vy', 'vz', 'v_xy_valid', 'v_z_valid')
        }),
        ('Attitude', {
            'fields': ('heading', 'heading_rate', 'heading_valid')
        }),
        ('Covariance Matrices', {
            'fields': ('position_covariance', 'velocity_covariance'),
            'classes': ('collapse',)
        }),
        ('Reference Frame', {
            'fields': ('ref_timestamp', 'ref_lat', 'ref_lon', 'ref_alt'),
            'classes': ('collapse',)
        }),
        ('Additional Data', {
            'fields': ('dist_bottom', 'dist_bottom_valid', 'dist_bottom_sensor_bitfield',
                      'eph', 'epv', 'evh', 'evv'),
            'classes': ('collapse',)
        }),
    )


@admin.register(GPSFixRaw)
class GPSFixRawAdmin(admin.ModelAdmin):
    list_display = ('session', 'timestamp', 'latitude', 'longitude', 'altitude', 
                   'fix_type', 'satellites_visible', 'eph', 'epv')
    list_filter = ('timestamp', 'fix_type', 'jamming_state')
    search_fields = ('session__session_id', 'session__asset__asset_name')
    readonly_fields = ('received_at',)
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Session & Timestamp', {
            'fields': ('session', 'timestamp', 'timestamp_usec', 'received_at', 'time_utc_usec')
        }),
        ('Position (WGS84)', {
            'fields': ('latitude', 'longitude', 'altitude')
        }),
        ('GPS Quality', {
            'fields': ('fix_type', 'satellites_visible', 'satellites_used', 
                      'hdop', 'vdop', 'pdop', 'eph', 'epv', 's_variance_m_s')
        }),
        ('Velocity', {
            'fields': ('vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s', 'vel_m_s', 'cog_rad')
        }),
        ('Jamming & Noise', {
            'fields': ('noise_per_ms', 'jamming_indicator', 'jamming_state'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('device_id',),
            'classes': ('collapse',)
        }),
    )


@admin.register(GPSFixEstimated)
class GPSFixEstimatedAdmin(admin.ModelAdmin):
    list_display = ('session', 'timestamp', 'latitude', 'longitude', 'altitude', 
                   'estimator_type', 'position_valid', 'velocity_valid', 'eph', 'epv')
    list_filter = ('timestamp', 'estimator_type', 'position_valid', 'velocity_valid')
    search_fields = ('session__session_id', 'session__asset__asset_name', 'estimator_type')
    readonly_fields = ('received_at',)
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Session & Timestamp', {
            'fields': ('session', 'timestamp', 'timestamp_usec', 'received_at')
        }),
        ('Estimated Position (WGS84)', {
            'fields': ('latitude', 'longitude', 'altitude', 'position_valid')
        }),
        ('Estimated Velocity', {
            'fields': ('vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s', 'velocity_valid')
        }),
        ('Covariance Matrices', {
            'fields': ('position_covariance', 'velocity_covariance'),
            'classes': ('collapse',)
        }),
        ('Quality Metrics', {
            'fields': ('eph', 'epv', 'evh', 'evv', 'estimator_type', 'confidence')
        }),
        ('References', {
            'fields': ('raw_gps_fix', 'local_position'),
            'classes': ('collapse',)
        }),
    )


# ============================================================================
# World Sampler Admin (for DeepGIS-XR geospatial sampling)
# ============================================================================

@admin.register(SampledLocation)
class SampledLocationAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'latitude',
        'longitude',
        'zoom_level',
        'score',
        'weight',
        'session_id',
        'sampled_at',
        'scored_at'
    ]
    list_filter = [
        'session_id',
        'zoom_level',
        'sampled_at',
        'scored_at',
    ]
    search_fields = [
        'latitude',
        'longitude',
        'session_id',
    ]
    readonly_fields = [
        'sampled_at',
    ]
    ordering = ['-sampled_at']
    
    fieldsets = (
        ('Location', {
            'fields': ('latitude', 'longitude', 'altitude', 'zoom_level')
        }),
        ('Scoring', {
            'fields': ('score', 'weight')
        }),
        ('Session', {
            'fields': ('session_id', 'user')
        }),
        ('Timestamps', {
            'fields': ('sampled_at', 'scored_at')
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
    )


@admin.register(SamplingSession)
class SamplingSessionAdmin(admin.ModelAdmin):
    list_display = [
        'session_id',
        'user',
        'initialization_method',
        'num_points',
        'total_samples',
        'total_updates',
        'created_at'
    ]
    list_filter = [
        'initialization_method',
        'created_at',
    ]
    search_fields = [
        'session_id',
    ]
    readonly_fields = [
        'created_at',
        'updated_at',
        'total_samples',
        'total_updates',
    ]
    ordering = ['-created_at']
    
    fieldsets = (
        ('Session Info', {
            'fields': ('session_id', 'user')
        }),
        ('Initialization', {
            'fields': (
                'num_points',
                'initialization_method',
                'lat_range_min',
                'lat_range_max',
                'lon_range_min',
                'lon_range_max',
                'alt_range_min',
                'alt_range_max',
            )
        }),
        ('Statistics', {
            'fields': ('total_samples', 'total_updates')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(DistributionUpdate)
class DistributionUpdateAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'session',
        'update_rule',
        'learning_rate',
        'radius',
        'applied_at'
    ]
    list_filter = [
        'update_rule',
        'applied_at',
    ]
    search_fields = [
        'session__session_id',
    ]
    readonly_fields = [
        'applied_at',
    ]
    ordering = ['-applied_at']
    filter_horizontal = ['feedback_locations']
    
    fieldsets = (
        ('Session', {
            'fields': ('session',)
        }),
        ('Update Parameters', {
            'fields': ('update_rule', 'learning_rate', 'radius')
        }),
        ('Feedback', {
            'fields': ('feedback_locations',)
        }),
        ('Metadata', {
            'fields': ('parameters', 'applied_at'),
            'classes': ('collapse',)
        }),
    )


# ============================================================================
# Mission Planning Admin (using models from dreams_laboratory.models)
# ============================================================================

class MissionWaypointInline(admin.TabularInline):
    model = MissionWaypoint
    fields = ('sequence', 'latitude', 'longitude', 'altitude', 'waypoint_type', 'command', 'speed', 'yaw')
    extra = 0
    show_change_link = True
    can_delete = True
    ordering = ['sequence']
    
    def get_queryset(self, request):
        return super().get_queryset(request).using('deepgis_xr')
    
    def save_model(self, request, obj, form, change):
        obj.save(using='deepgis_xr')

@admin.register(Mission)
class MissionAdmin(admin.ModelAdmin):
    list_display = ('name', 'mission_type', 'status', 'vehicle', 'num_waypoints', 'created_by_id', 'created_at')
    list_filter = ('status', 'mission_type', 'created_at')
    search_fields = ('name', 'description', 'vehicle__name')
    readonly_fields = ('num_waypoints', 'total_distance', 'created_at', 'updated_at', 'uploaded_at', 'started_at', 'completed_at')
    inlines = [MissionWaypointInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'mission_type', 'status', 'vehicle', 'created_by_id')
        }),
        ('Mission Parameters', {
            'fields': ('default_altitude', 'default_speed', 'return_to_home', 'waypoints')
        }),
        ('Statistics', {
            'fields': ('num_waypoints', 'total_distance'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'uploaded_at', 'started_at', 'completed_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).using('deepgis_xr')
    
    def save_model(self, request, obj, form, change):
        obj.save(using='deepgis_xr')
    
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        # Vehicle is in the same database (deepgis_xr)
        if db_field.name == 'vehicle':
            kwargs['queryset'] = db_field.related_model.objects.using('deepgis_xr')
        return super().formfield_for_foreignkey(db_field, request, **kwargs)

@admin.register(MissionWaypoint)
class MissionWaypointAdmin(admin.ModelAdmin):
    list_display = ('mission', 'sequence', 'latitude', 'longitude', 'altitude', 'waypoint_type', 'command')
    list_filter = ('waypoint_type', 'mission__status', 'mission__mission_type')
    search_fields = ('mission__name',)
    readonly_fields = ('created_at',)
    ordering = ['mission', 'sequence']
    
    def get_queryset(self, request):
        return super().get_queryset(request).using('deepgis_xr')
    
    def save_model(self, request, obj, form, change):
        obj.save(using='deepgis_xr')

@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = ('name', 'vehicle_id', 'vehicle_type', 'status', 'current_latitude', 'current_longitude', 'last_update')
    list_filter = ('status', 'vehicle_type', 'last_update')
    search_fields = ('name', 'vehicle_id')
    readonly_fields = ('last_update', 'created_at', 'position_age_seconds')
    
    def get_queryset(self, request):
        return super().get_queryset(request).using('deepgis_xr')
    
    def save_model(self, request, obj, form, change):
        obj.save(using='deepgis_xr')

@admin.register(VehicleType)
class VehicleTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'icon_symbol')
    list_filter = ('category',)
    search_fields = ('name',)
    
    def get_queryset(self, request):
        return super().get_queryset(request).using('deepgis_xr')
    
    def save_model(self, request, obj, form, change):
        obj.save(using='deepgis_xr')


