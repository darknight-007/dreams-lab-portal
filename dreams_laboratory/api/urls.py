"""
URL routing for telemetry API endpoints.
"""
from django.urls import path
from . import views

app_name = 'telemetry_api'

urlpatterns = [
    # API info endpoint (base path)
    path('telemetry/', views.api_info, name='api_info'),
    
    # Session management
    path('telemetry/session/create/', 
         views.create_telemetry_session, 
         name='create_telemetry_session'),
    
    # Individual telemetry endpoints
    path('telemetry/local-position-odom/', 
         views.post_local_position_odom, 
         name='post_local_position_odom'),
    
    path('telemetry/gps-fix-raw/', 
         views.post_gps_fix_raw, 
         name='post_gps_fix_raw'),
    
    path('telemetry/gps-fix-estimated/', 
         views.post_gps_fix_estimated, 
         name='post_gps_fix_estimated'),
    
    # Batch endpoint
    path('telemetry/batch/', 
         views.post_telemetry_batch, 
         name='post_telemetry_batch'),
    
    # Session path endpoints (for DeepGIS frontend)
    # IMPORTANT: More specific patterns must come before less specific ones
    path('telemetry/sessions/<str:session_id>/path/', 
         views.get_session_path, 
         name='get_session_path'),
    
    path('telemetry/sessions/', 
         views.list_sessions, 
         name='list_sessions'),
]

