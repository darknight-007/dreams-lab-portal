"""
URL routing for telemetry API endpoints.
"""
from django.urls import path
from . import views

app_name = 'telemetry_api'

urlpatterns = [
    # API info endpoint (base path)
    path('telemetry/', views.api_info, name='api_info'),
    
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
]

