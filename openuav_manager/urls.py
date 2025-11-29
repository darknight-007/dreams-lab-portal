from django.urls import path
from . import views

app_name = 'openuav_manager'

urlpatterns = [
    path('', views.container_list, name='container_list'),
    path('manage/', views.manage_view, name='manage'),
    path('containers/<str:container_id>/<str:action>/', views.container_action, name='container_action'),
    path('launch/', views.launch_openuav, name='launch'),
    path('manage/launch/', views.launch_openuav, name='manage_launch'),
    path('manage/batch-action/', views.batch_action, name='batch_action'),
    path('stop/', views.stop_openuav, name='stop'),
    path('reset/', views.reset_openuav, name='reset'),
    path('status/', views.openuav_status, name='status'),
    path('status/update/', views.container_status_update, name='container_status_update'),
    
    # Swarm management URLs
    path('swarm/', views.swarm_view, name='swarm'),
    path('swarm/launch/', views.swarm_view, name='swarm_launch'),
    
    # ROS Manager URLs
    path('ros/browse/', views.rosbag_browser, name='rosbag_browser'),
    path('ros/visualize/', views.launch_rosbag_viewer, name='launch_rosbag_viewer'),
    path('ros/playback/<str:container_id>/', views.rosbag_playback_control, name='rosbag_playback_control'),
    path('ros/sessions/', views.rosbag_sessions, name='rosbag_sessions'),
]
