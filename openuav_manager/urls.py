from django.urls import path
from . import views

app_name = 'openuav_manager'

urlpatterns = [
    # Main consolidated view
    path('', views.container_list, name='container_list'),  # This will now serve as the main page
    path('launch/', views.launch_openuav, name='launch_container'),
    path('containers/<str:container_id>/start/', views.container_action, {'action': 'start'}, name='start_container'),
    path('containers/<str:container_id>/stop/', views.container_action, {'action': 'stop'}, name='stop_container'),
    path('containers/<str:container_id>/delete/', views.container_action, {'action': 'delete'}, name='delete_container'),
    path('containers/<str:container_id>/logs/', views.container_logs, name='container_logs'),
    path('system/stats/', views.system_stats, name='system_stats'),
    path('config/', views.save_config, name='save_config'),
    path('batch-action/', views.batch_action, name='batch_action'),
]
