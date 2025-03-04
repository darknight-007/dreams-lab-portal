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
    path('status/', views.openuav_status, name='status'),
    path('status/update/', views.container_status_update, name='container_status_update'),
]
