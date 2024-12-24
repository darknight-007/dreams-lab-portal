from django.urls import path
from . import views

app_name = 'openuav_manager'

urlpatterns = [
    path('', views.manage_openuav, name='manage'),
    path('containers/', views.container_list, name='container_list'),
    path('containers/<str:container_id>/<str:action>/', views.container_action, name='container_action'),
    path('launch/', views.launch_openuav, name='launch'),
    path('stop/', views.stop_openuav, name='stop'),
    path('status/', views.openuav_status, name='status'),
]
