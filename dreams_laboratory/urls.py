"""django_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from dreams_laboratory.views import (
    # Core views
    home_view, run_blender, get_models,
    # Authentication views
    initiate_login, verify_login, initiate_login_view, verify_login_view,
    # Main feature views
    stereo_buddy_view, slam_buddy_view, cart_pole_buddy_view,
    gaussian_processes_buddy_view, param_estimation_buddy_view,
    image_buddy_view, generate_batch_report,
    # Computer vision views
    ransac_buddy, multiview_geometry_view,
    # Tutorial views
    particle_filter_buddy, loop_closure_buddy, sensor_fusion_buddy,
    visual_odometry_buddy, point_cloud_buddy, path_planning_buddy,
    # API views
    apply_filters, ransac_demo_data, ses598_robotic_exploration_and_mapping_quiz,
    ses598_robotic_exploration_and_mapping
)

urlpatterns = [
    # Core URLs
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path("run_blender/", run_blender, name="run_blender"),
    path('get_models/', get_models, name='get_models'),
    
    # Authentication URLs
    path('initiate-login/', initiate_login, name='initiate_login'),
    path('initiate-login-form/', initiate_login_view, name='initiate_login_form'),
    path('verify-login/', verify_login, name='verify_login'),
    path('verify-login-form/', verify_login_view, name='verify_login_form'),
    
    # Main feature URLs
    path('stereo-buddy/', stereo_buddy_view, name='stereo_buddy'),
    path('slam-buddy/', slam_buddy_view, name='slam_buddy'),
    path('cart-pole-buddy/', cart_pole_buddy_view, name='cart_pole_buddy'),
    path('gaussian-processes-buddy/', gaussian_processes_buddy_view, name='gaussian_processes_buddy'),
    path('param-estimation-buddy/', param_estimation_buddy_view, name='param_estimation_buddy'),
    path('image-buddy/', image_buddy_view, name='image_buddy'),
    path('generate_batch_report/', generate_batch_report, name='generate_batch_report'),
    path("api/apply-filters/", apply_filters, name="apply_filters"),
    path('ses598_quiz/', ses598_robotic_exploration_and_mapping_quiz, name='ses598_robotic_exploration_and_mapping_quiz'),
    path('rem/', ses598_robotic_exploration_and_mapping, name='ses598_robotic_exploration_and_mapping'),
  
    # Computer vision URLs
    path('ransac_buddy/', ransac_buddy, name='ransac-buddy'),
    path('multiview-geometry/', multiview_geometry_view, name='multiview_geometry'),
    path('ransac-demo-data/', ransac_demo_data, name='ransac-demo-data'),
    
    # Tutorial URLs
    path('particle-filter/', particle_filter_buddy, name='particle_filter_buddy'),
    path('loop-closure/', loop_closure_buddy, name='loop_closure_buddy'),
    path('sensor-fusion/', sensor_fusion_buddy, name='sensor_fusion_buddy'),
    path('visual-odometry/', visual_odometry_buddy, name='visual_odometry_buddy'),
    path('point-cloud/', point_cloud_buddy, name='point_cloud_buddy'),
    path('path-planning/', path_planning_buddy, name='path_planning_buddy'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
