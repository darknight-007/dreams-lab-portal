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
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from dreams_laboratory.views import (
    # Core views
    deepgis_home, dreamslab_home, openuav_home, run_blender, get_models, get_available_layers,
    get_category_info,
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
    apply_filters, ransac_demo_data,
    ses598_robotic_exploration_and_mapping, ses598_quiz,
    # Course view
    ses598_course_view,
    reset_quiz,
    # Tutorial and Quiz URLs
    tutorial_view,
    quiz_view,
    validate_quiz_answer,
    get_tutorial_hints,
    save_quiz_progress,
    load_quiz_progress,
    # Widget views
    widget_view,
    # Drone buddy views
    drone_buddy_view,
    save_drone_code,
    run_drone_tests,
    cart_pole_lqr_view,
)
from django.http import HttpResponse

def health_check(request):
    return HttpResponse("OK")

urlpatterns = [
    # Core URLs
    path('admin/', admin.site.urls),
    path('', deepgis_home, name='home'),  # New DeepGIS home page
    path('dreamslab/', dreamslab_home, name='dreamslab_home'),  # DREAMS Lab home page
    path('api/get_available_layers', get_available_layers, name='get_available_layers'),
    path('webclient/get_available_layers', get_available_layers, name='get_available_layers_webclient'),  # Added for compatibility
    path('api/get_category_info', get_category_info, name='get_category_info'),
    
    # OpenUAV URLs
    path('openuav/', openuav_home, name='openuav_home'),  # Main OpenUAV entry point
    path('openuav/manage/', include('openuav_manager.urls')),  # OpenUAV management interface
    
    # DREAMS Lab URLs (moved under dreamslab/ prefix)
    path('dreamslab/run_blender/', run_blender, name='run_blender'),
    path('dreamslab/get_models/', get_models, name='get_models'),
    path('dreamslab/stereo_buddy/', stereo_buddy_view, name='stereo_buddy'),
    path('dreamslab/slam_buddy/', slam_buddy_view, name='slam_buddy'),
    path('dreamslab/cart_pole_buddy/', cart_pole_buddy_view, name='cart_pole_buddy'),
    path('dreamslab/gaussian_processes_buddy/', gaussian_processes_buddy_view, name='gaussian_processes_buddy'),
    path('dreamslab/param_estimation_buddy/', param_estimation_buddy_view, name='param_estimation_buddy'),
    path('dreamslab/image_buddy/', image_buddy_view, name='image_buddy'),  # Underscore version
    path('dreamslab/image-buddy/', image_buddy_view, name='image_buddy_hyphen'),  # Hyphen version for backward compatibility
    path('dreamslab/widget/<str:widget_type>/', widget_view, name='widget'),
    
    # Drone buddy specific URLs
    path('dreamslab/drone_buddy/', drone_buddy_view, name='drone_buddy'),
    path('dreamslab/drone-buddy/save-code', save_drone_code, name='save_drone_code'),
    path('dreamslab/drone-buddy/run-tests', run_drone_tests, name='run_drone_tests'),
    
    path('dreamslab/generate_batch_report/', generate_batch_report, name='generate_batch_report'),
    path('dreamslab/ransac_buddy/', ransac_buddy, name='ransac_buddy'),
    path('dreamslab/multiview_geometry_buddy/', multiview_geometry_view, name='multiview_geometry'),
    path('dreamslab/particle_filter_buddy/', particle_filter_buddy, name='particle_filter_buddy'),
    path('dreamslab/loop_closure_buddy/', loop_closure_buddy, name='loop_closure_buddy'),
    path('dreamslab/sensor_fusion_buddy/', sensor_fusion_buddy, name='sensor_fusion_buddy'),
    path('dreamslab/visual_odometry_buddy/', visual_odometry_buddy, name='visual_odometry_buddy'),
    path('dreamslab/point_cloud_buddy/', point_cloud_buddy, name='point_cloud_buddy'),
    path('dreamslab/path_planning_buddy/', path_planning_buddy, name='path_planning_buddy'),
    path('dreamslab/cart_pole_lqr_buddy/', cart_pole_lqr_view, name='cart_pole_lqr_buddy'),
    path('dreamslab/ses598/', ses598_course_view, name='ses598_course'),
    path('dreamslab/ses598/quiz/', ses598_quiz, name='ses598_quiz'),
    
    # Authentication URLs
    path('auth/initiate-login/', initiate_login, name='initiate_login'),
    path('auth/initiate-login-form/', initiate_login_view, name='initiate_login_form'),
    path('auth/verify-login/', verify_login, name='verify_login'),
    path('auth/verify-login-form/', verify_login_view, name='verify_login_form'),
    
    # API URLs
    path('api/apply-filters/', apply_filters, name='apply_filters'),
    path('api/ransac-demo-data/', ransac_demo_data, name='ransac-demo-data'),
    
    # Tutorial and Quiz URLs
    path('tutorial/<str:tutorial_type>/<str:tutorial_id>/', tutorial_view, name='tutorial'),
    path('quiz/<str:quiz_type>/<str:quiz_id>/', quiz_view, name='quiz'),
    path('quiz/<str:quiz_type>/<str:quiz_id>/validate/', validate_quiz_answer, name='validate_quiz'),
    path('tutorial/<str:tutorial_type>/<str:tutorial_id>/hints/', get_tutorial_hints, name='tutorial_hints'),
    path('quiz/progress/save', save_quiz_progress, name='save_quiz_progress'),
    path('quiz/progress/load', load_quiz_progress, name='load_quiz_progress'),
    
    # Health check
    path('health/', health_check),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
