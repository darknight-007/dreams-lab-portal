"""django_project URL Configuration"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from dreams_laboratory.views import (
    home_view, dreamslab_home_view,
    multiview_geometry_view, stereo_buddy_view,
    slam_buddy_view, loop_closure_buddy_view,
    param_estimation_buddy_view, ransac_buddy_view,
    gaussian_processes_buddy_view, path_planning_buddy_view,
    cart_pole_lqr_buddy_view, particle_filter_buddy_view,
    point_cloud_buddy, sensor_fusion_buddy,
    image_buddy_view, ses598_course_view, ses598_quiz,
    drone_buddy_view, save_drone_code, run_drone_tests,
    sampling_buddy_view, gp_ucb_buddy_view,
    bundle_adjustment_buddy_view, tutorials_home,
    stereo_buddy, slam_buddy, sensor_fusion_buddy,
    multiview_geometry, ses598_quiz_part2, generate_certificate,
    reset_quiz, quiz_admin_view, multi_armed_bandit_buddy_view,
    # Semi-supervised labeling views
    semi_supervised_label_view, generate_assisted_labels,
    save_assisted_labels, get_label_images
)
from dreams_laboratory.quiz_views import ses598_2025_retrospective
from django.http import HttpResponse

def health_check(request):
    return HttpResponse("OK")

urlpatterns = [
    path('admin/', admin.site.urls),  # Django admin URLs
    path('', home_view, name='home'),
    path('dreamslab/', dreamslab_home_view, name='dreamslab_home'),
    path('dreamslab/ses598/', ses598_course_view, name='ses598_course'),
    path('dreamslab/ses598/quiz/', ses598_quiz, name='ses598_quiz'),
    path('dreamslab/ses598/quiz/part2/', ses598_quiz_part2, name='ses598_quiz_part2'),
    path('dreamslab/ses598/quiz/certificate/', generate_certificate, name='generate_certificate'),
    path('dreamslab/ses598/quiz/reset/', reset_quiz, name='reset_quiz'),
    path('dreamslab/ses598/quiz/admin/', quiz_admin_view, name='quiz_admin'),
    path('dreamslab/ses598/2025-retrospective/', ses598_2025_retrospective, name='ses598_2025_retrospective'),
    path('tutorials/multiview-geometry/', multiview_geometry_view, name='multiview_geometry'),
    path('tutorials/stereo-buddy/', stereo_buddy_view, name='stereo_buddy'),
    path('tutorials/slam-buddy/', slam_buddy_view, name='slam_buddy'),
    path('tutorials/bundle-adjustment-buddy/', bundle_adjustment_buddy_view, name='bundle_adjustment_buddy'),
    path('tutorials/param-estimation-buddy/', param_estimation_buddy_view, name='param_estimation_buddy'),
    path('tutorials/ransac-buddy/', ransac_buddy_view, name='ransac_buddy'),
    path('tutorials/gaussian-processes-buddy/', gaussian_processes_buddy_view, name='gaussian_processes_buddy'),
    path('tutorials/sampling-buddy/', sampling_buddy_view, name='sampling_buddy'),
    path('tutorials/gp-ucb-buddy/', gp_ucb_buddy_view, name='gp_ucb_buddy'),
    path('tutorials/path-planning-buddy/', path_planning_buddy_view, name='path_planning_buddy'),
    path('tutorials/cart-pole-lqr-buddy/', cart_pole_lqr_buddy_view, name='cart_pole_lqr_buddy'),
    path('tutorials/image-buddy/', image_buddy_view, name='image_buddy'),
    path('tutorials/particle-filter-buddy/', particle_filter_buddy_view, name='particle_filter_buddy'),
    path('tutorials/point-cloud-buddy/', point_cloud_buddy, name='point_cloud_buddy'),
    path('tutorials/sensor-fusion-buddy/', sensor_fusion_buddy, name='sensor_fusion_buddy'),
    path('tutorials/drone-buddy/', drone_buddy_view, name='drone_buddy'),
    path('tutorials/multi-armed-bandit-buddy/', multi_armed_bandit_buddy_view, name='multi_armed_bandit_buddy'),
    path('dreamslab/drone-buddy/save-code', save_drone_code, name='save_drone_code'),
    path('dreamslab/drone-buddy/run-tests', run_drone_tests, name='run_drone_tests'),
    path('openuav/', include('openuav_manager.urls')),
    path('tutorials/', tutorials_home, name='tutorials_home'),
    path('tutorials/stereo-buddy/', stereo_buddy, name='stereo_buddy'),
    path('tutorials/slam-buddy/', slam_buddy, name='slam_buddy'),
    path('tutorials/sensor-fusion-buddy/', sensor_fusion_buddy, name='sensor_fusion_buddy'),
    path('tutorials/multiview-geometry/', multiview_geometry, name='multiview_geometry'),
    # Semi-supervised labeling paths (following deepgis/label pattern)
    path('label/semi-supervised/', semi_supervised_label_view, name='semi_supervised_label'),
    path('label/semi-supervised/api/generate-labels/', generate_assisted_labels, name='generate_assisted_labels'),
    path('label/semi-supervised/api/save-labels/', save_assisted_labels, name='save_assisted_labels'),
    path('label/semi-supervised/api/get-images/', get_label_images, name='get_label_images'),
    # Telemetry API endpoints
    path('api/', include('dreams_laboratory.api.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
