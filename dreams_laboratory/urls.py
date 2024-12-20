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
from dreams_laboratory.views import (home_view, stereo_buddy_view, run_blender, get_models, initiate_login,
                                     verify_login, initiate_login_view, verify_login_view, cart_pole_buddy_view,
                                     gaussian_processes_buddy_view, generate_batch_report,
                                     param_estimation_buddy_view, image_buddy_view, slam_buddy_view,
                                     ses598_robotic_exploration_and_mapping_quiz, ransac_buddy, multiview_geometry_view)
from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from . import views

def ses598_robotic_exploration_and_mapping(request):
    return render(request, 'SES598_robotic_exploration_and_mapping.html')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path("run_blender/", run_blender, name="run_blender"),
    path('stereo-buddy/', stereo_buddy_view, name='stereo_buddy'),
    path('slam-buddy/', slam_buddy_view, name='slam_buddy'),

    path('cart-pole-buddy/', cart_pole_buddy_view, name='cart_pole_buddy'),
    path('gaussian-processes-buddy/', gaussian_processes_buddy_view, name='gaussian_processes_buddy'),
    path('param-estimation-buddy/', param_estimation_buddy_view, name='param_estimation_buddy'),
    path('generate_batch_report/', generate_batch_report, name='generate_batch_report'),
    path('image-buddy/', image_buddy_view, name='image_buddy'),
    path('ses598_quiz/', ses598_robotic_exploration_and_mapping_quiz, name='ses598_robotic_exploration_and_mapping_quiz'),
    path('ransac_buddy/', ransac_buddy, name='ransac-buddy'),
    path("api/apply-filters/", views.apply_filters, name="apply_filters"),
    # Add route for fetching available rock models
    path('get_models/', get_models, name='get_models'),
    path('initiate-login/', initiate_login, name='initiate_login'),
    path('verify-login/', verify_login, name='verify_login'),
    path('initiate-login-form/', initiate_login_view, name='initiate_login_form'),
    path('verify-login/', verify_login, name='verify_login'),
    path('verify-login-form/', verify_login_view, name='verify_login_form'),
    path('multiview-geometry/', multiview_geometry_view, name='multiview_geometry'),
    path('robotic-exploration-and-mapping/', views.ses598_robotic_exploration_and_mapping, name='ses598_robotic_exploration_and_mapping'),

    # Add any additional API endpoints needed for multiview functionality, for example:
    # path('api/multiview/process-images/', views.process_multiview_images, name='process_multiview_images'),
    # path('api/multiview/get-results/', views.get_multiview_results, name='get_multiview_results'),
]

if settings.DEBUG:
  urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
  urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)