import os
import json
import logging
import numpy as np
import io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d  # Correct import for convolve2d
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from matplotlib.backends.backend_svg import FigureCanvasSVG
from django.conf import settings
import subprocess
import os
from .models import People, Research, Publication, Project, Asset, FundingSource
from django.http import JsonResponse
import json
import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import subprocess
import tempfile

from django.conf import settings
from urllib.parse import urljoin

# Set up logging
logger = logging.getLogger(__name__)


@csrf_exempt
def image_buddy_view(request):
    """Render the Image Buddy interface."""
    return render(request, "image_buddy.html")


@csrf_exempt
def apply_filters(request):
    """
    Apply filters and transformations to a synthetic image and return SVG data.
    """
    if request.method == "POST":
        try:
            # Parse the POST data
            data = json.loads(request.body)
            brightness = float(data.get("brightness", 1.0))
            contrast = float(data.get("contrast", 1.0))
            filter_type = data.get("filterType", "none")

            # Generate a synthetic gradient image
            x = np.linspace(0, 1, 256)
            synthetic_image = np.outer(x, x)

            # Apply brightness and contrast adjustments
            transformed_image = synthetic_image * contrast + (brightness - 1)
            transformed_image = np.clip(transformed_image, 0, 1)  # Ensure pixel values are valid

            # Apply filters based on user selection
            if filter_type == "blur":
                transformed_image = gaussian_filter(transformed_image, sigma=3)
            elif filter_type == "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                transformed_image = convolve2d(transformed_image, kernel, mode="same", boundary="symm")
            elif filter_type == "edge_detect":
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                transformed_image = convolve2d(transformed_image, kernel, mode="same", boundary="symm")

            # Generate SVG visualization using matplotlib
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
            ax.imshow(transformed_image, cmap="gray", interpolation="nearest")
            ax.axis("off")
            output = io.StringIO()
            FigureCanvasSVG(fig).print_svg(output)

            return JsonResponse({"svg": output.getvalue()})

        except Exception as e:
            logger.error(f"Error in apply_filters: {e}")
            return JsonResponse({"error": f"An error occurred: {e}"}, status=500)

    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


@csrf_exempt
def initiate_login_view(request):
    """Render the initiate login template."""
    return render(request, 'initiate_login.html')


@csrf_exempt
def initiate_login(request):
    """Send a Twilio verification code to the phone number."""
    if request.method == "POST":
        data = json.loads(request.body)
        phone_number = data.get("phone_number")

        if not phone_number:
            return JsonResponse({"error": "Phone number is required."}, status=400)

        try:
            # Check if user exists
            user, created = User.objects.get_or_create(
                phone_number=phone_number,
                defaults={"username": phone_number, "is_active": True}
            )

            if created:
                # User was created (new user)
                user.set_unusable_password()  # Ensure password is empty for now
                user.save()

            # Send verification code via Twilio
            status = twilio_client.send_verification(phone_number)

            return JsonResponse({
                "status": "Verification code sent.",
                "new_user": created  # Indicate if a new user was registered
            }, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=400)


@csrf_exempt
def verify_login_view(request):
    """Render the verify login template."""
    return render(request, 'verify_login.html')


@csrf_exempt
def generate_batch_report(request):
    """
    Generate calculations and renders for all models, and return a summary report.
    """
    if request.method == "POST":
        try:
            # Load all models
            models_dir = os.path.join(settings.BASE_DIR, "moon-rocks")
            output_dir = os.path.join(settings.MEDIA_ROOT, "batch_outputs")
            os.makedirs(output_dir, exist_ok=True)

            models = []

            # Collect models
            for root, _, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".obj"):  # Only include .obj files
                        full_path = os.path.join(root, file)
                        models.append({"name": file, "path": full_path})

            report = []

            # Process each model
            for model in models:
                model_name = model["name"]
                model_path = model["path"]
                output_folder = os.path.join(output_dir, model_name.replace(".obj", ""))

                # Ensure output folder exists
                os.makedirs(output_folder, exist_ok=True)

                # Run Blender script for the current model
                try:
                    subprocess.run([
                        "blender", "--background", "--python", os.path.join(settings.BASE_DIR, "run_blender.py"),
                        "--", "8.44", "19.71", "290", "5.43", "1.25", model_path, output_folder
                    ], check=True)

                    # Correctly generate image URLs using MEDIA_URL
                    rendered_images = {
                        "left_camera": urljoin(settings.MEDIA_URL,
                                               f"batch_outputs/{model_name.replace('.obj', '')}/left_camera.png"),
                        "right_camera": urljoin(settings.MEDIA_URL,
                                                f"batch_outputs/{model_name.replace('.obj', '')}/right_camera.png"),
                        "disparity_map": urljoin(settings.MEDIA_URL,
                                                 f"batch_outputs/{model_name.replace('.obj', '')}/disparity_map.png"),
                        "depth_map": urljoin(settings.MEDIA_URL,
                                             f"batch_outputs/{model_name.replace('.obj', '')}/depth_map.png"),
                    }

                    # Generate summary data
                    summary = {
                        "model": model_name,
                        "images": rendered_images,
                        "status": "success"
                    }
                    report.append(summary)

                except subprocess.CalledProcessError as e:
                    report.append({
                        "model": model_name,
                        "status": "error",
                        "details": str(e)
                    })

            return JsonResponse({"status": "completed", "report": report})

        except Exception as e:
            return JsonResponse({"status": "error", "details": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def verify_login(request):
    """
    Verify the Twilio code and update the user's details if needed.
    """
    if request.method == "POST":
        data = json.loads(request.body)
        phone_number = data.get("phone_number")
        code = data.get("code")
        first_name = data.get("first_name")
        last_name = data.get("last_name")

        if not phone_number or not code:
            return JsonResponse({"error": "Phone number and verification code are required."}, status=400)

        # Verify the Twilio code
        status = twilio_client.check_verification(phone_number, code)
        if status == "approved":
            try:
                # Fetch the user by phone number
                user = User.objects.get(phone_number=phone_number)

                # Update user's name if provided
                if first_name:
                    user.first_name = first_name
                if last_name:
                    user.last_name = last_name

                # Mark the phone number as verified
                user.is_phone_verified = True
                user.save()

                backend = get_backends()[0]  # Use the first configured backend
                user.backend = f"{backend.__module__}.{backend.__class__.__name__}"

                # Log the user in
                login(request, user)
                return JsonResponse({
                    "status": "Verification successful. User logged in.",
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                }, status=200)
            except User.DoesNotExist:
                return JsonResponse({"error": "User does not exist."}, status=404)
        else:
            return JsonResponse({"error": "Invalid verification code."}, status=400)

    return JsonResponse({"error": "Invalid request method."}, status=400)


@csrf_exempt
def run_blender(request):
    if request.method == "POST":
        # Parse parameters from the frontend

        data = json.loads(request.body)
        selected_model = data.get("selectedModel", "")

        # Construct the full file path for the selected model
        model_path = os.path.join(settings.BASE_DIR, "moon-rocks", selected_model)

        # Validate file existence
        if not os.path.exists(model_path):
            return JsonResponse({"status": "error", "details": f"Model not found: {model_path}"}, status=404)

        # Paths for output
        output_folder = os.path.join(settings.MEDIA_ROOT, "render_outputs")
        os.makedirs(output_folder, exist_ok=True)

        try:
            # Run Blender script with arguments
            blender_script_path = os.path.join(os.getcwd(), "run_blender.py")
            subprocess.run(
                [
                    "blender", "--background", "--python", blender_script_path,
                    "--", str(data["sensorWidth"]), str(data["focalLength"]),
                    str(data["baseline"]), str(data["distance"]), str(data["toeInAngle"]),
                    model_path, output_folder
                ],
                check=True
            )

            return JsonResponse({"status": "success"})
        except subprocess.CalledProcessError as e:
            return JsonResponse({"status": "error", "details": str(e)}, status=500)
    else:
        return JsonResponse({"status": "error", "details": "Invalid request method."}, status=400)


def get_models(request):
    models_dir = os.path.join(settings.BASE_DIR, "moon-rocks")
    models = []

    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".obj"):  # Only include .obj files
                rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                models.append({"name": file, "path": rel_path})

    return JsonResponse(models, safe=False)


def home_view(request):
    # Fetch model objects from database
    people = People.objects.order_by('-role')
    research_areas = Research.objects.all().order_by('-effort')
    publications = Publication.objects.exclude(title__iexact='Generic')
    projects = Project.objects.all()
    assets = Asset.objects.all()
    funding_source = FundingSource.objects.all()
    
    # Pass data to the template (removed duplicate research_areas)
    context = {
        'people': people,
        'research_areas': research_areas,
        'publications': publications,
        'projects': projects,
        'assets': assets,
        'funding_source': funding_source,
    }
    return render(request, 'home.html', context)


def stereo_buddy_view(request):
    return render(request, 'stereo-buddy.html')


def cart_pole_buddy_view(request):
    return render(request, 'cart-pole-buddy.html')


def gaussian_processes_buddy_view(request):
    return render(request, 'gaussian-processes-buddy.html')


def param_estimation_buddy_view(request):
    return render(request, 'param-estimation-buddy.html')


def image_buddy_view(request):
    return render(request, 'image-buddy.html')


def slam_buddy_view(request):
    return render(request, 'slam-buddy.html')


def ses598_robotic_exploration_and_mapping_quiz(request):
    return render(request, 'ses-598-robotic-exploration-and-mapping-quiz.html')

def ransac_buddy(request):
    """
    View function for the RANSAC tutorial page.
    Demonstrates RANSAC algorithm for point cloud matching and outlier rejection.
    """
    return render(request, 'ransac-buddy.html') 

def multiview_geometry_view(request):
    """
    View function for the multi-view geometry tutorial page.
    Can include computed matrices as context if needed.
    """
    context = {
        'title': 'Multi-View Geometry Tutorial',
        # Add any computed matrices to the context if needed
        # 'essential_matrix': computed_essential_matrix,
        # 'fundamental_matrix': computed_fundamental_matrix,
        # 'camera_matrix': computed_camera_matrix,
    }
    return render(request, 'multiview-geometry.html', context)


def ses598_robotic_exploration_and_mapping(request):
    return render(request, 'SES598_robotic_exploration_and_mapping.html')

def particle_filter_buddy(request):
    return render(request, 'particle_filter_buddy.html')

def loop_closure_buddy(request):
    return render(request, 'loop_closure_buddy.html')

def sensor_fusion_buddy(request):
    return render(request, 'sensor_fusion_buddy.html')

def visual_odometry_buddy(request):
    return render(request, 'visual_odometry_buddy.html')

def point_cloud_buddy(request):
    return render(request, 'point_cloud_buddy.html')

def path_planning_buddy(request):
    """
    View function for the path planning tutorial page.
    Covers algorithms like A*, RRT, and potential fields with interactive demonstrations.
    """
    return render(request, 'path_planning_buddy.html')
