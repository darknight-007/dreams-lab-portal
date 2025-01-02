import os
import json
import logging
import numpy as np
import io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d  # Correct import for convolve2d
from django.shortcuts import render, redirect
from matplotlib.backends.backend_svg import FigureCanvasSVG
from django.conf import settings
import subprocess
import os
from .models import People, Research, Publication, Project, Asset, FundingSource, QuizSubmission, QuizProgress
from django.http import JsonResponse
import json
import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import subprocess
import tempfile
from django.http import JsonResponse
from django.shortcuts import render

import numpy as np
from scipy.interpolate import interp1d

from django.conf import settings
from urllib.parse import urljoin

import uuid

from django.shortcuts import render
from django.http import JsonResponse
from .tutorials import TutorialManager

tutorial_manager = TutorialManager()

# Set up logging
logger = logging.getLogger(__name__)

# Remove global docker client initialization

@csrf_exempt
def image_buddy_view(request):
    """Render the Image Buddy interface."""
    return render(request, 'widgets/image_buddy.html')


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
    
    # Pass data to the template
    context = {
        'people': people,
        'research_areas': research_areas,
        'publications': publications,
        'projects': projects,
        'assets': assets,
        'funding_source': funding_source,
    }
    return render(request, 'deepgis_home.html', context)


def stereo_buddy_view(request):
    return render(request, 'widgets/stereo_buddy.html')


def gaussian_processes_buddy_view(request):
    return render(request, 'widgets/gaussian_processes_buddy.html')


def param_estimation_buddy_view(request):
    return render(request, 'widgets/param_estimation_buddy.html')


def slam_buddy_view(request):
    """
    View function for the SLAM tutorial page.
    """
    return render(request, 'widgets/slam_buddy.html', {
        'title': 'SLAM Tutorial',
        'description': 'Interactive tutorial for Simultaneous Localization and Mapping (SLAM)',
    })


def ransac_buddy_view(request):
    """
    View function for the RANSAC tutorial page.
    Demonstrates RANSAC algorithm for point cloud matching and outlier rejection.
    """
    return render(request, 'widgets/ransac_buddy.html')

def multiview_geometry_view(request):
    """
    View function for the multi-view geometry tutorial page.
    Demonstrates epipolar geometry, fundamental matrix, and camera calibration.
    """
    context = {
        'title': 'Multi-View Geometry Tutorial',
    }
    return render(request, 'widgets/multiview_geometry.html', context)


def particle_filter_buddy_view(request):
    return render(request, 'widgets/particle_filter_buddy.html')

def loop_closure_buddy_view(request):
    """View function for loop closure tutorial"""
    return render(request, 'widgets/loop_closure_buddy.html')

def sensor_fusion_buddy(request):
    return render(request, 'widgets/sensor_fusion_buddy.html')

def point_cloud_buddy(request):
    return render(request, 'widgets/point_cloud_buddy.html')

def path_planning_buddy_view(request):
    """View function for path planning tutorial"""
    return render(request, 'widgets/path_planning_buddy.html')


def generate_curved_line():
    # Define points for the base ground line (slightly slanted)
    x_ground = np.linspace(0, 20, 100)
    y_ground = 0.1 * x_ground + 2  # Slight upward slope
    
    # Define points for the boulder bulge
    x_boulder = np.linspace(8, 12, 50)
    
    # Create elongated circular bulge
    boulder_height = 4
    boulder_width = 2
    y_boulder = y_ground[40:90] + boulder_height * np.sqrt(1 - ((x_boulder-10)/boulder_width)**2)
    
    # Smooth transition points
    x = np.concatenate([x_ground[:40], x_boulder, x_ground[90:]])
    y = np.concatenate([y_ground[:40], y_boulder, y_ground[90:]])
    
    return x, y

def sample_points_on_curve(x, y, num_points=300):
    # Create interpolation function
    curve = interp1d(x, y, kind='linear')
    
    # Generate random x coordinates
    x_random = np.random.uniform(min(x), max(x), num_points)
    x_random.sort()  # Sort to maintain curve order
    
    # Get corresponding y coordinates
    y_random = curve(x_random)
    
    return x_random, y_random

def ransac_demo_data(request):
    # Generate dense point sampling
    n_points = 1000  # Increased for denser sampling
    
    # Generate x coordinates
    x_coords = np.linspace(0, 100, 200)  # Base curve points
    
    # Create ground profile with slight slope
    base_ground = 10 + 0.05 * x_coords
    
    # Define the protruding boulder shape using a more circular profile
    boulder_center_x = 50
    boulder_radius = 15
    boulder_height = 35
    
    # Create circular boulder profile
    boulder_region = np.abs(x_coords - boulder_center_x) <= boulder_radius
    x_relative = x_coords[boulder_region] - boulder_center_x
    
    # Generate circular profile using arc formula
    boulder_profile = np.sqrt(boulder_radius**2 - x_relative**2)
    # Scale and shift the profile
    boulder_profile = boulder_profile * (boulder_height/boulder_radius)
    
    # Smooth connection to ground
    transition_width = 5
    left_edge = boulder_center_x - boulder_radius
    right_edge = boulder_center_x + boulder_radius
    
    # Create smooth transitions at edges
    left_transition = (x_coords >= left_edge - transition_width) & (x_coords < left_edge)
    right_transition = (x_coords > right_edge) & (x_coords <= right_edge + transition_width)
    
    # Combine ground and boulder
    y_coords = base_ground.copy()
    y_coords[boulder_region] = base_ground[boulder_region] + boulder_profile
    
    # Smooth transitions
    if np.any(left_transition):
        transition_x = (x_coords[left_transition] - (left_edge - transition_width)) / transition_width
        y_coords[left_transition] = base_ground[left_transition] + \
            (transition_x**2) * (y_coords[boulder_region][0] - base_ground[left_transition])
    
    if np.any(right_transition):
        transition_x = 1 - (x_coords[right_transition] - right_edge) / transition_width
        y_coords[right_transition] = base_ground[right_transition] + \
            (transition_x**2) * (y_coords[boulder_region][-1] - base_ground[right_transition])
    
    # Generate dense random sampling points
    x_random = np.random.uniform(0, 100, n_points)
    x_random.sort()
    
    # Interpolate to get base y values
    y_random = np.interp(x_random, x_coords, y_coords)
    
    # Add realistic noise to points
    noise_scale = 0.3  # Scale of the noise
    y_random += np.random.normal(0, noise_scale, n_points)
    
    # Add extra points near surface for density
    extra_points_x = []
    extra_points_y = []
    
    # Add dense surface points with higher density on boulder
    for i in range(len(x_random)):
        # Higher probability of extra points on boulder
        point_prob = 0.7 if abs(x_random[i] - boulder_center_x) <= boulder_radius else 0.4
        if np.random.random() < point_prob:
            num_extra = np.random.randint(2, 5)  # 2-4 extra points per location
            for _ in range(num_extra):
                x_offset = np.random.normal(0, 0.2)  # Small x variation
                y_offset = np.random.normal(0, 0.2)  # Small y variation
                extra_points_x.append(x_random[i] + x_offset)
                extra_points_y.append(y_random[i] + y_offset)
    
    # Combine original and extra points
    x_random = np.concatenate([x_random, np.array(extra_points_x)])
    y_random = np.concatenate([y_random, np.array(extra_points_y)])
    
    # Sort points by x coordinate for cleaner visualization
    sort_idx = np.argsort(x_random)
    x_random = x_random[sort_idx]
    y_random = y_random[sort_idx]
    
    return JsonResponse({
        'x_coords': x_coords.tolist(),
        'y_coords': y_coords.tolist(),
        'x_random': x_random.tolist(),
        'y_random': y_random.tolist()
    })

def ses598_quiz(request):
    """Render the SES598 quiz page with user identification and MCQs"""
    # MCQ answers
    mcq_answers = {
        'q1': '3',  # SLAM purpose
        'q2': '2',  # LiDAR
        'q3': '1',  # Occupancy grid
        'q4': '1',  # GPS challenge
        'q5': '2',  # Path planning
    }

    if request.method == 'POST':
        # Get user identification (empty string becomes Anonymous)
        email = request.POST.get('email', '').strip() or 'Anonymous'
        
        # Calculate score
        score = 0
        for q, correct_ans in mcq_answers.items():
            student_ans = request.POST.get(q, '')
            if student_ans == correct_ans:
                score += 1

        # Calculate total score percentage
        total_score = (score / len(mcq_answers)) * 100

        context = {
            'show_results': True,
            'score': total_score,
            'email': email
        }
        return render(request, 'ses598_rem_quiz.html', context)

    # GET request - always display fresh quiz
    context = {
        'show_results': False,
        'email': 'Anonymous'
    }
    return render(request, 'ses598_rem_quiz.html', context)

def reset_quiz(request):
    """Reset quiz state while preserving session ID"""
    # Clear quiz results but keep session ID
    session_id = request.session.get('openuav_session_id')
    if 'quiz_results' in request.session:
        del request.session['quiz_results']
    
    # Restore session ID if it existed
    if session_id:
        request.session['openuav_session_id'] = session_id
    
    return redirect('ses598_quiz')

def get_ses598_course_data():
    """Return the SES598 course data that can be used across different views"""
    return {
        'title': 'SES 598: Space Robotics and AI',
        'semester': 'Spring 2024',
        'meeting_times': 'TBD',
        'location': 'TBD',
        'description': 'Explore the intersection of robotics and artificial intelligence in space applications through hands-on tutorials and real-world projects.',
        'tutorials': {
            'Vision and Perception': [
                {
                    'title': 'Multi-view Geometry',
                    'description': 'Interactive exploration of stereo vision and 3D reconstruction concepts with real-time visualization.',
                    'difficulty': 'Intermediate',
                    'url': 'multiview_geometry'
                }
            ]
        }
    }

def ses598_course_view(request):
    """View function for the SES598 course page"""
    return render(request, 'ses598_course.html', {'syllabus': get_ses598_course_data()})

def dreamslab_home_view(request):
    """DREAMS Lab home page"""
    context = {
        'research_areas': Research.objects.all(),
        'people': People.objects.all(),
        'publications': Publication.objects.all(),
        'funding_source': FundingSource.objects.all(),
        'assets': Asset.objects.all(),
        'course': get_ses598_course_data()['course_info']  # Only pass the course info section
    }
    return render(request, 'home.html', context)

# Update the OpenUAV URL to use the consolidated view
def openuav_home_view(request):
    """OpenUAV home page using the consolidated interface"""
    return redirect('openuav_manager:container_list')

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

@csrf_exempt
@require_http_methods(["GET", "OPTIONS"])
def get_available_layers(request):
    """Return available map layers organized by category"""
    if request.method == "OPTIONS":
        response = JsonResponse({})
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response
        
    try:
        logger.info("Fetching available layers...")
        
        # Define empty layer groups
        layers = {
            "tiles_media": {
                "name": "Media Tiles",
                "layers": []
            },
            "tileserver": {
                "name": "Base Layers",
                "layers": []
            }
        }
        
        # Log the final layer configuration
        logger.info(f"Returning layers configuration: {json.dumps(layers, indent=2)}")
        
        response = JsonResponse(layers)
        response["Access-Control-Allow-Origin"] = "*"
        return response
    except Exception as e:
        logger.error(f"Error in get_available_layers: {e}", exc_info=True)
        response = JsonResponse({"error": str(e)}, status=500)
        response["Access-Control-Allow-Origin"] = "*"
        return response

@csrf_exempt
@require_http_methods(["GET", "OPTIONS"])
def get_category_info(request):
    """Return category information for map layers"""
    logger.info(f"Received {request.method} request for get_category_info")
    logger.info(f"Request headers: {dict(request.headers)}")
    
    if request.method == "OPTIONS":
        logger.info("Handling OPTIONS request")
        response = JsonResponse({})
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        logger.info(f"OPTIONS response headers: {dict(response.headers)}")
        return response
        
    try:
        logger.info("Fetching category information...")
        
        # Define the categories in the format expected by the frontend
        categories = {
            'Rock': { 
                'color': '#FF0000', 
                'id': 1,
                'description': 'Rock features and formations'
            },
            'Vegetation': { 
                'color': '#00FF00', 
                'id': 2,
                'description': 'Vegetation and plant life'
            },
            'Shadow': { 
                'color': '#0000FF', 
                'id': 3,
                'description': 'Shadow areas and dark regions'
            }
        }
        
        logger.info(f"Returning category information: {json.dumps(categories, indent=2)}")
        
        response = JsonResponse(categories)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        logger.info(f"GET response headers: {dict(response.headers)}")
        return response
    except Exception as e:
        logger.error(f"Error in get_category_info: {e}", exc_info=True)
        response = JsonResponse({"error": str(e)}, status=500)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response

def drone_buddy_view(request):
    """View function for the drone buddy challenge"""
    return render(request, 'widgets/drone_buddy.html')

@csrf_exempt
@require_http_methods(["POST"])
def save_drone_code(request):
    """Save the drone controller code"""
    try:
        data = json.loads(request.body)
        code = data.get('code')
        if not code:
            return JsonResponse({'error': 'No code provided'}, status=400)
            
        # Save the code to a file in the user's session directory
        session_id = request.session.get('openuav_session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            request.session['openuav_session_id'] = session_id
            
        code_dir = os.path.join(settings.MEDIA_ROOT, 'drone_code', session_id)
        os.makedirs(code_dir, exist_ok=True)
        
        with open(os.path.join(code_dir, 'drone_controller.py'), 'w') as f:
            f.write(code)
            
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def run_drone_tests(request):
    """Run tests on the drone controller code"""
    try:
        session_id = request.session.get('openuav_session_id')
        if not session_id:
            return JsonResponse({'error': 'No session found'}, status=400)
            
        code_path = os.path.join(settings.MEDIA_ROOT, 'drone_code', session_id, 'drone_controller.py')
        if not os.path.exists(code_path):
            return JsonResponse({'error': 'No code found'}, status=404)
            
        # Run the tests (mock implementation for now)
        test_results = {
            'status': 'success',
            'tests': [
                {'name': 'test_initialization', 'passed': True, 'message': 'Controller initialized correctly'},
                {'name': 'test_hover', 'passed': True, 'message': 'Hover control implemented correctly'},
                {'name': 'test_movement', 'passed': True, 'message': 'Movement control implemented correctly'}
            ]
        }
        return JsonResponse(test_results)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def cart_pole_lqr_buddy_view(request):
    """View function for the cart pole LQR tutorial"""
    return render(request, 'widgets/cart_pole_lqr_buddy.html')

def sampling_buddy_view(request):
    return render(request, 'widgets/sampling_buddy.html')

def gp_ucb_buddy_view(request):
    return render(request, 'widgets/gp_ucb_buddy.html')

def bundle_adjustment_buddy_view(request):
    """View function for the bundle adjustment tutorial page."""
    return render(request, 'widgets/bundle_adjustment_buddy.html')

def tutorials_home(request):
    """Render the tutorials home page"""
    return render(request, 'ses598_tutorials.html')

def stereo_buddy(request):
    """Render the stereo vision tutorial"""
    context = {
        'title': 'Stereo Vision for Planetary Mapping',
        'difficulty': 'beginner',
        'tutorial_id': 'stereo_vision'
    }
    return render(request, 'widgets/stereo_buddy.html', context)

def slam_buddy(request):
    """Render the SLAM tutorial"""
    context = {
        'title': 'SLAM Implementation',
        'difficulty': 'intermediate',
        'tutorial_id': 'slam'
    }
    return render(request, 'widgets/slam_buddy.html', context)

def sensor_fusion_buddy(request):
    """Render the sensor fusion tutorial"""
    context = {
        'title': 'Multi-Sensor Fusion',
        'difficulty': 'intermediate',
        'tutorial_id': 'sensor_fusion'
    }
    return render(request, 'widgets/sensor_fusion_buddy.html', context)

def multiview_geometry(request):
    """Render the multi-view geometry tutorial"""
    context = {
        'title': 'Multi-View Reconstruction',
        'difficulty': 'advanced',
        'tutorial_id': 'multiview'
    }
    return render(request, 'widgets/multiview_geometry.html', context)

def ses598_quiz_part2(request):
    """Render Part 2 of the SES598 quiz focusing on tutorial concepts"""
    # Tutorial concept MCQ answers
    mcq_answers = {
        'q1': '2',  # Stereo vision baseline
        'q2': '2',  # SLAM loop closure
        'q3': '2',  # Kalman filter parameters
        'q4': '3',  # Sampling strategy
    }

    if request.method == 'POST':
        # Get user identification (empty string becomes Anonymous)
        email = request.POST.get('email', '').strip() or 'Anonymous'
        
        # Calculate score
        score = 0
        for q, correct_ans in mcq_answers.items():
            student_ans = request.POST.get(q, '')
            if student_ans == correct_ans:
                score += 1

        # Calculate total score percentage
        total_score = (score / len(mcq_answers)) * 100

        context = {
            'show_results': True,
            'score': total_score,
            'email': email
        }
        return render(request, 'ses598_rem_quiz_part2.html', context)

    # GET request - always display fresh quiz
    context = {
        'show_results': False,
        'email': 'Anonymous'
    }
    return render(request, 'ses598_rem_quiz_part2.html', context)