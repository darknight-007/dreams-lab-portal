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
from .models import People, Research, Publication, Project, Asset, FundingSource
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
    # Clear any existing quiz results when first loading the page
    if request.method == 'GET':
        if 'quiz_results' in request.session:
            del request.session['quiz_results']
        return render(request, 'ses598_quiz.html', {'show_results': False})
    
    # Define correct answers
    correct_answers = {
        'q1': 'a', 'q2': 'b',  # Computer Vision
        'q3': 'a', 'q4': 'b',  # SLAM
        'q5': 'b', 'q6': 'a',  # Estimation
        'q7': 'b', 'q8': 'a',  # Sensing
        'q9': 'c', 'q10': 'a'  # Motion Planning
    }
    
    if request.method == 'POST':
        # Calculate section scores (each section has 2 questions worth 50 points each)
        cv_score = 0
        if request.POST.get('q1') == correct_answers['q1']: cv_score += 50
        if request.POST.get('q2') == correct_answers['q2']: cv_score += 50

        slam_score = 0
        if request.POST.get('q3') == correct_answers['q3']: slam_score += 50
        if request.POST.get('q4') == correct_answers['q4']: slam_score += 50

        estimation_score = 0
        if request.POST.get('q5') == correct_answers['q5']: estimation_score += 50
        if request.POST.get('q6') == correct_answers['q6']: estimation_score += 50

        sensing_score = 0
        if request.POST.get('q7') == correct_answers['q7']: sensing_score += 50
        if request.POST.get('q8') == correct_answers['q8']: sensing_score += 50

        motion_score = 0
        if request.POST.get('q9') == correct_answers['q9']: motion_score += 50
        if request.POST.get('q10') == correct_answers['q10']: motion_score += 50

        # Calculate total score (average of all sections)
        total_score = (cv_score + slam_score + estimation_score + sensing_score + motion_score) / 5

        # Store results in session
        quiz_results = {
            'show_results': True,
            'cv_score': cv_score,
            'slam_score': slam_score,
            'estimation_score': estimation_score,
            'sensing_score': sensing_score,
            'motion_score': motion_score,
            'total_score': total_score
        }
        request.session['quiz_results'] = quiz_results

        return render(request, 'ses598_quiz.html', quiz_results)
    
    return render(request, 'ses598_quiz.html', {'show_results': False})

def reset_quiz(request):
    if 'quiz_results' in request.session:
        del request.session['quiz_results']
    return redirect('ses598_quiz')

def ses598_course_view(request):
    syllabus = {
        'course_info': {
            'title': 'SES 598: AI and Robotics for Space Exploration',
            'semester': 'Spring 2025',
            'credits': 3,
            'location': 'ISTB4 401',
            'meeting_times': 'Tuesday/Thursday 3:00 PM - 4:15 PM',
            'instructor': 'Dr. Travis Marsh',
            'office_hours': 'Wednesday 2:00 PM - 4:00 PM or by appointment',
            'contact': 'travis.marsh@asu.edu'
        },
        'course_description': '''
            Advanced course focusing on AI and robotics applications in earth and space exploration. 
            Covers fundamental algorithms, state-of-the-art techniques, and practical implementations 
            for autonomous systems operating in extreme environments.
        ''',
        'prerequisites': [
            'Graduate standing in Engineering, Computer Science, or related field',
            'Programming experience (Python)',
            'Basic linear algebra and probability'
        ],
        'modules': [
            {
                'week': '1-2',
                'title': 'Fundamentals of Computer Vision in Space',
                'topics': [
                    'Image processing in extreme lighting conditions',
                    'Feature detection and matching',
                    'Stereo vision for depth estimation',
                    'Case study: Mars Rover vision systems'
                ],
                'assignment': 'Project 1: Implement robust feature detection for Mars surface images'
            },
            {
                'week': '3-4',
                'title': 'SLAM and Mapping',
                'topics': [
                    'Visual SLAM algorithms',
                    'Loop closure detection',
                    'Map representation and updates',
                    'Resource-constrained SLAM'
                ],
                'assignment': 'Project 2: Develop a lightweight SLAM system'
            },
            {
                'week': '5-6',
                'title': 'Terrain Classification & Navigation',
                'topics': [
                    'Soil/rock type identification',
                    'Traversability analysis',
                    'Hazard avoidance',
                    'Energy-efficient path planning'
                ],
                'assignment': 'Project 3: Terrain classification system'
            },
            {
                'week': '7-8',
                'title': 'Autonomous Sample Collection',
                'topics': [
                    'Target identification',
                    'Manipulation planning',
                    'Sample analysis and prioritization',
                    'Resource-constrained decision making'
                ],
                'assignment': 'Midterm Project: End-to-end sample collection system'
            },
            {
                'week': '9-10',
                'title': 'Multi-Robot Coordination',
                'topics': [
                    'Swarm robotics',
                    'Distributed task allocation',
                    'Communication constraints',
                    'Collaborative mapping'
                ],
                'assignment': 'Project 4: Multi-robot coordination simulation'
            },
            {
                'week': '11-12',
                'title': 'Extreme Environment Operations',
                'topics': [
                    'Thermal management',
                    'Radiation-hardened computing',
                    'Dust mitigation',
                    'Low-light operation'
                ],
                'assignment': 'Project 5: Environmental challenge solutions'
            },
            {
                'week': '13-14',
                'title': 'Resource-Aware Planning & Adaptive Learning',
                'topics': [
                    'Power management',
                    'Online and transfer learning',
                    'Uncertainty estimation',
                    'Fault detection and recovery'
                ],
                'assignment': 'Project 6: Adaptive learning system'
            },
            {
                'week': '15-16',
                'title': 'Space-Specific Challenges & Bio-Inspired Solutions',
                'topics': [
                    'Zero/low-gravity operations',
                    'Bio-mimetic locomotion',
                    'Time-delayed teleoperation',
                    'Orbital mechanics for navigation'
                ],
                'assignment': 'Final Project: Comprehensive space robotics challenge'
            }
        ],
        'grading': {
            'Projects (6)': '40%',
            'Midterm Project': '20%',
            'Final Project': '25%',
            'Class Participation': '15%'
        },
        'learning_outcomes': [
            'Design and implement computer vision systems for extreme environments',
            'Develop SLAM and navigation solutions for unknown terrains',
            'Create resource-aware planning algorithms for autonomous systems',
            'Implement multi-robot coordination strategies',
            'Solve challenges specific to space robotics applications'
        ]
    }
    
    return render(request, 'ses598_course.html', {'syllabus': syllabus})