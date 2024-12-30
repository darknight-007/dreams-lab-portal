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
    """Render the SES598 quiz page with MCQs and interactive components"""
    # Create quiz components with enhanced features
    quiz_components = [
        {
            'id': 'stereo_challenge',
            'title': 'Stereo Vision Challenge',
            'description': 'Use the stereo vision widget to determine the depth of the marked point.',
            'widget_type': 'stereo_buddy',
            'difficulty': 'medium',
            'hint': 'Consider the relationship between disparity and depth. Remember that depth is inversely proportional to disparity.',
            'example': 'For a baseline of 20cm and focal length of 50mm, a disparity of 100 pixels corresponds to a depth of approximately 2 meters.',
            'parameters': {
                'baseline': 0.2,
                'focal_length': 50.0,
                'sensor_width': 36.0,
                'point_depth': 2.0,
                'noise_level': 0.02,
                'num_points': 10
            },
            'validation_rules': {
                'type': 'numeric',
                'tolerance': 0.1,
                'correctValue': 2.0
            }
        },
        {
            'id': 'ransac_challenge',
            'title': 'RANSAC Model Fitting',
            'description': 'Use RANSAC to fit a line to the noisy data and identify outliers.',
            'widget_type': 'ransac_buddy',
            'difficulty': 'hard',
            'hint': 'Try different threshold values. A good threshold should balance between including valid points and excluding outliers.',
            'example': 'With 100 points and 30% outliers, a threshold of 1.0 typically works well.',
            'parameters': {
                'num_points': 100,
                'outlier_ratio': 0.3,
                'noise_std': 0.2,
                'model_params': {'slope': 1.0, 'intercept': 0.0}
            },
            'validation_rules': {
                'type': 'numeric',
                'tolerance': 2,
                'correctValue': 30
            }
        },
        {
            'id': 'param_estimation',
            'title': 'Parameter Estimation Challenge',
            'description': 'Estimate the optimal learning rate for fastest convergence using the parameter estimation tool.',
            'widget_type': 'param_estimation_buddy',
            'difficulty': 'medium',
            'hint': 'A good learning rate should converge quickly without overshooting. Watch the convergence plot.',
            'example': 'If the error oscillates, the learning rate is too high. If it converges very slowly, the learning rate is too low.',
            'parameters': {
                'target_params': {'amplitude': 2.0, 'frequency': 0.5, 'phase': 0.785},
                'noise_level': 0.1,
                'num_points': 50
            },
            'validation_rules': {
                'type': 'numeric',
                'tolerance': 0.01,
                'correctValue': 0.1
            }
        }
    ]
    
    # MCQ answers and scoring
    mcq_answers = {
        'q1': '3',  # SLAM purpose
        'q2': '2',  # LiDAR
        'q3': '1',  # Occupancy grid
        'q4': '1',  # GPS challenge
        'q5': '2',  # Path planning
    }
    
    context = {
        'quiz_id': request.session.get('quiz_id', 'Not assigned'),
        'quiz_components': quiz_components,
        'total_questions': len(mcq_answers) + len(quiz_components),
        'show_results': False  # Will be True after submission
    }
    
    return render(request, 'ses598_rem_quiz.html', context)

def reset_quiz(request):
    if 'quiz_results' in request.session:
        del request.session['quiz_results']
    return redirect('ses598_quiz')

def ses598_course_view(request):
    syllabus = {
        'course_info': {
            'title': 'SES 598: Robotics and AI for Planetary Exploration',
            'semester': 'Spring 2024',
            'meeting_times': 'Tu/Th 3:00 PM - 4:15 PM',
            'location': 'ISTB4 401',
            'instructor': 'Dr. Jnaneshwar Das',
            'office_hours': 'By appointment',
            'contact': 'jdas@asu.edu'
        },
        'course_description': '''This course provides a comprehensive introduction to robotic exploration and mapping, 
        focusing on the fundamental concepts and advanced techniques in computer vision, SLAM (Simultaneous Localization and Mapping), 
        and autonomous navigation. Students will learn both theoretical foundations and practical implementations of robotic 
        perception systems, with emphasis on visual-inertial navigation, 3D reconstruction, and environment mapping. The course 
        includes hands-on experience with real robotic systems and modern software tools used in the industry.''',
        
        'prerequisites': [
            'Mathematics: Linear algebra (vectors, matrices, eigenvalues), calculus (derivatives, gradients), and probability theory (Bayes rule, distributions)',
            'Programming: Strong Python programming skills with experience in scientific computing libraries (NumPy, SciPy, PyTorch/TensorFlow)',
            'Computer Vision: Basic understanding of image processing, feature detection, and geometric transformations',
            'Computing Systems: Experience with Linux/Unix systems, version control (Git), and command-line tools',
            'Recommended: Prior exposure to ROS (Robot Operating System), CUDA programming, or parallel computing',
            'Required Software: Students must have a computer capable of running Linux and processing 3D graphics'
        ],
        
        'learning_outcomes': [
            'Master advanced robotics and AI techniques for exploration in challenging environments',
            'Design and implement autonomous systems for Earth and Space applications',
            'Develop expertise in 3D perception, mapping, and scene understanding',
            'Create intelligent decision-making systems for robotic exploration',
            'Build distributed multi-robot systems for collaborative missions',
            'Apply modern machine learning to real-world robotics problems'
        ],
        
        'grading': {
            'Projects': {
                'Project 1: 3D Reconstruction': '8%',
                'Project 2: Neural Scene Understanding': '8%',
                'Project 3: Digital Twin System': '8%',
                'Project 4: Multi-Robot Exploration': '8%',
                'Project 5: Extreme Environment Operations': '8%'
            },
            'Midterm Project': {
                'Information-Driven Mapping System': '20%'
            },
            'Final Project': {
                'End-to-End Space Robotics System': '25%'
            },
            'Class Participation': {
                'Interactive Tutorials': '5%',
                'Discussion & Code Reviews': '5%',
                'Project Presentations': '5%'
            }
        },
        
        'modules': [
            {
                'week': '1-2',
                'title': 'Computer Vision & 3D Reconstruction Fundamentals',
                'topics': [
                    'Image formation and camera models',
                    'Feature detection and matching',
                    'Epipolar geometry and stereo vision',
                    'Structure from Motion (SfM)',
                    'Multi-View Stereo (MVS)'
                ],
                'assignment': 'Project 1: 3D reconstruction pipeline'
            },
            {
                'week': '3-4',
                'title': 'Advanced Scene Representation & Neural Rendering',
                'topics': [
                    'Gaussian Splatting fundamentals',
                    'Neural Radiance Fields (NeRF)',
                    'View synthesis techniques',
                    'Real-time rendering strategies',
                    'Photogrammetry workflows'
                ],
                'assignment': 'Project 2: Neural scene representation and rendering'
            },
            {
                'week': '5-6',
                'title': 'Sampling Strategies & Information Theory',
                'topics': [
                    'Information theory fundamentals',
                    'Active sampling and exploration',
                    'Multi-armed bandits and Thompson sampling',
                    'Bayesian optimization for parameter tuning',
                    'Information gain in exploration'
                ],
                'assignment': 'Project 2: Optimal sampling strategy implementation'
            },
            {
                'week': '7-8',
                'title': 'Digital Twins & Online Learning',
                'topics': [
                    'Digital twin fundamentals',
                    'Online Bayesian learning',
                    'Adaptive exploration strategies',
                    'Sequential decision making',
                    'Real-time model updates'
                ],
                'assignment': 'Project 3: Adaptive digital twin system'
            },
            {
                'week': '9-10',
                'title': 'SLAM and Active Perception',
                'topics': [
                    'Information-theoretic SLAM',
                    'Active view selection',
                    'Uncertainty-aware mapping',
                    'Exploration-exploitation trade-offs',
                    'Resource-constrained planning'
                ],
                'assignment': 'Midterm Project: Information-driven mapping system'
            },
            {
                'week': '11-12',
                'title': 'Multi-Robot Coordination & Distributed Learning',
                'topics': [
                    'Distributed bandit algorithms',
                    'Multi-agent exploration strategies',
                    'Collaborative information gathering',
                    'Decentralized decision making',
                    'Communication-aware sampling'
                ],
                'assignment': 'Project 4: Multi-robot exploration system'
            },
            {
                'week': '13-14',
                'title': 'Extreme Environment Operations',
                'topics': [
                    'Risk-aware exploration',
                    'Robust sampling strategies',
                    'Adaptive resource allocation',
                    'Environmental uncertainty modeling',
                    'Safety-constrained learning'
                ],
                'assignment': 'Project 5: Robust exploration system'
            },
            {
                'week': '15-16',
                'title': 'Integration & Advanced Applications',
                'topics': [
                    'Meta-learning for exploration',
                    'Transfer learning in space applications',
                    'Lifelong learning systems',
                    'Integrated perception-planning-learning',
                    'Real-world deployment strategies'
                ],
                'assignment': 'Final Project: End-to-end space robotics system'
            }
        ]
    }
    
    return render(request, 'ses598_course.html', {'syllabus': syllabus})

# Define correct answers
correct_answers = {
    'q1': 'a',  # Perspective projection matrix
    'q2': 'b',  # Feature tracking for SfM
    'q3': 'a',  # SLAM chicken-and-egg problem
    'q4': 'b',  # Information gain in SLAM
    'q5': 'b',  # Bayesian optimization
    'q6': 'a',  # Thompson sampling
    'q7': 'b',  # Multi-modal sensing
    'q8': 'b',  # Uncertainty modeling
    'q9': 'a',  # Local vs global planning
    'q10': 'b'  # Distributed decision-making
}

def validate_interactive_answer(question_id, student_answer, correct_answer, tolerance):
    try:
        student_val = float(student_answer)
        correct_val = float(correct_answer)
        return abs(student_val - correct_val) <= tolerance
    except ValueError:
        return False

def process_quiz_submission(request):
    if request.method == 'POST':
        interactive_scores = []
        for component in quiz_questions['interactive_components']:
            answer_key = f"{component['id']}_answer"
            student_answer = request.POST.get(answer_key)
            is_correct = validate_interactive_answer(
                component['id'],
                student_answer,
                component['correct_answer'],
                component['tolerance']
            )
            interactive_scores.append(is_correct)

def tutorial_view(request, tutorial_type, tutorial_id):
    """View for rendering a tutorial component"""
    try:
        # Create tutorial component
        tutorial = tutorial_manager.create_tutorial(
            tutorial_type,
            tutorial_id,
            difficulty='medium'  # Could be passed as a parameter
        )
        
        # Get context for rendering
        context = tutorial.get_context()
        
        return render(request, 'interactive_component.html', context)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)

def quiz_view(request, quiz_type, quiz_id):
    """View for rendering a quiz component"""
    try:
        # Create quiz component
        quiz = tutorial_manager.create_quiz(
            quiz_type,
            quiz_id,
            difficulty=request.GET.get('difficulty', 'medium')
        )
        
        # Get context for rendering
        context = quiz.get_context(mode='quiz')
        
        return render(request, 'interactive_component.html', context)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)

def validate_quiz_answer(request, component_id):
    """Validate a quiz answer and return feedback"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        # Get the quiz component
        component = tutorial_manager.create_quiz(
            component_id.split('_')[0],  # Extract type from id
            component_id,
            difficulty=request.GET.get('difficulty', 'medium')
        )
        
        # Get student answer from POST data
        student_answer = request.POST.get('answer')
        if student_answer is None:
            return JsonResponse({'error': 'No answer provided'}, status=400)
        
        # Generate correct answer
        if hasattr(component, 'generate_ground_truth'):
            ground_truth = component.generate_ground_truth()
            correct_answer = ground_truth['depth']  # For stereo vision
        else:
            # For other types of quizzes, get correct answer differently
            correct_answer = component.get_correct_answer()
        
        # Validate the answer
        is_correct = tutorial_manager.validate_answer(
            component,
            student_answer,
            correct_answer
        )
        
        # Save progress
        if request.user.is_authenticated:
            QuizProgress.objects.update_or_create(
                user=request.user,
                component_id=component_id,
                defaults={'is_correct': is_correct}
            )
        
        # Return detailed feedback
        feedback = {
            'is_correct': is_correct,
            'correct_answer': correct_answer,
            'message': 'Correct! Well done!' if is_correct else 'Not quite right. Try again!',
            'hint': component.get_hint() if hasattr(component, 'get_hint') else None
        }
        
        return JsonResponse(feedback)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)

def get_tutorial_hints(request, component_id):
    """Get hints for a quiz component"""
    try:
        # Create tutorial component
        component = tutorial_manager.create_tutorial(
            component_id.split('_')[0],  # Extract type from id
            component_id
        )
        
        # Get hints
        hints = component.get_hints() if hasattr(component, 'get_hints') else []
        
        return JsonResponse({'hints': hints})
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)

def save_quiz_progress(request):
    """Save the user's quiz progress"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        # Get progress data from request
        data = json.loads(request.body)
        component_id = data.get('component_id')
        is_correct = data.get('is_correct')
        
        if not component_id or is_correct is None:
            return JsonResponse({'error': 'Missing required fields'}, status=400)
        
        # Get or create progress record
        progress, created = QuizProgress.objects.get_or_create(
            user=request.user,
            component_id=component_id,
            defaults={'is_correct': is_correct}
        )
        
        if not created:
            progress.is_correct = is_correct
            progress.save()
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def load_quiz_progress(request):
    """Load the user's quiz progress"""
    try:
        # Get all progress records for the user
        progress = QuizProgress.objects.filter(user=request.user)
        
        # Format progress data
        progress_data = {
            p.component_id: {
                'is_correct': p.is_correct,
                'timestamp': p.updated_at.isoformat()
            }
            for p in progress
        }
        
        return JsonResponse(progress_data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def widget_view(request, widget_type):
    """View for rendering interactive widgets"""
    # Map widget types to their corresponding views
    widget_map = {
        'stereo_buddy': stereo_buddy_view,
        'ransac_buddy': ransac_buddy,
        'param_estimation_buddy': param_estimation_buddy_view,
        'slam_buddy': slam_buddy_view,
        'cart_pole_buddy': cart_pole_buddy_view,
        'gaussian_processes_buddy': gaussian_processes_buddy_view,
        'image_buddy': image_buddy_view,
        'particle_filter_buddy': particle_filter_buddy,
        'loop_closure_buddy': loop_closure_buddy,
        'sensor_fusion_buddy': sensor_fusion_buddy,
        'visual_odometry_buddy': visual_odometry_buddy,
        'point_cloud_buddy': point_cloud_buddy,
        'path_planning_buddy': path_planning_buddy,
    }
    
    # Get the corresponding view function
    view_func = widget_map.get(widget_type)
    if not view_func:
        return JsonResponse({'error': f'Widget type {widget_type} not found'}, status=404)
    
    # Pass all query parameters to the view
    return view_func(request)

def deepgis_home(request):
    """New home page for deepgis.org"""
    return render(request, 'deepgis_home.html')

def dreamslab_home(request):
    """DREAMS Lab home page (previously the main home page)"""
    context = {
        'research_areas': Research.objects.all(),
        'people': People.objects.all(),
        'publications': Publication.objects.all(),
        'funding_source': FundingSource.objects.all(),
        'assets': Asset.objects.all()
    }
    return render(request, 'home.html', context)

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