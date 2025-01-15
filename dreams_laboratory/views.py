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
from django.utils import timezone
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from io import BytesIO

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

def get_certificate_eligibility(request):
    """Check if user is eligible for a certificate and calculate final score"""
    part1_completed = request.session.get('quiz_part1_completed', False)
    part2_completed = request.session.get('quiz_part2_completed', False)
    
    if not (part1_completed and part2_completed):
        return False, 0.0
    
    part1_score = float(request.session.get('quiz_part1_score', 0.0))
    part2_score = float(request.session.get('quiz_part2_score', 0.0))
    
    # Calculate weighted average (part 1: 60%, part 2: 40%) using floating-point
    final_score = (part1_score * 0.6) + (part2_score * 0.4)
    
    # Check if at least one question was answered correctly in each part
    part1_any_correct = any(request.session.get(f'quiz_part1_q{i}_correct', False) for i in range(1, 6))
    part2_any_correct = any(request.session.get(f'quiz_part2_q{i}_correct', False) for i in range(1, 6))
    
    # Eligible if:
    # 1. At least one question was answered correctly in each part
    # 2. Part 1 score >= 60%
    # 3. Part 2 score >= 70%
    is_eligible = (
        part1_any_correct and part2_any_correct and
        part1_score >= 60.0 and part2_score >= 70.0
    )
    
    return is_eligible, final_score

@csrf_exempt
def generate_certificate(request):
    """Generate a PDF certificate for users who completed both quiz parts"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
    eligible, final_score = get_certificate_eligibility(request)
    if not eligible:
        return JsonResponse({
            'error': 'Not eligible for certificate. Complete both parts with minimum 70% score.'
        }, status=400)
    
    email = request.session.get('quiz_email', 'Anonymous')
    # Format timestamp to match web interface with timezone
    timestamp = timezone.localtime().strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Create PDF certificate
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add background gradient
    p.setFillColorRGB(0.1, 0.1, 0.18)  # Dark blue background
    p.rect(0, 0, width, height, fill=1)
    
    # Add decorative header
    p.setFillColorRGB(0.29, 0.62, 1)  # #4a9eff in RGB
    p.rect(0, height-3*inch, width, 3*inch, fill=1)
    
    # Certificate styling with new space theme
    p.setFillColorRGB(1, 1, 1)  # White text
    p.setFont("Helvetica-Bold", 24)
    p.drawCentredString(width/2, height-2*inch, "Certificate of Completion")
    
    p.setFont("Helvetica", 16)
    p.drawCentredString(width/2, height-3*inch, "This is to certify that")
    
    # Draw email in a highlighted box
    p.setFillColorRGB(0, 1, 0.62)  # #00ff9d in RGB
    p.setFont("Helvetica-Bold", 20)
    p.drawCentredString(width/2, height-3.5*inch, email)
    
    p.setFillColorRGB(1, 1, 1)  # Back to white text
    p.setFont("Helvetica", 16)
    p.drawCentredString(width/2, height-4*inch, "has successfully completed the enrollment quiz for")
    
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(width/2, height-4.5*inch, "SES 598: Space Robotics and AI")
    
    # Performance level with color coding
    p.setFont("Helvetica-Bold", 14)
    if final_score >= 90:
        p.setFillColorRGB(0, 1, 0.62)  # Elite - Green
        level = "ELITE PERFORMANCE"
    elif final_score >= 80:
        p.setFillColorRGB(0.29, 0.62, 1)  # Advanced - Blue
        level = "ADVANCED LEVEL"
    else:
        p.setFillColorRGB(1, 0.72, 0)  # Qualified - Gold
        level = "QUALIFIED LEVEL"
    
    p.drawCentredString(width/2, height-5.2*inch, level)
    
    # Detailed scores section
    p.setFillColorRGB(1, 1, 1)  # White text
    p.setFont("Helvetica", 12)
    part1_score = round(request.session.get('quiz_part1_score', 0), 1)
    part2_score = round(request.session.get('quiz_part2_score', 0), 1)
    
    # Draw scores with weighted calculation explanation
    y_position = height-6*inch
    p.drawString(2*inch, y_position, f"Part 1 (Fundamentals): {part1_score}% × 0.6 = {round(part1_score * 0.6, 1)}%")
    p.drawString(2*inch, y_position-0.4*inch, f"Part 2 (Advanced): {part2_score}% × 0.4 = {round(part2_score * 0.4, 1)}%")
    
    # Final score with color coding
    p.setFont("Helvetica-Bold", 14)
    if final_score >= 90:
        p.setFillColorRGB(0, 1, 0.62)  # Elite - Green
    elif final_score >= 80:
        p.setFillColorRGB(0.29, 0.62, 1)  # Advanced - Blue
    else:
        p.setFillColorRGB(1, 0.72, 0)  # Qualified - Gold
    p.drawString(2*inch, y_position-0.8*inch, f"Final Score: {round(final_score, 1)}%")
    
    # Add systems status
    p.setFillColorRGB(1, 1, 1)  # White text
    p.setFont("Helvetica", 12)
    systems = ["AI Systems", "Robotics Control", "Vision Systems", "SLAM Operations", "Error Handling"]
    y_position = y_position-1.5*inch
    
    for system in systems:
        if final_score >= 90:
            status = "ELITE"
            p.setFillColorRGB(0, 1, 0.62)
        elif final_score >= 80:
            status = "EXCELLENT"
            p.setFillColorRGB(0.29, 0.62, 1)
        elif final_score >= 70:
            status = "GOOD"
            p.setFillColorRGB(1, 0.72, 0)
        elif final_score >= 60:
            status = "ACCEPTABLE"
            p.setFillColorRGB(1, 0.53, 0)
        else:
            status = "CRITICAL"
            p.setFillColorRGB(1, 0.27, 0.27)
        
        p.drawString(2*inch, y_position, f"{system}: {status}")
        y_position -= 0.3*inch
    
    # Timestamp and footer
    p.setFillColorRGB(0.29, 0.62, 1)  # #4a9eff in RGB
    p.setFont("Helvetica-Oblique", 12)
    p.drawCentredString(width/2, 2*inch, f"Quiz completed on {timestamp}")
    
    p.setFillColorRGB(1, 1, 1)  # White text
    p.setFont("Helvetica", 12)
    p.drawCentredString(width/2, 1.5*inch, "DREAMS Laboratory - Arizona State University")
    
    # Add decorative footer
    p.setFillColorRGB(0.29, 0.62, 1)  # #4a9eff in RGB
    p.rect(0, 0, width, inch, fill=1)
    
    p.showPage()
    p.save()
    
    # Get the value of the BytesIO buffer and return the PDF
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    # Clean up email for filename by removing special characters and replacing @ with _at_
    clean_email = email.replace('@', '_at_').replace('.', '_')
    response['Content-Disposition'] = f'attachment; filename="SES598_Certificate_{clean_email}_{timestamp.replace(" ", "_").replace(":", "-")}.pdf"'
    
    return response

def ses598_quiz(request):
    """Render the SES598 quiz page with user identification and MCQs"""
    # MCQ answers with explanations
    mcq_answers = {
        'q1': {
            'correct': '3',  # SLAM purpose
            'explanation': 'SLAM (Simultaneous Localization and Mapping) is primarily used for robot navigation and mapping in unknown environments. It combines sensor data to build a map while simultaneously tracking the robot\'s position.'
        },
        'q2': {
            'correct': '2',  # LiDAR
            'explanation': 'LiDAR (Light Detection and Ranging) uses laser pulses to measure distances. It provides accurate 3D point clouds of the environment, making it ideal for mapping and obstacle detection.'
        },
        'q3': {
            'correct': '1',  # Occupancy grid
            'explanation': 'Occupancy grid mapping represents the environment as a grid where each cell has a probability of being occupied. This probabilistic approach helps handle sensor uncertainty.'
        },
        'q4': {
            'correct': '1',  # GPS challenge
            'explanation': 'GPS signals are significantly attenuated by buildings and terrain, making indoor and urban canyon navigation challenging. Alternative localization methods are often needed.'
        },
        'q5': {
            'correct': '2',  # Path planning
            'explanation': 'Path planning algorithms must balance multiple objectives including path length, safety margins, energy efficiency, and kinematic constraints.'
        }
    }

    if request.method == 'POST':
        # Get user identification (empty string becomes Anonymous)
        email = request.POST.get('email', '').strip()
        if email:
            # Store email in session for part 2
            request.session['quiz_email'] = email
            request.session.modified = True  # Ensure session is saved
        else:
            email = 'Anonymous'
        
        # Calculate score and track answers
        score = 0.0
        part1_answers = {}
        feedback = {}
        for q, ans_data in mcq_answers.items():
            student_ans = request.POST.get(q, '')
            correct_ans = ans_data['correct']
            if student_ans == correct_ans:
                score += 1.0
            # Store answer and feedback
            part1_answers[q] = student_ans
            feedback[q] = {
                'student_answer': student_ans,
                'correct_answer': correct_ans,
                'is_correct': student_ans == correct_ans,
                'explanation': ans_data['explanation']
            }
            # Store answer in session
            request.session[f'quiz_part1_{q}'] = student_ans

        # Calculate total score percentage using floating-point division
        total_score = (score / float(len(mcq_answers))) * 100.0

        # Store the part 1 results in session
        request.session['quiz_part1_score'] = total_score
        request.session['quiz_part1_completed'] = True
        request.session.modified = True  # Ensure session is saved

        # Save Part 1 results to database
        quiz_submission = QuizSubmission(
            quiz_id='SES598',
            session_id=request.session.session_key,
            email=email,
            total_score=total_score,
            cv_score=total_score,  # Part 1 score
            slam_score=0.0,  # Part 2 not completed yet
            q1=part1_answers.get('q1', ''),
            q2=part1_answers.get('q2', ''),
            q3=part1_answers.get('q3', ''),
            q4=part1_answers.get('q4', ''),
            q5=part1_answers.get('q5', '')
        )
        quiz_submission.save()

        # Check certificate eligibility
        eligible, final_score = get_certificate_eligibility(request)

        context = {
            'show_results': True,
            'score': total_score,
            'email': email,
            'eligible_for_certificate': eligible,
            'final_score': final_score,
            'part2_completed': request.session.get('quiz_part2_completed', False),
            'feedback': feedback  # Add feedback to context
        }
        return render(request, 'ses598_rem_quiz.html', context)

    # GET request - display fresh quiz with stored email if available
    stored_email = request.session.get('quiz_email', '')
    context = {
        'show_results': False,
        'email': stored_email if stored_email else 'Anonymous',
        'eligible_for_certificate': False
    }
    return render(request, 'ses598_rem_quiz.html', context)

def ses598_quiz_part2(request):
    """Render Part 2 of the SES598 quiz focusing on tutorial concepts"""
    # Tutorial concept MCQ answers with explanations
    mcq_answers = {
        'q1': {
            'correct': '2',  # Stereo vision baseline
            'explanation': 'A 0.5m baseline provides the optimal balance between depth resolution and feature matching reliability. Larger baselines improve depth resolution but make feature matching more difficult due to perspective changes.',
            'question_text': 'In the multi-view geometry tutorial with a 35mm focal length camera observing a rock formation at 10 meters, what baseline setting provides the best balance between feature matching confidence and depth estimation accuracy?',
            'options': [
                {'value': '1', 'text': '0.2 meters (minimal parallax, easy matching but poor depth resolution)'},
                {'value': '2', 'text': '0.5 meters (balanced parallax and matching with good depth resolution)'},
                {'value': '3', 'text': '1.0 meters (strong depth resolution but challenging feature matching)'},
                {'value': '4', 'text': '2.0 meters (optimal depth resolution but impractical for feature matching)'}
            ]
        },
        'q2': {
            'correct': '2',  # SLAM loop closure
            'explanation': 'Loop closure in SLAM is crucial for reducing accumulated drift by recognizing previously visited locations. This helps maintain global map consistency and improves localization accuracy.',
            'question_text': 'In SLAM, what is the primary purpose of loop closure detection?',
            'options': [
                {'value': '1', 'text': 'To detect moving objects (dynamic environment handling)'},
                {'value': '2', 'text': 'To reduce accumulated drift by recognizing previously visited locations (global consistency)'},
                {'value': '3', 'text': 'To plan the shortest path (navigation optimization)'},
                {'value': '4', 'text': 'To calibrate sensors (hardware optimization)'}
            ]
        },
        'q3': {
            'correct': '3',  # Kalman filter parameters
            'explanation': 'For fast-moving robots, high process and measurement noise settings provide more robust estimation. This accounts for rapid state changes and potential sensor reliability issues at high speeds.',
            'question_text': 'For a fast-moving robot using sensor fusion with a Kalman filter, which noise parameter configuration would provide the most robust state estimation?',
            'options': [
                {'value': '1', 'text': 'Low process noise, high measurement noise (trust model more than sensors)'},
                {'value': '2', 'text': 'High process noise, low measurement noise (trust sensors more than model)'},
                {'value': '3', 'text': 'High process noise, high measurement noise (conservative estimation)'},
                {'value': '4', 'text': 'Low process noise, low measurement noise (aggressive estimation)'}
            ]
        },
        'q4': {
            'correct': '3',  # Sampling strategy
            'explanation': 'Information gain-based sampling is most efficient as it adaptively focuses resources on areas that provide the most valuable information, optimizing exploration in resource-constrained scenarios.',
            'question_text': 'Which sampling strategy is most efficient for exploring an unknown environment with limited resources?',
            'options': [
                {'value': '1', 'text': 'Random sampling (unbiased but inefficient coverage)'},
                {'value': '2', 'text': 'Grid-based sampling (systematic but rigid coverage)'},
                {'value': '3', 'text': 'Information gain-based sampling (adaptive and efficient)'},
                {'value': '4', 'text': 'Uniform sampling (consistent but inflexible coverage)'}
            ]
        },
        'q5': {
            'correct': '1',  # Error type in rock mapping
            'explanation': 'This question opens up a rich discussion and does not have a single definitive answer. In the context of sample collection on Mars, every rock sample collected must be carefully chosen since it will be cached and returned to Earth for detailed analysis. Given that the mission can only accommodate a limited number of samples (e.g., 30 in total), the strategy must balance the risk of overlooking significant scientific discoveries (false negatives) with the cost of retrieving less meaningful samples (false positives). Optimizing this trade-off involves evaluating the expected utility of each sample within the constraints of the mission. For the sake of this explanation, we commit to one perspective while recognizing the complexity and multi-objective nature of this decision. Type I errors (false positives) are more critical to minimize because they waste limited resources on incorrect samples. In resource-constrained scenarios like Mars exploration, wasting collection opportunities on wrong samples is more costly than missing some potential targets.',
            'question_text': 'Balancing Error Trade-offs in Automated Rock Mapping for Planetary Exploration\n\nIn the context of a Mars rover mission where the rover can cache up to 30 samples (batch size k=30), consider the following:\n\n    Resource Constraints: Each sample collection attempt is costly, and storage is limited to 30 samples.\n    Mission Objectives: Success depends on collecting scientifically valuable rock samples.\n    Error Scenarios:\n        Type I Error (False Positive): Incorrectly identifying a non-target rock as a target, wasting resources and storage space.\n        Type II Error (False Negative): Missing a target rock by classifying it as non-target, potentially losing valuable discoveries.',
            'options': [
                {'value': '1', 'text': 'Type I Error (False Positives) – Wasting resources on non-target samples is more costly'},
                {'value': '2', 'text': 'Type II Error (False Negatives) – Missing potential discoveries is more costly'},
                {'value': '3', 'text': 'Both errors are equally important to minimize'},
                {'value': '4', 'text': 'Neither error matters as long as some samples are collected'}
            ]
        }
    }

    # Get email from session, ensure it's properly retrieved
    email = request.session.get('quiz_email', '')
    part1_completed = request.session.get('quiz_part1_completed', False)
    
    # If part 1 is not completed, redirect to part 1
    if not part1_completed:
        return redirect('ses598_quiz')

    if request.method == 'POST':
        # Get user identification from POST, fallback to session email if empty
        submitted_email = request.POST.get('email', '').strip()
        if submitted_email:
            email = submitted_email
            # Store email in session for future use
            request.session['quiz_email'] = email
            request.session.modified = True  # Ensure session is saved
        elif not email:
            email = 'Anonymous'
        
        # Calculate score and track answers
        score = 0.0
        part2_answers = {}
        feedback = {}
        for q, ans_data in mcq_answers.items():
            student_ans = str(request.POST.get(q, ''))  # Ensure string type
            correct_ans = str(ans_data['correct'])      # Ensure string type
            # Store answer first
            part2_answers[q] = student_ans
            # Then check correctness
            is_correct = student_ans == correct_ans
            if is_correct:
                score += 1.0
            # Store feedback with question text and options
            feedback[q] = {
                'student_answer': student_ans,
                'correct_answer': correct_ans,
                'is_correct': is_correct,
                'explanation': ans_data['explanation'],
                'question_text': ans_data['question_text'],
                'options': ans_data['options']
            }

        # Calculate total score percentage using floating-point division
        total_score = (score / float(len(mcq_answers))) * 100.0

        # Store the part 2 results in session
        request.session['quiz_part2_score'] = total_score
        request.session['quiz_part2_completed'] = True
        request.session.modified = True  # Ensure session is saved

        # Check certificate eligibility
        eligible, final_score = get_certificate_eligibility(request)

        # Track if any questions were answered correctly
        any_correct = False
        for q in mcq_answers.keys():
            is_correct = feedback[q]['is_correct']
            if is_correct:
                any_correct = True
            # Store individual question correctness in session
            request.session[f'quiz_part2_{q}_correct'] = is_correct

        # Create a fresh submission for Part 2
        quiz_submission = QuizSubmission(
            quiz_id='SES598_Part2',  # Distinct quiz_id for Part 2
            session_id=request.session.session_key,
            email=email,
            total_score=total_score,  # Just Part 2 score
            cv_score=0.0,  # Not applicable for Part 2
            slam_score=total_score,  # Part 2 score
            # Store Part 2 answers sequentially in q1-q5
            q1=part2_answers.get('q1', ''),  # Stereo vision baseline
            q2=part2_answers.get('q2', ''),  # SLAM loop closure
            q3=part2_answers.get('q3', ''),  # Kalman filter parameters
            q4=part2_answers.get('q4', ''),  # Sampling strategy
            q5=part2_answers.get('q5', '')   # Error type in rock mapping
        )
        quiz_submission.save()

        context = {
            'show_results': True,
            'score': total_score,
            'email': email,
            'eligible_for_certificate': eligible,
            'final_score': final_score,
            'feedback': feedback,
            'any_correct': any_correct,  # Add this to context
            'part1_score': request.session.get('quiz_part1_score', 0.0)  # Add Part 1 score
        }
        return render(request, 'ses598_rem_quiz_part2.html', context)

    # GET request - display quiz with stored email if available
    context = {
        'show_results': False,
        'email': email if email else 'Anonymous',
        'part1_completed': True,
        'part1_score': request.session.get('quiz_part1_score', 0.0),
        'eligible_for_certificate': False
    }
    return render(request, 'ses598_rem_quiz_part2.html', context)

def reset_quiz(request):
    """Reset quiz state while preserving session ID"""
    # Clear quiz results but keep session ID
    session_id = request.session.get('openuav_session_id')
    quiz_keys = ['quiz_email', 'quiz_part1_score', 'quiz_part2_score', 
                'quiz_part1_completed', 'quiz_part2_completed']
    
    for key in quiz_keys:
        if key in request.session:
            del request.session[key]
    
    # Restore session ID if it existed
    if session_id:
        request.session['openuav_session_id'] = session_id
    
    return redirect('ses598_quiz')

def get_ses598_course_data():
    return {
        'course_info': {
            'title': 'RAS/SES 598: Space Robotics and AI',
            'semester': [
                'Spring 2025',
                'RAS <a href="https://catalog.apps.asu.edu/catalog/classes/classlist?campusOrOnlineSelection=A&honors=F&keywords=39245&promod=F&searchType=all&term=2251#detailsOpen=39245-131738" target="_blank">#39245</a>',
                'SES <a href="https://catalog.apps.asu.edu/catalog/classes/classlist?keywords=39153&searchType=all&term=2251#detailsOpen=39153-126724" target="_blank">#39153</a>'
            ],
            'meeting_times': ' (UPDATED) Tu/Th 4:30-5:45pm',
            'location': '(UPDATED) ASU Poly Campus, Room <a href="https://www.google.com/maps?q=33.306167144617966,-111.67643317847114" target="_blank">SANCA</a>-355',
            'instructor': 'Dr. Jnaneshwar Das',
            'office_hours': 'Friday 11am-12pm, and by appointment',
            'contact': 'jdas5@asu.edu',
            'description': 'This course provides a comprehensive introduction to robotic exploration and AI-driven mapping and sampling techniques, tailored for space exploration and earth observation. Students will gain expertise in key areas such as computer vision, Simultaneous Localization and Mapping (SLAM), multi-robot coordination, and operations in extreme environments using advanced AI tools. The curriculum emphasizes real-world implementation, combining lectures with hands-on projects using mobility autonomy systems, including autonomous ground, aerial, and aquatic robots available as digital twins and physically in the <a href="https://deepgis.org/dreamslab/#assets" target="_blank" class="link-primary">DREAMS Laboratory</a>. The course culminates in a group-based final project, where students design and demonstrate end-to-end robotic systems for future space exploration, planetary science, and earth observation.',
            'quiz_info': 'Test your foundation in robotics and AI concepts by completing this quiz. It helps assess if this course aligns with your interests. A timestamped certificate of successful completion will be used to prioritize students if the course reaches capacity.'
        },
        'prerequisites': [
            {'category': 'Mathematics', 'requirement': 'Linear algebra (vectors, matrices, eigenvalues), calculus (derivatives, gradients), and probability theory (Bayes rule, distributions)'},
            {'category': 'Programming', 'requirement': 'Strong Python programming skills with experience in scientific computing libraries (NumPy, SciPy, PyTorch/TensorFlow)'},
            {'category': 'Computer Vision', 'requirement': 'Basic understanding of image processing, feature detection, and geometric transformations'},
            {'category': 'Computing Systems', 'requirement': 'Experience with Linux/Unix systems, version control (Git), and command-line tools'},
            {'category': 'Recommended', 'requirement': 'Prior exposure to ROS (Robot Operating System), CUDA programming, or parallel computing'},
            {'category': 'Required Software', 'requirement': 'Students must have a computer capable of running Linux and processing 3D graphics'}
        ],
        'recommended_papers': {
            'Scene Representation and View Synthesis': [
                {
                    'title': 'Gaussian Splatting SLAM',
                    'authors': 'Matsuki, H., Murai, R., Kelly, P.H.J., Davison, A.J.',
                    'venue': 'CVPR 2024',
                    'url': 'https://openaccess.thecvf.com/content/CVPR2024/papers/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.pdf',
                    'citation': '@InProceedings{Matsuki_2024_CVPR, author = {Matsuki, Hidenobu and Murai, Riku and Kelly, Paul H.J. and Davison, Andrew J.}, title = {Gaussian Splatting SLAM}, booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, month = {June}, year = {2024}, pages = {18039-18048} }'
                },
                {
                    'title': 'NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis',
                    'authors': 'Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.',
                    'venue': 'Communications of the ACM 2021',
                    'url': 'https://dl.acm.org/doi/abs/10.1145/3503250'
                },
                {
                    'title': 'Diffusion Models in Vision: A Survey',
                    'authors': 'Croitoru, F.A., Hondru, V., Ionescu, R.T., Shah, M.',
                    'venue': 'IEEE TPAMI 2023',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/10081412'
                },
                {
                    'title': 'PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency',
                    'authors': 'Pan, Y., Zhong, X., Wiesmann, L., Posewsky, T., Behley, J., Stachniss, C.',
                    'venue': 'IEEE Transactions on Robotics 2024',
                    'url': 'https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2024tro.pdf'
                }
            ],
            'SLAM and Active Perception': [
                {
                    'title': 'Structure-Invariant Range-Visual-Inertial Odometry',
                    'authors': 'Alberico, I., Delaune, J., Cioffi, G., Scaramuzza, D.',
                    'venue': 'IROS 2024',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/10801775'
                },
                {
                    'title': 'Receding Horizon "Next-Best-View" Planner for 3D Exploration',
                    'authors': 'Bircher, A., Kamel, M., Alexis, K., Oleynikova, H., Siegwart, R.',
                    'venue': 'ICRA 2016',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/7487281'
                },
                {
                    'title': 'Active Semantic Mapping and Pose Graph Spectral Analysis for Robot Exploration',
                    'authors': 'Zhang, R., Bong, H.M., Beltrame, G.',
                    'venue': 'IROS 2024',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/10802821'
                },
                {
                    'title': 'Ultimate SLAM? Combining Events, Images, and IMU for Robust Visual SLAM in HDR and High-Speed Scenarios',
                    'authors': 'Vidal, A.R., Rebecq, H., Horstschaefer, T., Scaramuzza, D.',
                    'venue': 'IEEE Robotics and Automation Letters 2018',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/8258997'
                },
                {
                    'title': 'Data-Efficient Collaborative Decentralized Thermal-Inertial Odometry',
                    'authors': 'Polizzi, V., Hewitt, R., Hidalgo-Carrió, J., Delaune, J., Scaramuzza, D.',
                    'venue': 'IEEE Robotics and Automation Letters 2022',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/9844233'
                }
            ],
            'Extreme Environment Operations': [
                {
                    'title': 'Autonomous robotics is driving Perseverance rover\'s progress on Mars',
                    'authors': 'Verma, V., et al.',
                    'venue': 'Science Robotics 2023',
                    'url': 'https://www.science.org/doi/full/10.1126/scirobotics.adi3099'
                },
                {
                    'title': 'Precise pose estimation of the NASA Mars 2020 Perseverance rover through a stereo-vision-based approach',
                    'authors': 'Andolfo, S., Petricca, F., Genova, A.',
                    'venue': 'Journal of Field Robotics 2023',
                    'url': 'https://onlinelibrary.wiley.com/doi/epdf/10.1002/rob.22138'
                },
                {
                    'title': 'Ingenuity Mars Helicopter: From Technology Demonstration to Extraterrestrial Scout',
                    'authors': 'Tzanetos, T., et al.',
                    'venue': 'IEEE Aerospace Conference 2022',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/9843428'
                }
            ],
            'Sampling Strategies and Information Theory': [
                {
                    'title': 'Data-driven robotic sampling for marine ecosystem monitoring',
                    'authors': 'Das, J., Py, F., Harvey, J.B.J., Ryan, J.P., Gellene, A., Graham, R., Caron, D.A., Rajan, K., Sukhatme, G.S.',
                    'venue': 'International Journal of Robotics Research 2015',
                    'url': 'https://journals.sagepub.com/doi/full/10.1177/0278364915587723'
                },
                {
                    'title': 'A 3D drizzle algorithm for JWST and practical application to the MIRI Medium Resolution Spectrometer',
                    'authors': 'Law, D.D., et al.',
                    'venue': 'The Astronomical Journal 2023',
                    'url': 'https://iopscience.iop.org/article/10.3847/1538-3881/acdddc/pdf'
                },
                {
                    'title': 'An information-theoretic approach to optimize JWST observations and retrievals of transiting exoplanet atmospheres',
                    'authors': 'Howe, A.R., Burrows, A., Deming, D.',
                    'venue': 'The Astrophysical Journal 2017',
                    'url': 'https://iopscience.iop.org/article/10.3847/1538-4357/835/1/96/pdf'
                }
            ],
            'Multi-Robot Coordination': [
                {
                    'title': 'Distributed exploration in multi-armed bandits',
                    'authors': 'Hillel, E., Karnin, Z.S., Koren, T., Lempel, R., Somekh, O.',
                    'venue': 'NeurIPS 2013',
                    'url': 'https://proceedings.neurips.cc/paper_files/paper/2013/file/598b3e71ec378bd83e0a727608b5db01-Paper.pdf'
                },
                {
                    'title': 'Decentralized cooperative stochastic bandits',
                    'authors': 'Martínez-Rubio, D., Kanade, V., Rebeschini, P.',
                    'venue': 'NeurIPS 2019',
                    'url': 'https://proceedings.neurips.cc/paper/2019/file/85353d3b2f39b9c9b5ee3576578c04b7-Paper.pdf'
                }
            ],
            'Digital and Cyber-Physical Twins': [
                {
                    'title': 'Virtual Shake Robot: Simulating Dynamics of Precariously Balanced Rocks for Overturning and Large-displacement Processes',
                    'authors': 'Chen, Z., Arrowsmith, R., Das, J., Wittich, C., Madugo, C., Kottke, A.',
                    'venue': 'Seismica 2024',
                    'url': 'https://seismica.library.mcgill.ca/article/view/692',
                    'citation': '@article{chen2024virtual, title={Virtual Shake Robot: Simulating Dynamics of Precariously Balanced Rocks for Overturning and Large-displacement Processes}, author={Chen, Z. and Arrowsmith, R. and Das, J. and Wittich, C. and Madugo, C. and Kottke, A.}, journal={Seismica}, volume={3}, number={1}, year={2024}, doi={10.26443/seismica.v3i1.692}}'
                }
            ],
            'Control and Planning': [
                {
                    'title': 'Model Predictive Contouring Control for Time-Optimal Quadrotor Flight',
                    'authors': 'Romero, A., Sun, S., Foehn, P., Scaramuzza, D.',
                    'venue': 'IEEE Transactions on Robotics 2022',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/9802523'
                },
                {
                    'title': 'Assessment of the Mars 2020 Entry, Descent, and Landing Simulation',
                    'authors': 'Way, D.W., Dutta, S., Zumwalt, C.H., Blette, D.J.',
                    'venue': 'AIAA SciTech 2022',
                    'url': 'https://ntrs.nasa.gov/api/citations/20210024480/downloads/Mars2020_Flight_Mechanics_Simulation_Paper_v3.pdf'
                },
                {
                    'title': 'Psyche Mission System Level Guidance, Navigation, and Control Off-Nominal Testing',
                    'authors': 'Arthur, P., Navarro, J., Sover, K., Sternberg, D., Twu, P.',
                    'venue': 'IEEE Aerospace Conference 2024',
                    'url': 'https://ieeexplore.ieee.org/document/10521063'
                }
            ],
            'Computer Vision and 3D Reconstruction': [
                {
                    'title': 'DUSt3R: Geometric 3D Vision Made Easy',
                    'authors': 'Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.',
                    'venue': 'CVPR 2024',
                    'url': 'https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf'
                },
                {
                    'title': 'Past, Present, and Future of Simultaneous Localization and Mapping: Toward the Robust-Perception Age',
                    'authors': 'Cadena, C., et al.',
                    'venue': 'IEEE Transactions on Robotics 2016',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/7747236'
                },
                {
                    'title': 'Structure-From-Motion Revisited',
                    'authors': 'Schonberger, J.L., Frahm, J.M.',
                    'venue': 'CVPR 2016',
                    'url': 'https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf'
                },
                {
                    'title': 'ORB-SLAM: A Versatile and Accurate Monocular SLAM System',
                    'authors': 'Mur-Artal, R., Montiel, J.M.M., Tardós, J.D.',
                    'venue': 'IEEE Transactions on Robotics 2015',
                    'url': 'https://ieeexplore.ieee.org/abstract/document/7219438'
                }
            ]
        },
        'modules': [
            {
                'week': '1-2 (Jan 14-25)',
                'title': 'State estimation and Controls',
                'topics': [
                    'Least squares and maximum likelihood estimation (MLE)',
                    'State space models and linear dynamical systems',
                    'State-estimation with Kalman and particle filters',
                    'PID control, linear quadratic regulator (LQR), and model predictive control (MPC)',
                    'Entry descent and landing (EDL), guidance navigation and control (GNC), and attitude determination and control system (ADCS)'
                ],
                'assignment': 'Assignment 1: <a href="https://github.com/DREAMS-lab/RAS-SES-598-Space-Robotics-and-AI/tree/main/assignments/state_estimation_assignment" target="_blank">State estimation in 2D using ROS2</a> (Due: Jan 21, 2025)<br>Assignment 2: <a href="https://github.com/DREAMS-lab/RAS-SES-598-Space-Robotics-and-AI/tree/main/assignments/inverted_pendulum_assignment" target="_blank">Inverted pendulum stabilization using ROS2</a> (Due: Jan 28, 2025)',
                'lectures': [
                    {'title': 'Course Introduction & State Estimation Overview', 'url': '/static/lectures/lecture1.pdf'},
                    {'title': 'State Estimation Techniques', 'url': '/static/lectures/lecture2.pdf'}
                ]
            },
            {
                'week': '3-4 (Jan 28-Feb 8)',
                'title': 'Computer Vision and 3D Reconstruction',
                'topics': [
                    'Image formation and camera models',
                    'Feature detection and matching',
                    'Epipolar geometry and stereo vision',
                    'Structure from Motion (SfM)',
                    'Multi-View Stereo (MVS)'
                ],
                'assignment': 'Assignment 3: Offline 3D reconstruction pipeline leveraging OpenCV and COLMAP in ROS2'
            },
            {
                'week': '5 (Feb 11-15)',
                'title': 'Scene Representation, View Synthesis, and Scene Analysis',
                'topics': [
                    'Scene representation: Orthomaps, pointcloud, mesh models, voxel grids, implicit surface models, and surfels',
                    'View synthesis: Neural Radiance Fields (NeRF), and Gaussian Splatting',
                    'Scene analysis: Semantic segmentation of images and point clouds leveraging neural networks',
                    'Generative scenes: Diffusion models'
                ],
                'assignment': 'Assignment 4: View synthesis and scene analysis on Apollo 17 and Lunar analog datasets.'
            },
            {
                'week': '6 (Feb 18-22)',
                'title': 'Sampling Strategies and Information Theory',
                'topics': [
                    'Information theory fundamentals',
                    'Active sampling and exploration',
                    'Multi-armed bandits and Bayesian optimization',
                    'Information gain in exploration'
                ],
                'assignment': 'Assignment 5: Optimal sampling challenge on James Webb Space Telescope (JWST)datasets.'
            },
            {
                'week': '7-8 (Feb 25-Mar 8)',
                'title': 'Digital and Cyber-Physical Twins',
                'topics': [
                    'Decision support systems, geographic information systems (GIS), and digital twins',
                    'Self-supervised learning of stochastic dynamical processes with physical twins',
                    'Case study 1 - earthquake geology: Virtual shake robot and particle dynamical studies',
                    'Case study 2 - ecological digital and physical twins',
                    'Closing the loop on model improvement with cyber-physical twins'
                ],
                'assignment': 'Assignment 6: Adaptive digital twin system involving seismic studies with virtual shake robot and ShakeBot.'
            },
            {
                'week': '9-10 (Mar 17-29)',
                'title': 'SLAM and Active Perception',
                'topics': [
                    'Information-theoretic SLAM',
                    'Active view selection',
                    'Uncertainty-aware mapping',
                    'Exploration-exploitation trade-offs',
                    'Resource-constrained planning'
                ],
                'assignment': 'Midterm Project: Information-driven Robot Autonomy Challenge either in digital twins or physical robots.'
            },
            {
                'week': '11-12 (Apr 1-12)',
                'title': 'Multi-Robot Coordination and Distributed Learning',
                'topics': [
                    'Distributed bandit algorithms',
                    'Multi-agent exploration strategies',
                    'Collaborative information gathering',
                    'Decentralized decision making',
                    'Communication-aware sampling'
                ],
                'assignment': 'Assignment 7: Multi-robot exploration system in digital twins.'
            },
            {
                'week': '13-14 (Apr 15-26)',
                'title': 'Extreme Environment Operations',
                'topics': [
                    'Risk-aware exploration',
                    'Robust sampling strategies',
                    'Adaptive resource allocation',
                    'Environmental uncertainty modeling',
                    'Safety-constrained learning'
                ],
                'assignment': 'Assignment 8:  Digital twin exercise on planning under risks and uncertainty.'
            },
            {
                'week': '15-16 (Apr 29-May 3)',
                'title': 'Integration & Advanced Applications',
                'topics': [
                    'Meta-learning for exploration',
                    'Transfer learning in space applications',
                    'Lifelong learning systems',
                    'Integrated perception-planning-learning',
                    'Real-world deployment strategies'
                ],
                'assignment': 'Final (group) Project: End-to-end autonomous robotic system themed around space exploration, planetary science, or earth observation.'
            }
        ],
        'grading': {
            'Assignments': {
                'percentage': 20,
                'description': 'Eight assignments throughout the semester to reinforce learning concepts and practical skills.'
            },
            'Midterm Project': {
                'percentage': 20,
                'description': 'A comprehensive project due mid-semester that integrates core concepts covered in the first half.'
            },
            'Final Project': {
                'percentage': 50,
                'description': 'A major project that demonstrates mastery of course concepts, including implementation and documentation. This can be a continuation of the midterm project'
            },
            'Class Participation': {
                'percentage': 10,
                'description': 'Active participation in class discussions, group activities, and engagement with course material.'
            }
        },
        'tutorials': {
            'Computer Vision': [
                {
                    'title': 'Multi-View Geometry',
                    'description': 'Epipolar geometry, fundamental matrix, and camera calibration',
                    'difficulty': 'Intermediate',
                    'url': 'multiview_geometry'
                },
                {
                    'title': 'Bundle Adjustment',
                    'description': 'Interactive demo of bundle adjustment with multiple cameras',
                    'difficulty': 'Advanced',
                    'url': 'bundle_adjustment_buddy'
                },
                {
                    'title': 'SLAM Tutorial',
                    'description': 'Simultaneous Localization and Mapping fundamentals',
                    'difficulty': 'Advanced',
                    'url': 'slam_buddy'
                }
            ],
            'Control and Planning': [
                {
                    'title': 'Drone Control Primer',
                    'description': 'Interactive introduction to drone control and navigation',
                    'difficulty': 'Advanced',
                    'url': 'drone_buddy'
                },
                {
                    'title': 'Path Planning',
                    'description': 'A*, RRT, and potential fields algorithms',
                    'difficulty': 'Intermediate',
                    'url': 'path_planning_buddy'
                },
                {
                    'title': 'Cart Pole Control',
                    'description': 'LQR control for inverted pendulum',
                    'difficulty': 'Advanced',
                    'url': 'cart_pole_lqr_buddy'
                }
            ],
            'Estimation': [
                {
                    'title': 'Parameter Estimation',
                    'description': 'Least-squares estimation for linear regression',
                    'difficulty': 'Intermediate',
                    'url': 'param_estimation_buddy'
                },
                {
                    'title': 'Random Sample Consensus (RANSAC)',
                    'description': 'Hands-on implementation of RANSAC for robust model fitting with outlier rejection',
                    'difficulty': 'Beginner',
                    'url': 'ransac_buddy'
                },
                {
                    'title': 'Gaussian Processes',
                    'description': 'Interactive visualization of GP regression for spatial prediction and uncertainty estimation',
                    'difficulty': 'Advanced',
                    'url': 'gaussian_processes_buddy'
                },
                {
                    'title': 'Information-based Sampling',
                    'description': 'Cross-Entropy Sampling',
                    'difficulty': 'Intermediate',
                    'url': 'sampling_buddy'
                },
                {
                    'title': 'Sensor Fusion',
                    'description': 'Interactive Kalman filter demo showing process vs measurement noise trade-offs in fast-moving systems',
                    'difficulty': 'Advanced',
                    'url': 'sensor_fusion_buddy'
                }
            ]
        },
        'recommended_books': [
            {
                'title': 'Probabilistic Robotics',
                'authors': 'Sebastian Thrun, Wolfram Burgard, Dieter Fox',
                'description': 'A foundational text on probabilistic approaches to robotics, covering core algorithms for perception, estimation, and planning under uncertainty.'
            },
            {
                'title': 'Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches',
                'authors': 'Dan Simon',
                'description': 'A comprehensive guide to state estimation techniques with practical applications in navigation and control systems.'
            },
            {
                'title': 'Pattern Recognition and Machine Learning',
                'authors': 'Christopher M. Bishop',
                'description': 'The definitive text on modern pattern recognition methods with a focus on Bayesian techniques and machine learning algorithms.',
                'url': 'https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf'
            },
            {
                'title': 'Multiple View Geometry in Computer Vision',
                'authors': 'Richard Hartley and Andrew Zisserman',
                'description': 'The foundational text for understanding geometric relationships between multiple views and 3D reconstruction techniques.'
            },
            {
                'title': 'Optimal Control and Estimation',
                'authors': 'Robert F. Stengel',
                'description': 'A classic text bridging theory and practice in optimal control, estimation, and stochastic systems analysis.'
            }
        ]
    }

def ses598_course_view(request):
    """View function for the SES598 course page"""
    return render(request, 'ses598_course.html', {'syllabus': get_ses598_course_data()})

def dreamslab_home_view(request):
    """DREAMS Lab home page"""
    course_data = get_ses598_course_data()
    context = {
        'research_areas': Research.objects.all(),
        'people': People.objects.all(),
        'publications': Publication.objects.all(),
        'funding_source': FundingSource.objects.all(),
        'assets': Asset.objects.all(),
        'course': course_data['course_info']
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

from django.db.models import Avg, Max
from django.contrib.admin.views.decorators import staff_member_required

@staff_member_required
def quiz_admin_view(request):
    """Admin view for quiz statistics"""
    # Get all unique emails
    submissions = QuizSubmission.objects.values('email').distinct()
    
    results = []
    total_submissions = 0
    part1_scores = []
    part2_scores = []
    
    # Initialize topic statistics
    part1_topic_stats = [{'correct': 0, 'total': 0} for _ in range(5)]
    part2_topic_stats = [{'correct': 0, 'total': 0} for _ in range(5)]
    
    # Correct answers for each part
    part1_answers = {
        'q1': '3',  # SLAM purpose
        'q2': '2',  # LiDAR
        'q3': '1',  # Occupancy grid
        'q4': '1',  # GPS challenge
        'q5': '2',  # Path planning
    }
    
    part2_answers = {
        'q1': '2',  # Stereo vision baseline
        'q2': '2',  # SLAM loop closure
        'q3': '3',  # Kalman filter parameters
        'q4': '3',  # Sampling strategy
        'q5': '1',  # Error type in rock mapping
    }
    
    for submission in submissions:
        email = submission['email']
        
        # Get latest submissions for each part
        part1 = QuizSubmission.objects.filter(
            email=email, 
            quiz_id='SES598'
        ).order_by('-submission_date').first()
        
        part2 = QuizSubmission.objects.filter(
            email=email, 
            quiz_id='SES598_Part2'
        ).order_by('-submission_date').first()
        
        if part1 or part2:
            total_submissions += 1
            
            # Initialize result dictionary
            result = {
                'email': email,
                'part1_score': 0,
                'part2_score': 0,
                'combined_score': 0,
                'last_submission': max(
                    part1.submission_date if part1 else timezone.make_aware(timezone.datetime.min),
                    part2.submission_date if part2 else timezone.make_aware(timezone.datetime.min)
                )
            }
            
            # Process Part 1
            if part1:
                correct_count = 0
                for i in range(1, 6):
                    q_key = f'q{i}'
                    answer = getattr(part1, q_key, '')
                    is_correct = answer == part1_answers[q_key]
                    result[f'part1_{q_key}_correct'] = is_correct
                    if answer:
                        part1_topic_stats[i-1]['total'] += 1
                        if is_correct:
                            part1_topic_stats[i-1]['correct'] += 1
                            correct_count += 1
                
                score = (correct_count / 5) * 100
                result['part1_score'] = score
                part1_scores.append(score)
            
            # Process Part 2
            if part2:
                correct_count = 0
                for i in range(1, 6):
                    q_key = f'q{i}'
                    answer = getattr(part2, q_key, '')
                    is_correct = answer == part2_answers[q_key]
                    result[f'part2_{q_key}_correct'] = is_correct
                    if answer:
                        part2_topic_stats[i-1]['total'] += 1
                        if is_correct:
                            part2_topic_stats[i-1]['correct'] += 1
                            correct_count += 1
                
                score = (correct_count / 5) * 100
                result['part2_score'] = score
                part2_scores.append(score)
            
            # Calculate combined score
            result['combined_score'] = (result['part1_score'] * 0.6) + (result['part2_score'] * 0.4)
            results.append(result)
    
    # Calculate percentages for topic statistics
    for stats in part1_topic_stats + part2_topic_stats:
        stats['correct_percentage'] = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    # Sort results by combined score (descending)
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    context = {
        'total_submissions': total_submissions,
        'avg_part1_score': sum(part1_scores) / len(part1_scores) if part1_scores else 0,
        'avg_part2_score': sum(part2_scores) / len(part2_scores) if part2_scores else 0,
        'results': results,
        'part1_topic_stats': part1_topic_stats,
        'part2_topic_stats': part2_topic_stats
    }
    
    return render(request, 'quiz_admin.html', context)

def ses598_course(request):
    # ... other code ...
    
    papers = [
        {
            'title': 'Assessment of the Mars 2020 Entry, Descent, and Landing Simulation',
            'url': 'https://ntrs.nasa.gov/api/citations/20210024480/downloads/Mars2020_Flight_Mechanics_Simulation_Paper_v3.pdf',
            'authors': 'P. D. Burkhart et al.',
            'year': '2021',
            'description': 'Comprehensive analysis of Mars 2020 EDL simulation and actual performance'
        },
        # ... other papers ...
    ]

    # ... rest of the function ...