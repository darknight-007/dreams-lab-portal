from django.shortcuts import render
from .models import People, Research, Publication, Project, Asset, FundingSource
import subprocess
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json


@csrf_exempt
def run_blender(request):
    if request.method == "POST":
        # Parse parameters from the frontend
        data = json.loads(request.body)
        sensor_width = data.get("sensorWidth", 8.44)
        focal_length = data.get("focalLength", 19.71)
        baseline = data.get("baseline", 240)
        distance = data.get("distance", 5)
        toe_in_angle = data.get("toeInAngle", 0)

        # Paths for output
        output_folder = os.path.join(settings.MEDIA_ROOT, "render_outputs")
        os.makedirs(output_folder, exist_ok=True)
        print(output_folder)
        try:
            # Run Blender script with arguments
            blender_script_path = os.path.join(os.getcwd(), "run_blender.py")
            subprocess.run(
                [
                    "blender", "--background", "--python", blender_script_path,
                    "--", str(sensor_width), str(focal_length), str(baseline),
                    str(distance), str(toe_in_angle), output_folder
                ],
                check=True
            )

            # Return the paths to rendered images
            return JsonResponse({
                "status": "success",

            })
        except subprocess.CalledProcessError as e:
            return JsonResponse({"status": "error", "details": str(e)}, status=500)
    else:
        return JsonResponse({"status": "error", "details": "Invalid request method."}, status=400)


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
        'research_areas': research_areas,
        'assets': assets,
        'funding_source': funding_source,

    }
    return render(request, 'home.html', context)


def stereo_buddy_view(request):
    return render(request, 'stereo-buddy.html')
# views.py
