import sys
import bpy
import os
import numpy as np
import cv2

import bpy
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


import time

def setup_scene(sensor_width, focal_length, baseline, toe_in_angle, distance):
    """Set up the Blender scene with a perturbed geodesic sphere and two cameras using configurator parameters."""
    # Clear all existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Create a perturbed geodesic sphere at the specified distance
    sphere_location = (0, distance, 0)  # Place the sphere `distance` meters in front of the cameras
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1, location=sphere_location)
    sphere = bpy.context.view_layer.objects.active  # Get the created object
    if sphere is None:
        raise RuntimeError("Failed to create geodesic sphere.")

    # Add displacement modifier
    bpy.ops.object.modifier_add(type='DISPLACE')
    displacement = sphere.modifiers[-1]
    displacement.texture = bpy.data.textures.new("DisplacementTexture", 'CLOUDS')
    displacement.texture.noise_scale = 0.5
    displacement.strength = 0.5
    bpy.ops.object.shade_smooth()

    # Compute camera positions based on baseline (in meters)
    baseline_m = baseline / 1000  # Convert mm to meters
    left_camera_position = (-baseline_m / 2, 0, 0)
    right_camera_position = (baseline_m / 2, 0, 0)

    # Add left camera
    bpy.ops.object.camera_add(location=left_camera_position, rotation=(np.radians(90), 0, np.radians(toe_in_angle)))
    left_camera = bpy.context.view_layer.objects.active
    if left_camera is None:
        raise RuntimeError("Failed to create the left camera.")
    left_camera.name = "LeftCamera"

    # Add right camera
    bpy.ops.object.camera_add(location=right_camera_position, rotation=(np.radians(90), 0, -np.radians(toe_in_angle)))
    right_camera = bpy.context.view_layer.objects.active
    if right_camera is None:
        raise RuntimeError("Failed to create the right camera.")
    right_camera.name = "RightCamera"

    # Set camera sensor width and focal length
    for camera in [left_camera, right_camera]:
        camera.data.lens = focal_length  # Set focal length in mm
        camera.data.sensor_width = sensor_width  # Set sensor width in mm

    # Add light source
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 8))
    light = bpy.context.view_layer.objects.active
    if light is None:
        raise RuntimeError("Failed to create the light source.")
    light.data.energy = 1000

    return left_camera, right_camera



def setup_render_settings():
    """Set render settings to output RGB images."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.render.resolution_x = 1440
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"


def render_image(camera, output_path):
    """Render an image from the given camera."""
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def compute_disparity(left_img_path, right_img_path):
    """Compute disparity from stereo images."""
    # Read stereo images
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    # Compute disparity map using StereoBM
    stereo = cv2.StereoBM_create(numDisparities=16 * 4, blockSize=15)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    return disparity


def compute_depth(disparity, focal_length_px, baseline_m):
    """Compute depth from disparity."""
    with np.errstate(divide="ignore"):  # Handle divide by zero
        depth = (focal_length_px * baseline_m) / (disparity + 1e-6)  # Add small value to avoid division by zero
    return depth


def save_disparity_and_depth(disparity, depth, output_folder):
    """Save disparity and depth images."""
    # Normalize disparity for saving
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = disparity_normalized.astype(np.uint8)

    # Normalize depth for saving
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # Save images
    disparity_path = os.path.join(output_folder, "disparity_map.png")
    depth_path = os.path.join(output_folder, "depth_map.png")

    cv2.imwrite(disparity_path, disparity_normalized)
    cv2.imwrite(depth_path, depth_normalized)

    print(f"Disparity map saved to: {disparity_path}")
    print(f"Depth map saved to: {depth_path}")

    return disparity_path, depth_path


def visualize_results(left_img_path, right_img_path, disparity_path, depth_path):
    """Display the stereo images, disparity map, and depth map."""
    left_img = plt.imread(left_img_path)
    right_img = plt.imread(right_img_path)
    disparity_img = plt.imread(disparity_path)
    depth_img = plt.imread(depth_path)

    plt.figure(figsize=(18, 12))

    # Left image
    plt.subplot(2, 2, 1)
    plt.imshow(left_img)
    plt.title("Left Camera RGB")
    plt.axis("off")

    # Right image
    plt.subplot(2, 2, 2)
    plt.imshow(right_img)
    plt.title("Right Camera RGB")
    plt.axis("off")

    # Disparity map
    plt.subplot(2, 2, 3)
    plt.imshow(disparity_img, cmap="plasma")
    plt.title("Disparity Map")
    plt.colorbar(label="Disparity")
    plt.axis("off")

    # Depth map
    plt.subplot(2, 2, 4)
    plt.imshow(depth_img, cmap="viridis")
    plt.title("Depth Map")
    plt.colorbar(label="Depth (m)")
    plt.axis("off")

    # plt.show()



def main(sensor_width, focal_length, baseline, distance, toe_in_angle, output_folder):
    # Set up the Blender scene with parameters
    left_camera, right_camera = setup_scene(sensor_width, focal_length, baseline, toe_in_angle, distance)

    # Set render settings
    setup_render_settings()

    # Render stereo images
    left_img_path = os.path.join(output_folder, "left_camera.png")
    right_img_path = os.path.join(output_folder, "right_camera.png")
    render_image(left_camera, left_img_path)
    render_image(right_camera, right_img_path)

    # Camera parameters
    focal_length_px = (focal_length / sensor_width) * bpy.context.scene.render.resolution_x
    baseline_m = baseline / 1000  # mm to meters

    # Compute disparity and depth
    disparity = compute_disparity(left_img_path, right_img_path)
    depth = compute_depth(disparity, focal_length_px, baseline_m)

    # Save disparity and depth maps
    save_disparity_and_depth(disparity, depth, output_folder)



if __name__ == "__main__":
    # Parse command-line arguments
    args = sys.argv[sys.argv.index("--") + 1:]
    sensor_width = float(args[0])
    focal_length = float(args[1])
    baseline = float(args[2])
    distance = float(args[3])
    toe_in_angle = float(args[4])
    output_folder = args[5]

    main(sensor_width, focal_length, baseline, distance, toe_in_angle, output_folder)
