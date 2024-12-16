import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import bpy


def clear_scene():
    """Efficiently clear the Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def load_rock_model(model_path, location=(0, 5, 1)):
    """Load the selected rock model into Blender."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model: {model_path}")

    # Import the .obj file
    bpy.ops.wm.obj_import(filepath=model_path)

    # Set the location of the imported object
    imported_objects = bpy.context.selected_objects
    for obj in imported_objects:
        obj.location = location
    return imported_objects

def create_gazebo_world(cameras, model_path, world_output_path):
    """
    Create a Gazebo .world file based on the Blender setup.
    """
    # Gazebo template for a world file
    gazebo_world_template = f"""<?xml version="1.0" ?>
<sdf version="1.6">
    <world name="default">
        <!-- Include Ground Plane -->
        <include>
            <uri>model://ground_plane</uri>
        </include>

        <!-- Include Sun -->
        <include>
            <uri>model://sun</uri>
        </include>

        <!-- Rock Model -->
        <model name="rock_model">
                    <static>true</static>
            <pose>{cameras[0].location[0]} {cameras[0].location[1]} {cameras[0].location[2]} 0 0 0</pose>
            <link name="link">
                <visual name="visual">
                    <geometry>
                        <mesh>
                            <uri>file://{model_path}</uri>
                        </mesh>
                    </geometry>
                </visual>
            </link>
        </model>

        <!-- Cameras -->
        <model name="left_camera">
                    <static>true</static>
            <pose>{cameras[0].location[0]} {cameras[0].location[1]} {cameras[0].location[2]} 0 0 0</pose>
            <link name="link">
                <sensor name="camera" type="camera">
                    <camera>
                        <horizontal_fov>1.047</horizontal_fov>
                        <image>
                            <width>1920</width>
                            <height>1080</height>
                        </image>
                        <clip>
                            <near>0.1</near>
                            <far>100</far>
                        </clip>
                    </camera>
                </sensor>
            </link>
        </model>

        <model name="right_camera">
                    <static>true</static>
            <pose>{cameras[1].location[0]} {cameras[1].location[1]} {cameras[1].location[2]} 0 0 0</pose>
            <link name="link">
                <sensor name="camera" type="camera">
                    <camera>
                        <horizontal_fov>1.047</horizontal_fov>
                        <image>
                            <width>1920</width>
                            <height>1080</height>
                        </image>
                        <clip>
                            <near>0.1</near>
                            <far>100</far>
                        </clip>
                    </camera>
                </sensor>
            </link>
        </model>
    </world>
</sdf>
"""
    # Write the world file
    with open(world_output_path, "w") as world_file:
        world_file.write(gazebo_world_template)

    print(f"Gazebo world file saved to: {world_output_path}")


def setup_scene(sensor_width, focal_length, baseline, toe_in_angle, distance, model_path, subdivisions=30, displacement_strength=2.0):
    """Set up the Blender scene with a rock model and two cameras."""
    clear_scene()

    # Load the rock model at the specified distance
    sphere_location = (0, distance, 0)  # Place the model `distance` meters in front of the cameras
    load_rock_model(model_path, location=sphere_location)

    # Compute camera positions
    baseline_m = baseline / 1000  # Convert mm to meters
    left_camera_position = (-baseline_m / 2, 0, 0)
    right_camera_position = (baseline_m / 2, 0, 0)

    # Add left and right cameras
    cameras = []
    for position, angle, name in [(left_camera_position, toe_in_angle, "LeftCamera"),
                                  (right_camera_position, -toe_in_angle, "RightCamera")]:
        camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        camera.location = position
        camera.rotation_euler = (np.radians(90), 0, np.radians(angle))
        bpy.context.collection.objects.link(camera)
        cameras.append(camera)

    # Set camera sensor width and focal length
    for camera in cameras:
        camera.data.lens = focal_length
        camera.data.sensor_width = sensor_width

    # Add light source
    light = bpy.data.objects.new("PointLight", bpy.data.lights.new("PointLight", type='POINT'))
    light.location = (0, 1, 3)
    light.data.energy = 3000
    bpy.context.collection.objects.link(light)

    return cameras




def setup_render_settings():
    """Optimize render settings for faster performance."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"  # Use GPU rendering
    scene.cycles.samples = 1024  # Reduce samples for faster renders
    scene.cycles.use_adaptive_sampling = True  # Adaptive sampling for efficiency
    scene.cycles.max_bounces = 4  # Reduce bounces to speed up rendering
    scene.cycles.use_denoising = True  # Use denoising to clean up renders

    # Enable GPU rendering
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True

    # Set resolution and output settings
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 50  # Lower resolution percentage for test renders
    scene.render.image_settings.file_format = "PNG"


def render_image(camera, output_path):
    """Render an image from the given camera."""
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def compute_disparity(left_img_path, right_img_path):
    """Compute disparity map with enhanced accuracy using StereoSGBM."""
    # Read stereo images in grayscale
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    # Validate image loading
    if left_img is None or right_img is None:
        raise ValueError("Error loading stereo images. Check file paths.")

    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 11
    min_disp = -128
    max_disp = 128
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 5
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 200
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    disparity_SGBM = stereo.compute(left_img, right_img)

    # Normalize the values to a range from 0..255 for a grayscale image
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)
    return disparity_SGBM



def compute_depth(disparity, focal_length_px, baseline_m):
    """Compute depth from disparity."""
    with np.errstate(divide="ignore"):  # Handle divide by zero
        depth = (focal_length_px * baseline_m) / (disparity + 1e-6)  # Add small value to avoid division by zero
    return depth

def save_blend_file(output_path):
    """
    Save the current Blender scene as a .blend file.
    """
    try:
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
        print(f"Blender file saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save Blender file: {e}")

def save_disparity_and_depth(disparity, depth, output_folder):
    """Save and visualize disparity and depth images with a color map."""
    # Normalize disparity for saving
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = disparity_normalized.astype(np.uint8)

    # Normalize depth for saving
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # Save images
    disparity_path = os.path.join(output_folder, "disparity_map.png")
    # depth_path = os.path.join(output_folder, "depth_map.png")

    cv2.imwrite(disparity_path, disparity_normalized)
    # cv2.imwrite(depth_path, depth_normalized)

    print(f"Disparity map saved to: {disparity_path}")
    # print(f"Depth map saved to: {depth_path}")

    # Visualize and save colormapped depth image
    plt.figure(figsize=(10, 8))
    plt.imshow(depth, cmap='jet')
    plt.colorbar(label="Depth (arbitrary units)")
    plt.title("Depth Map (Jet Colormap)")
    plt.axis('off')
    depth_path = os.path.join(output_folder, "depth_map.png")
    plt.savefig(depth_path)
    plt.close()

    print(f"Colormapped depth map saved to: {depth_path}")

    return disparity_path, depth_path



def main(sensor_width, focal_length, baseline, distance, toe_in_angle, model_path, output_folder):
    # Set up the Blender scene with parameters
    print("model_path:", model_path)

    left_camera, right_camera = setup_scene(sensor_width, focal_length, baseline, toe_in_angle, distance, model_path)

    # Save the Blender .blend file
    blend_file_path = os.path.join(output_folder, "scene.blend")
    save_blend_file(blend_file_path)

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

    # Generate a Gazebo world
    world_output_path = os.path.join(output_folder, "scene.world")
    create_gazebo_world([left_camera, right_camera], model_path, world_output_path)



if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    sensor_width = float(args[0])
    focal_length = float(args[1])
    baseline = float(args[2])
    distance = float(args[3])
    toe_in_angle = float(args[4])
    model_path = args[5]  # Ensure the model path is passed
    output_folder = args[6]  # Ensure the output folder is passed
    print("model_path:", model_path)
    main(sensor_width, focal_length, baseline, distance, toe_in_angle, model_path, output_folder)
