import os
import sys
from PIL import Image
import trimesh

# Get the base directory from the command line
if len(sys.argv) != 2:
    print("Usage: python script.py <base_directory>")
    sys.exit(1)

base_dir = sys.argv[1]

# Thumbnail size
thumbnail_size = (256, 256)

# Supported 3D model file type
model_extensions = [".obj"]

# Function to create a thumbnail for a 3D model file
def create_model_thumbnail(model_path, thumbnail_path):
    try:
        # Load the 3D model using trimesh
        mesh = trimesh.load(model_path)

        # Use pyglet for rendering
        import pyglet
        from pyglet.gl import Config

        config = Config(double_buffer=True)
        window = pyglet.window.Window(width=thumbnail_size[0], height=thumbnail_size[1], config=config, visible=False)

        # Render the model
        scene = trimesh.Scene(mesh)
        image = scene.save_image(resolution=thumbnail_size)

        # Save the image as a PNG thumbnail
        with open(thumbnail_path, "wb") as f:
            f.write(image)

        window.close()
        print(f"Thumbnail created for model: {model_path}")

    except Exception as e:
        print(f"Failed to create thumbnail for model {model_path}: {e}")

# Walk through the directory structure
for root, _, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_name, file_extension = os.path.splitext(file)

        # Skip if thumbnail already exists
        thumbnail_path = os.path.join(root, f"{file_name}_thumbnail.png")
        if os.path.exists(thumbnail_path):
            continue

        # Handle 3D models
        if file_extension.lower() in model_extensions:
            create_model_thumbnail(file_path, thumbnail_path)

print("Thumbnail generation complete.")
