import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from flask import jsonify
import io
import base64

# Blue dots for ground points
# Red dots for potential rock points during the search phase
# Purple stars for confirmed rock points after convergence
# Green line for the fitted ground line
# Yellow dots for selected points during the search phase

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

def ransac_iteration(X, y, best_model=None, best_score=float('inf'), 
                    n_samples=30, threshold=0.3):
    # Randomly select points with bias towards lower points
    n_points = len(X)
    
    # Add bias towards selecting lower points
    weights = 1 / (1 + np.abs(y - np.min(y)))
    weights /= np.sum(weights)
    
    indices = np.random.choice(n_points, n_samples, replace=False, p=weights)
    
    # Fit line to selected points
    model = LinearRegression()
    model.fit(X[indices], y[indices])
    
    # Calculate residuals
    y_pred = model.predict(X)
    residuals = y - y_pred  # Changed to signed residuals
    
    # Points are inliers if they are close to or below the line
    inlier_mask = (residuals < threshold)
    
    # Calculate model score (sum of absolute residuals for inliers)
    score = np.sum(np.abs(residuals[inlier_mask]))
    
    # Update best model if current one is better and has enough inliers
    if score < best_score and np.sum(inlier_mask) > len(X) * 0.4:
        best_model = model
        best_score = score
    
    # Use the current model for visualization during search
    if not CONVERGED:
        return model, inlier_mask, indices, best_score
    else:
        # Use the best model when converged
        if best_model is not None:
            y_pred = best_model.predict(X)
            residuals = y - y_pred
            inlier_mask = (residuals < threshold)
        return best_model, inlier_mask, indices, best_score

# Create figure and axis for animation
fig, ax = plt.subplots(figsize=(12, 6))

# Generate the curved line and sample points
x_coords, y_coords = generate_curved_line()
x_random, y_random = sample_points_on_curve(x_coords, y_coords)
X = x_random.reshape(-1, 1)

# Initialize plot elements
line_original, = ax.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.3)
points_ground, = ax.plot([], [], 'b.', markersize=2, label='Ground points')
points_boulder, = ax.plot([], [], 'r.', markersize=2, label='Searching rock points')
points_rock = ax.scatter([], [], c='purple', marker='*', s=50, label='Detected rock', alpha=0.8)
line_fit, = ax.plot([], [], 'g-', linewidth=1, label='Ground fit')
points_selected, = ax.plot([], [], 'y.', markersize=4, label='Selected points')

# Set axis properties
ax.set_xlim(min(x_random), max(x_random))
ax.set_ylim(min(y_random) - 1, max(y_random) + 1)
ax.grid(False)
ax.legend()

# Add variables for convergence tracking
best_model = None
best_score = float('inf')
prev_score = float('inf')
convergence_counter = 0
CONVERGED = False
CONVERGENCE_THRESHOLD = 0.01
CONVERGENCE_PATIENCE = 10

def init():
    points_ground.set_data([], [])
    points_boulder.set_data([], [])
    points_rock.set_offsets(np.c_[[], []])
    line_fit.set_data([], [])
    points_selected.set_data([], [])
    return points_ground, points_boulder, points_rock, line_fit, points_selected

def animate(frame):
    global best_model, best_score, prev_score, convergence_counter, CONVERGED
    
    # If already converged, stop the animation
    if CONVERGED:
        anim.event_source.stop()
        return points_ground, points_boulder, points_rock, line_fit, points_selected
    
    # Perform RANSAC iteration using best model so far
    best_model, inlier_mask, selected_indices, best_score = ransac_iteration(
        X, y_random, best_model, best_score)
    
    # Check for convergence
    score_diff = abs(prev_score - best_score)
    if score_diff < CONVERGENCE_THRESHOLD:
        convergence_counter += 1
    else:
        convergence_counter = 0
    
    CONVERGED = convergence_counter >= CONVERGENCE_PATIENCE
    prev_score = best_score
    
    # Get signed residuals to identify points above the line
    y_pred = best_model.predict(X)
    residuals = y_random - y_pred
    
    # Points are rocks if they are significantly above the line
    rock_threshold = 0.5  # Minimum height above the line to be considered a rock
    rock_mask = residuals > rock_threshold
    
    # Update ground points (points near or below the line)
    points_ground.set_data(x_random[~rock_mask], y_random[~rock_mask])
    
    if CONVERGED:
        # When converged, show rock points with stars
        points_boulder.set_data([], [])  # Hide the searching points
        rock_points = np.c_[x_random[rock_mask], y_random[rock_mask]]
        points_rock.set_offsets(rock_points)
    else:
        # During search, show potential rock points as red dots
        points_boulder.set_data(x_random[rock_mask], y_random[rock_mask])
        points_rock.set_offsets(np.c_[[], []])  # Hide the rock stars
    
    # Update fitted line using best model
    line_x = np.array([min(x_random), max(x_random)])
    line_y = best_model.predict(line_x.reshape(-1, 1))
    line_fit.set_data(line_x, line_y)
    
    # Update selected points
    points_selected.set_data(x_random[selected_indices], y_random[selected_indices])
    
    return points_ground, points_boulder, points_rock, line_fit, points_selected

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=200,
                    interval=100, blit=True)

plt.show()
