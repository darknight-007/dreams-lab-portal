import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import RANSACRegressor
from scipy.signal import savgol_filter

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

def compute_curvature(x, y):
    # Smooth the points first
    y_smooth = savgol_filter(y, window_length=11, polyorder=3)
    
    # Compute first and second derivatives
    dy = np.gradient(y_smooth)
    dx = np.gradient(x)
    d2y = np.gradient(dy)
    d2x = np.gradient(dx)
    
    # Compute curvature
    curvature = np.abs(d2y * dx - dy * d2x) / (dx * dx + dy * dy)**1.5
    return curvature

def ransac_line_fitting(x, y, curvature, threshold=0.3):
    X = x.reshape(-1, 1)
    
    # Weight points by inverse curvature
    weights = 1 / (1 + 10 * curvature)  # Reduce weight of high-curvature points
    
    # Initial RANSAC fit
    ransac = RANSACRegressor(
        residual_threshold=threshold,
        random_state=42,
        min_samples=50,
        max_trials=200,
        stop_probability=0.999
    )
    ransac.fit(X, y)
    
    # Refine the inlier/outlier classification
    initial_inliers = ransac.inlier_mask_
    residuals = np.abs(y - ransac.predict(X))
    
    # Combine residuals with curvature information
    combined_score = residuals + 0.5 * curvature
    
    # Final classification using adaptive threshold
    threshold_refined = np.mean(combined_score[initial_inliers]) + \
                       2 * np.std(combined_score[initial_inliers])
    inlier_mask = combined_score < threshold_refined
    outlier_mask = ~inlier_mask
    
    return inlier_mask, outlier_mask, ransac

# Create the plot
plt.figure(figsize=(12, 6))

# Generate the curved line and sample points
x_coords, y_coords = generate_curved_line()
x_random, y_random = sample_points_on_curve(x_coords, y_coords)

# Compute curvature
curvature = compute_curvature(x_random, y_random)

# Fit RANSAC with curvature information
inlier_mask, outlier_mask, ransac = ransac_line_fitting(x_random, y_random, curvature)

# Plot results
plt.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.3)  # Original curve
plt.plot(x_random[inlier_mask], y_random[inlier_mask], 'b.', 
         markersize=2, label='Ground (inliers)')
plt.plot(x_random[outlier_mask], y_random[outlier_mask], 'r.', 
         markersize=2, label='Boulder (outliers)')

# Plot the fitted line
line_x = np.array([min(x_random), max(x_random)])
line_y = ransac.predict(line_x.reshape(-1, 1))
plt.plot(line_x, line_y, 'g-', linewidth=1, label='Fitted ground line')

plt.axis('equal')
plt.grid(False)
plt.legend()
plt.show()
