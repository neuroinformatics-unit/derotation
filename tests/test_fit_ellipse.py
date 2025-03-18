import numpy as np
import pytest
from derotation.fit_ellipse import fit_ellipse_to_points  # Adjust this import if needed

def generate_ellipse_points(center, axes, angle, num_points=100):
    """
    Generate a set of points along an ellipse.
    
    Parameters:
      center (tuple): (cx, cy) center of the ellipse.
      axes (tuple): (a, b) semi-axes lengths.
      angle (float): rotation angle in radians.
      num_points (int): number of points to generate.
      
    Returns:
      numpy.ndarray: Array of shape (num_points, 2) with (x, y) coordinates.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = axes[0] * np.cos(t)
    y = axes[1] * np.sin(t)
    # Rotate the points by the given angle
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    x_rot = cos_angle * x - sin_angle * y + center[0]
    y_rot = sin_angle * x + cos_angle * y + center[1]
    return np.column_stack((x_rot, y_rot))

def test_fit_ellipse_center():
    # Define known parameters for the synthetic ellipse
    known_center = (10.0, 20.0)
    axes = (5.0, 3.0)
    angle = np.pi / 6  # 30 degrees in radians

    # Generate points on the ellipse with the known parameters
    points = generate_ellipse_points(known_center, axes, angle)

    # Fit an ellipse to these points using the function under test
    fitted_params = fit_ellipse_to_points(points)
    
    # Check that the returned parameters include a 'center'
    fitted_center = fitted_params.get('center')
    assert fitted_center is not None, "The fitted parameters must include a 'center' key."
    
    # Validate that the fitted center is close to the known center within a tolerance
    np.testing.assert_allclose(fitted_center, known_center, atol=1e-1)
