import numpy as np

from derotation.analysis.fit_ellipse import fit_ellipse_to_points


def test_fit_ellipse_to_points_circular_pattern():
    """
    Test fit_ellipse_to_points with a circular pattern of points.
    
    This test verifies that the function correctly estimates the center of the
    fitted ellipse when given a well-defined set of points arranged in a circle.
    """
    # Create a circular pattern of points centered at (128, 128)
    center_x, center_y = 128, 128
    radius = 50
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    
    # Generate points on the circle
    x_points = center_x + radius * np.cos(angles)
    y_points = center_y + radius * np.sin(angles)
    points = np.column_stack([x_points, y_points])
    
    # Fit ellipse to the points
    fitted_cx, fitted_cy, a, b, theta = fit_ellipse_to_points(points)
    
    # Verify that the fitted center is close to the expected center
    assert np.isclose(fitted_cx, center_x, atol=5), \
        f"Expected center_x close to {center_x}, got {fitted_cx}"
    assert np.isclose(fitted_cy, center_y, atol=5), \
        f"Expected center_y close to {center_y}, got {fitted_cy}"
    
    # For a circle, semi-major and semi-minor axes should be similar
    assert np.isclose(a, b, rtol=0.2), \
        f"For a circle, semi-axes should be similar: a={a}, b={b}"


def test_fit_ellipse_to_points_horizontal_ellipse():
    """
    Test fit_ellipse_to_points with a horizontal elliptical pattern.
    
    This test verifies correct estimation of ellipse center for a
    horizontally-oriented ellipse.
    """
    # Create a horizontal elliptical pattern
    center_x, center_y = 200, 150
    major_axis = 80
    minor_axis = 40
    angles = np.linspace(0, 2 * np.pi, 25, endpoint=False)
    
    # Generate points on the ellipse
    x_points = center_x + major_axis * np.cos(angles)
    y_points = center_y + minor_axis * np.sin(angles)
    points = np.column_stack([x_points, y_points])
    
    # Fit ellipse to the points
    fitted_cx, fitted_cy, a, b, theta = fit_ellipse_to_points(points)
    
    # Verify that the fitted center is close to the expected center
    assert np.isclose(fitted_cx, center_x, atol=10), \
        f"Expected center_x close to {center_x}, got {fitted_cx}"
    assert np.isclose(fitted_cy, center_y, atol=10), \
        f"Expected center_y close to {center_y}, got {fitted_cy}"


def test_fit_ellipse_to_points_minimum_points():
    """
    Test fit_ellipse_to_points with minimum number of valid points.
    
    The function requires at least 5 valid points to fit an ellipse.
    """
    # Create exactly 5 points forming a simple pattern
    points = np.array([
        [100, 100],
        [150, 100],
        [150, 150],
        [100, 150],
        [125, 125],
    ])
    
    # Should work with 5 points
    fitted_cx, fitted_cy, a, b, theta = fit_ellipse_to_points(points)
    
    # Verify valid outputs
    assert isinstance(fitted_cx, (int, float, np.number))
    assert isinstance(fitted_cy, (int, float, np.number))
    assert a > 0, "Semi-major axis should be positive"
    assert b > 0, "Semi-minor axis should be positive"


def test_fit_ellipse_to_points_raises_error_too_few_points():
    """
    Test that fit_ellipse_to_points raises an error with insufficient points.
    """
    # Create only 4 points (less than minimum required)
    points = np.array([
        [100, 100],
        [150, 100],
        [150, 150],
        [100, 150],
    ])
    
    # Should raise ValueError
    try:
        fit_ellipse_to_points(points)
        assert False, "Expected ValueError for insufficient points"
    except ValueError as e:
        assert "Not enough valid points" in str(e)


def test_fit_ellipse_to_points_with_nan_values():
    """
    Test that fit_ellipse_to_points correctly handles NaN values.
    
    The function should skip NaN points and still work if enough valid points remain.
    """
    # Create points with some NaN values
    points = np.array([
        [100, 100],
        [150, 100],
        [np.nan, np.nan],  # NaN point
        [150, 150],
        [100, 150],
        [125, 125],
        [np.nan, 200],  # Partial NaN
    ])
    
    # Should work by ignoring NaN rows
    fitted_cx, fitted_cy, a, b, theta = fit_ellipse_to_points(points)
    
    # Verify valid outputs
    assert not np.isnan(fitted_cx), "Center x should not be NaN"
    assert not np.isnan(fitted_cy), "Center y should not be NaN"
    assert a > 0, "Semi-major axis should be positive"
    assert b > 0, "Semi-minor axis should be positive"
