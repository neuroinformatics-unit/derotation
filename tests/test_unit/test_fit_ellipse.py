import numpy as np
import pytest

from derotation.analysis.fit_ellipse import fit_ellipse_to_points


def generate_ellipse_points(
    center=(120.0, 100.0),
    axes=(40.0, 20.0),
    angle=np.pi / 6,
    n_points=200,
    noise_std=0.5,
):
    """
    Generate synthetic points lying on ellipse with optional noise.
    """
    cx, cy = center
    a, b = axes

    t = np.linspace(0, 2 * np.pi, n_points)

    # Parametric ellipse before rotation
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Rotation matrix
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)],
        ]
    )

    points = R @ np.vstack((x, y))
    points[0] += cx
    points[1] += cy

    # Add Gaussian noise
    points += np.random.normal(scale=noise_std, size=points.shape)

    return points.T


def test_fit_ellipse_recovers_center():
    """
    Validate that fit_ellipse_to_points correctly estimates
    the center of a well-defined ellipse.
    """
    true_center = (120.0, 100.0)

    points = generate_ellipse_points(center=true_center)

    center_x, center_y, a, b, theta = fit_ellipse_to_points(points)

    estimated_center = np.array([center_x, center_y])

    assert np.allclose(
        estimated_center,
        true_center,
        atol=2.0,
    ), f"Estimated center {estimated_center} deviates from true center {true_center}"


def test_fit_ellipse_circle_case_center():
    """
    Edge case: circular ellipse (a == b).
    """
    true_center = (80.0, 150.0)

    points = generate_ellipse_points(
        center=true_center,
        axes=(30.0, 30.0),
        noise_std=0.0,
    )

    center_x, center_y, a, b, theta = fit_ellipse_to_points(points)

    assert np.allclose(
        [center_x, center_y],
        true_center,
        atol=1e-1,
    )
