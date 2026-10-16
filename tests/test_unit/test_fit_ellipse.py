"""Unit tests for fit_ellipse_to_points."""

import numpy as np

from derotation.analysis.fit_ellipse import fit_ellipse_to_points


def test_fit_ellipse_center_outside_the_square_of_the_height():
    """On a wide frame, the center x can be larger than the frame height."""
    width, height = 200, 80
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    centers = np.stack(
        [150 + 20 * np.cos(theta), 40 + 20 * np.sin(theta)], axis=1
    )

    center_x, center_y, a, b, _ = fit_ellipse_to_points(
        centers, pixels_in_row=width, num_rows=height
    )

    assert abs(center_x - 150) < 1
    assert abs(center_y - 40) < 1
    assert abs(a - 20) < 1 and abs(b - 20) < 1
