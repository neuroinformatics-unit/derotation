"""
Unit tests for derotation.analysis.fit_ellipse.fit_ellipse_to_points

Resolves GitHub issue #36 by validating center, eccentricity, and orientation.

Covers the three parameters specified in the GitHub issue:
  1. Center (cx, cy)       -- primary concern
  2. Eccentricity          -- derived from semi-axes a, b
  3. Orientation (theta)   -- rotation angle of the fitted ellipse

All tests are deterministic via a seeded rng fixture.
"""

import numpy as np
import pytest

from derotation.analysis.fit_ellipse import fit_ellipse_to_points


# ============================================================================
# HELPERS
# ============================================================================

def generate_ellipse_points(
    center=(120.0, 100.0),
    axes=(40.0, 20.0),
    angle=np.pi / 6,
    n_points=200,
    noise_std=0.0,
    rng=None,
):
    """
    Return (n_points, 2) array of points on a rotated, optionally noisy ellipse.
    """
    cx, cy = center
    a, b = axes
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    x = a * np.cos(t)
    y = b * np.sin(t)

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)],
    ])
    pts = R @ np.vstack((x, y))
    pts[0] += cx
    pts[1] += cy

    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        pts += rng.normal(scale=noise_std, size=pts.shape)

    return pts.T


def eccentricity(a, b):
    """Return eccentricity in [0, 1) from semi-axes; order does not matter."""
    major, minor = max(a, b), min(a, b)
    return np.sqrt(1.0 - (minor / major) ** 2)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def rng():
    """Seeded RNG for deterministic noise in tests."""
    return np.random.default_rng(42)


# ============================================================================
# 1. CENTER RECOVERY
# ============================================================================

@pytest.mark.parametrize("center, axes, angle, noise_std, atol", [
    ((120.0, 100.0), (40.0, 20.0), np.pi / 6, 0.5, 2.0),  # noisy rotated ellipse
    ((128.0, 128.0), (50.0, 50.0), 0.0,       0.0, 0.5),  # circle, no noise
    ((80.0,  150.0), (30.0, 30.0), 0.0,       0.0, 0.5),  # circle, different center
    ((200.0, 150.0), (80.0, 40.0), 0.0,       0.0, 0.5),  # horizontal ellipse
    ((0.0,     0.0), (40.0, 20.0), np.pi / 4, 0.0, 0.5),  # origin-centered
    ((512.0, 512.0), (60.0, 30.0), np.pi / 3, 0.0, 0.5),  # large-scale coordinates
])
def test_center_recovery(center, axes, angle, noise_std, atol, rng):
    """Verify center recovery across positions, axes, angles, and noise levels."""
    points = generate_ellipse_points(
        center=center, axes=axes, angle=angle, noise_std=noise_std, rng=rng,
    )
    cx, cy, _, _, _ = fit_ellipse_to_points(points)

    assert np.allclose([cx, cy], center, atol=atol), (
        f"Center mismatch: got ({cx:.3f}, {cy:.3f}), expected {center}"
    )


def test_center_recovery_with_nan_values(rng):
    """Ensure NaN rows are ignored when enough valid points remain."""
    true_center = (120.0, 100.0)
    points = generate_ellipse_points(
        center=true_center,
        axes=(40.0, 20.0),
        angle=np.pi / 6,
        noise_std=0.5,
        rng=rng,
    )
    # Inject NaN rows to simulate missing blob detections
    points[0] = np.nan
    points[10] = np.nan
    points[50] = np.nan

    cx, cy, _, _, _ = fit_ellipse_to_points(points)

    assert np.allclose([cx, cy], true_center, atol=2.0), (
        f"Center with NaNs: got ({cx:.3f}, {cy:.3f}), expected {true_center}"
    )


# ============================================================================
# 2. ECCENTRICITY
# ============================================================================

@pytest.mark.parametrize("axes, expected_ecc, rtol", [
    ((50.0, 50.0), 0.0,                        0.05),  # circle -> ecc ≈ 0
    ((40.0, 20.0), eccentricity(40.0, 20.0),   0.05),  # moderate elongation
    ((80.0, 20.0), eccentricity(80.0, 20.0),   0.05),  # high elongation
])
def test_eccentricity_recovery(axes, expected_ecc, rtol):
    """Verify eccentricity derived from fitted axes matches the ground truth."""
    points = generate_ellipse_points(
        center=(100.0, 100.0), axes=axes, angle=0.0, noise_std=0.0,
    )
    _, _, a, b, _ = fit_ellipse_to_points(points)

    fitted_ecc = eccentricity(a, b)
    assert np.isclose(fitted_ecc, expected_ecc, rtol=rtol, atol=0.02), (
        f"Eccentricity mismatch: got {fitted_ecc:.4f}, "
        f"expected {expected_ecc:.4f} for axes={axes}"
    )


def test_circle_has_near_zero_eccentricity():
    """A perfect circle must produce eccentricity close to zero."""
    points = generate_ellipse_points(
        center=(128.0, 128.0), axes=(50.0, 50.0), noise_std=0.0,
    )
    _, _, a, b, _ = fit_ellipse_to_points(points)

    ecc = eccentricity(a, b)
    assert ecc < 0.05, (
        f"Circle eccentricity should be ≈ 0, got {ecc:.4f}"
    )


# ============================================================================
# 3. ORIENTATION
# ============================================================================

@pytest.mark.parametrize("true_angle", [0.0, np.pi / 6, np.pi / 4, np.pi / 3])
def test_orientation_recovery(true_angle):
    """Verify fitted angle matches ground truth modulo pi/2 (ellipse symmetry)."""
    points = generate_ellipse_points(
        center=(100.0, 100.0),
        axes=(60.0, 20.0),
        angle=true_angle,
        noise_std=0.0,
    )
    _, _, _, _, theta = fit_ellipse_to_points(points)

    # Ellipse orientation is defined modulo pi/2
    diff = abs(theta - true_angle) % (np.pi / 2)
    diff = min(diff, np.pi / 2 - diff)

    assert diff < 0.1, (
        f"Angle mismatch: got {theta:.4f} rad, "
        f"expected {true_angle:.4f} rad, diff={diff:.4f} rad"
    )


def test_angle_within_valid_range():
    """Fitted angle theta must lie within [-pi/2, pi/2]."""
    points = generate_ellipse_points(angle=np.pi / 5)
    _, _, _, _, theta = fit_ellipse_to_points(points)

    assert -np.pi / 2 <= theta <= np.pi / 2, (
        f"Angle {theta:.4f} rad is outside valid range [-pi/2, pi/2]"
    )


# ============================================================================
# 4. AXES RECOVERY
# ============================================================================

@pytest.mark.parametrize("axes", [(40.0, 20.0), (80.0, 40.0), (50.0, 50.0)])
def test_axes_recovery(axes):
    """Semi-axes are recovered within 5% rtol; order-independent comparison."""
    points = generate_ellipse_points(
        center=(100.0, 100.0), axes=axes, angle=0.0, noise_std=0.0,
    )
    _, _, a, b, _ = fit_ellipse_to_points(points)

    assert np.allclose(sorted([a, b]), sorted(axes), rtol=0.05), (
        f"Axes mismatch: got {sorted([a, b])}, expected {sorted(axes)}"
    )


# ============================================================================
# 5. EDGE CASES
# ============================================================================

def test_nan_rows_still_return_valid_axes():
    """NaN rows are filtered out; remaining valid points produce positive axes."""
    points = np.array([
        [100.0, 100.0],
        [150.0, 100.0],
        [np.nan, np.nan],
        [150.0, 150.0],
        [100.0, 150.0],
        [125.0, 125.0],
        [np.nan, 200.0],
    ])
    cx, cy, a, b, _ = fit_ellipse_to_points(points)

    assert not np.isnan(cx), f"Expected cx to be finite, got {cx}"
    assert not np.isnan(cy), f"Expected cy to be finite, got {cy}"
    assert a > 0, f"Expected a > 0, got {a}"
    assert b > 0, f"Expected b > 0, got {b}"


def test_minimum_five_points():
    """Function succeeds with exactly 5 points and returns positive axes."""
    points = np.array([
        [100.0, 100.0],
        [150.0, 100.0],
        [150.0, 150.0],
        [100.0, 150.0],
        [125.0, 125.0],
    ])
    cx, cy, a, b, theta = fit_ellipse_to_points(points)

    assert isinstance(cx, (int, float, np.floating))
    assert isinstance(cy, (int, float, np.floating))
    assert a > 0, f"Expected a > 0, got {a}"
    assert b > 0, f"Expected b > 0, got {b}"
    assert np.isfinite(theta), f"Expected finite theta, got {theta}"


def test_too_few_points_raises_value_error():
    """Fewer than 5 points raises ValueError with 'Not enough valid points'."""
    points = np.array([
        [100.0, 100.0],
        [150.0, 100.0],
        [150.0, 150.0],
        [100.0, 150.0],
    ])
    with pytest.raises(ValueError, match="Not enough valid points"):
        fit_ellipse_to_points(points)


def test_noisy_ellipse_center_within_tolerance(rng):
    """With noise_std=1.0 center estimate stays within atol=3.0 of truth."""
    true_center = (120.0, 100.0)
    points = generate_ellipse_points(
        center=true_center,
        axes=(40.0, 20.0),
        angle=np.pi / 6,
        noise_std=1.0,
        rng=rng,
    )
    cx, cy, _, _, _ = fit_ellipse_to_points(points)

    assert np.allclose([cx, cy], true_center, atol=3.0), (
        f"Noisy center: got ({cx:.3f}, {cy:.3f}), expected {true_center}"
    )


# ============================================================================
# 6. OUTPUT VALIDITY
# ============================================================================

def test_returns_five_numeric_values():
    """Function always returns exactly 5 numeric scalars."""
    result = fit_ellipse_to_points(generate_ellipse_points())

    assert len(result) == 5, f"Expected 5 values, got {len(result)}"
    for i, val in enumerate(result):
        assert isinstance(val, (int, float, np.floating)), (
            f"Return value [{i}] is not numeric: {type(val)}"
        )


def test_axes_are_strictly_positive():
    """Semi-axes a and b must always be strictly positive."""
    _, _, a, b, _ = fit_ellipse_to_points(generate_ellipse_points())

    assert a > 0, f"Expected a > 0, got {a}"
    assert b > 0, f"Expected b > 0, got {b}"

