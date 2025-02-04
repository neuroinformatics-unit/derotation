from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from derotation.analysis.blob_detection import BlobDetection
from derotation.simulate.synthetic_data import SyntheticData

# -----------------------------------------------------
# Test the integration pipeline
# -----------------------------------------------------


@pytest.mark.parametrize("center_of_rotation_initial", [(44, 51), (51, 44)])
def test_derotation_with_shifted_center(
    center_of_rotation_initial: Tuple[int, int],
):
    """Test if the two circles are detected in the derotated stack
    at the expected locations when the center of rotation is estimated
    with the incremental pipeline. It is parametrized with two different
    initial centers of rotation.

    It allows some error tolerance in the center of the detected blobs,
    precisely 5% of the image length. This is because the derotation
    itself would loose some information and the reconstructed blobs might
    miss pixels and consequently be detected at slightly different locations.
    Test will fail with more than 5% errors.

    Parameters
    ----------
    center_of_rotation_initial : Tuple[int, int]
        Center of rotation
    """

    # Create a test image with two circles
    s_data = SyntheticData()

    # Run the integration pipeline and obtain the derotated stack
    s_data.integration_pipeline()

    assert_blob_detection(
        s_data.derotated_sinusoidal,
        s_data.center_of_bright_cell,
        center_of_rotation_initial,
    )


@pytest.mark.parametrize(
    "center_of_rotation_offset, "
    "rotation_plane_angle, "
    "rotation_plane_orientation",
    [
        ((0, 0), 0, 0),  # null case
        ((0, 0), 0, 5),  # null case
        ((0, 0), 5, 0),
        ((0, 0), 10, 0),
        ((0, 0), 20, 0),
        ((0, 0), 20, 10),
        ((0, 0), 20, 20),
        ((0, 0), 20, 30),
        ((0, 0), 20, 45),
        ((0, 0), 20, 90),
        ((0, 0), 30, 0),
        ((0, 0), 45, 0),
        ((0, 0), 5, 5),
        ((0, 0), 5, 10),
        ((0, 0), 10, 5),
        ((0, 0), 25, 0),
        ((0, 0), 25, 90),
        ((-6, 1), 0, 0),  # null case
        ((-6, 1), 0, 5),  # null case
        ((-6, 1), 5, 0),
        ((-6, 1), 5, 5),
        ((-6, 1), 5, 10),
        ((-6, 1), 10, 5),
        ((-6, 1), 25, 90),
        ((1, -6), 0, 0),  # null case
        ((1, -6), 0, 5),  # null case
        ((1, -6), 5, 0),
        ((1, -6), 5, 5),
        ((1, -6), 5, 10),
        ((1, -6), 10, 5),
        ((1, -6), 25, 90),
    ],
)
def test_derotation_with_rotation_out_of_plane(
    center_of_rotation_offset: Tuple[int, int],
    rotation_plane_angle: int,
    rotation_plane_orientation: int,
    plots: bool = False,
):
    """Test if the two circles are detected in the derotated stack
    at the expected locations when the center of rotation is estimated
    with the incremental pipeline. It is parametrized with two different
    initial centers of rotation.

    Parameters
    ----------
    center_of_rotation_offset : Tuple[int, int]
        Center of rotation offset
    rotation_plane_angle : int
        The angle of the rotation plane
    rotation_plane_orientation : int
        The orientation of the rotation plane
    """
    s_data = SyntheticData(
        radius=5,
        center_of_rotation_offset=center_of_rotation_offset,
        rotation_plane_angle=rotation_plane_angle,
        rotation_plane_orientation=rotation_plane_orientation,
        num_frames=50,
        pad=20,
        background_value=80,
        plots=plots,
    )
    s_data.integration_pipeline()

    assert_blob_detection(
        s_data.derotated_sinusoidal,
        s_data.center_of_bright_cell,
        s_data.center_of_rotation_offset,
    )


def assert_blob_detection(
    derotated_sinusoidal: np.ndarray,
    center_1: Tuple[int, int],
    center_of_rotation_offset: Tuple[int, int],
):
    """Assert that the brightest blob is detected in the derotated stack
    at the expected location. It allows some error tolerance in the center
    of the detected blobs, precisely 5% of the image length. This is because
    the derotation itself would loose some information and the reconstructed
    blobs might miss pixels and consequently be detected at slightly different
    locations. Test will fail with more than 5% errors.

    Parameters
    ----------
    derotated_sinusoidal : np.ndarray
        The derotated sinusoidal stack
    center_1 : Tuple[int, int]
        The center of the first circle
    center_of_rotation_offset : Tuple[int, int]
        The center of rotation offset

    Raises
    ------
    AssertionError
        If the detected blobs are not at the expected locations
    """

    # Detect the blobs in the derotated stack
    coords = BlobDetection(
        blob_log_params={"min_sigma": 3, "max_sigma": 7, "threshold_rel": 0.8},
        debug_plots_folder=Path("debug"),
        debugging_plots=True,
    ).get_coords_of_largest_blob(derotated_sinusoidal)

    # Compare the first and second blob to the expected values
    errors = 0
    # 5% pixels tolerance for the center of the blobs from image length
    atol = int(derotated_sinusoidal.shape[1] * 0.05)

    for coord in coords:
        if not np.allclose(coord, center_1, atol=atol):
            errors += 1

    # we do not want errors in more than 5% of the frames
    assert errors < derotated_sinusoidal.shape[0] * 0.05, (
        f"More than 5% errors ({errors}) in derotation "
        + f" with center offset {center_of_rotation_offset}"
    )


# -----------------------------------------------------
# Run the integration pipeline as a standalone script
# to generate debugging plots
# -----------------------------------------------------

if __name__ == "__main__":
    Path("debug/").mkdir(parents=True, exist_ok=True)
    test_derotation_with_rotation_out_of_plane((0, 0), 20, 45, plots=True)
