#  -----------------------------------------------------
#  This test is an integration test that combines the incremental derotation
#  pipeline with the full derotation pipeline. The test generates a test image
#  stack with two circles and rotates it incrementally and sinusoidally using
#  a variable center of rotation.
#
#  The image looks like this:
#
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ███████████████████▒▒▓██████████████████
#     ██████████████████▒  ░██████████████████
#     ██████████████████▓░░▒██████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ███████████████████████████████▓▓███████
#     ██████████████████████████████▓▒▒▓██████
#     ███████████████████████████████▓▓▓██████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#
#  The center of rotation is then estimated by the incremental pipeline and
#  used to derotate the sinusoidal stack. The test checks if the two circles
#  are detected in the derotated stack. The test is parametrized with two
#  different initial centers of rotation.
#
#  This test can also be run as a standalone script to generate debugging
#  plots.
#
#  Sections:
#  1. Prepare the 3D image stack and the rotation angles
#  2. Integration pipeline with mock of the IncrementalPipeline
#  3. Debugging plots
#  4. Test the integration pipeline
#  5. Run the integration pipeline as a standalone script
#  -----------------------------------------------------


from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from skimage.feature import blob_log

from derotation.simulate.synthetic_data import SyntheticData

# -----------------------------------------------------
# Test the integration pipeline
# -----------------------------------------------------


@pytest.mark.parametrize("center_of_rotation_initial", [(44, 51), (51, 44)])
def test_derotation_with_shifted_center(
    center_of_rotation_initial: Tuple[int, int]
):
    """Test if the two circles are detected in the derotated stack
    at the expected locations when the center of rotation is estimated
    with the incremental pipeline. It is parametrized with two different
    initial centers of rotation.

    It allows some error tolerance in the center of the detected blobs,
    precisely 6 pixels in any direction. This is because the derotation
    itself would loose some information and the reconstructed blobs might
    miss pixels and consequently be detected at slightly different locations.
    Test will fail with more than 5% errors.

    Parameters
    ----------
    center_of_rotation_initial : Tuple[int, int]
        Center of rotation
    """

    # -----------------------------------------------------
    # Setting up the test

    # Create a test image with two circles
    s_data = SyntheticData()
    test_image = s_data.create_sample_image_with_two_cells()

    # Run the integration pipeline and obtain the derotated stack
    derotated_sinusoidal = s_data.integration_pipeline(test_image)

    assert_blob_detection(
        derotated_sinusoidal,
        s_data.center_of_bright_cell,
        s_data.center_of_dimmer_cell,
        center_of_rotation_initial,
    )


@pytest.mark.parametrize(
    "center_of_rotation_offset, rotation_plane_angle, rotation_plane_orientation",
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
    center_of_rotation_offset,
    rotation_plane_angle,
    rotation_plane_orientation,
):
    s_data = SyntheticData(
        radius=3,
        center_of_rotation_offset=center_of_rotation_offset,
        rotation_plane_angle=rotation_plane_angle,
        rotation_plane_orientation=rotation_plane_orientation,
        num_frames=50,
        plots=True,
    )

    image = s_data.create_sample_image_with_two_cells()
    pad = 20
    image = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")
    image[image == 0] = 80
    s_data.lines_per_frame = image.shape[0]
    derotated_sinusoidal = s_data.integration_pipeline(image)

    #  plot mean projection
    plt.close()
    mean_projection = np.mean(derotated_sinusoidal, axis=0)
    fig, ax = plt.subplots()
    ax.imshow(mean_projection, cmap="gray")
    plt.savefig(
        f"debug/mean_projection_{center_of_rotation_offset}_{rotation_plane_angle}_{rotation_plane_orientation}.png"
    )
    plt.close()

    # assert_blob_detection(
    #     derotated_sinusoidal,
    #     s_data.center_of_bright_cell,
    #     s_data.center_of_dimmer_cell,
    #     s_data.center_of_rotation_offset,
    # )


def assert_blob_detection(
    derotated_sinusoidal: np.ndarray,
    center_1: Tuple[int, int],
    center_2: Tuple[int, int],
    center_of_rotation_offset: Tuple[int, int],
):
    # -----------------------------------------------------
    # Are the blobs detected in the derotated stack where we expect them?
    # If yes, the derotation was successful and the test passes
    # If not, the derotation was not successful and the test fails

    # Detect the blobs in the derotated stack
    blobs = [
        blob_log(img, min_sigma=3, max_sigma=5) for img in derotated_sinusoidal
    ]

    # Get the center of the blobs
    # for every frame, place first the blob with the smallest x value
    blobs = [sorted(blob, key=lambda x: x[1]) for blob in blobs]

    # Compare the first and second blob to the expected values
    errors = 0
    atol = 6  # 6 pixels tolerance for the center of the blobs
    for blob in blobs:
        if not np.allclose(blob[0][:2][::-1], center_1, atol=atol):
            errors += 1
        if len(blob) > 1 and not np.allclose(
            blob[-1][:2][::-1], center_2, atol=atol, rtol=0
        ):
            errors += 1

    # we do not expect more than 5% errors
    # (there are 100 frames and 2 blobs per frame)
    assert errors < derotated_sinusoidal.shape[0] * 2 * 0.05, (
        f"More than 5% errors ({errors}) in derotation "
        + f" with center offset {center_of_rotation_offset}"
    )


# -----------------------------------------------------
# Run the integration pipeline as a standalone script
# to generate debugging plots
# -----------------------------------------------------

if __name__ == "__main__":
    Path("debug/").mkdir(parents=True, exist_ok=True)
    # test_derotation_with_shifted_center((44, 51))
    test_derotation_with_rotation_out_of_plane((0, 0), 45, 0)
