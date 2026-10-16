import logging

import numpy as np

from derotation.analysis.mean_images import calculate_mean_images


def test_mean_images_skip_angles_with_no_frames(caplog):
    rng = np.random.default_rng(0)
    image_stack = rng.random((181, 8, 8))
    # the rotation only covers 0 to 180 degrees
    rot_deg_frame = np.arange(0, 181, 1.0)

    with caplog.at_level(logging.WARNING):
        mean_images = calculate_mean_images(image_stack, rot_deg_frame)

    # one mean image for each of 10, 20, ..., 180 degrees
    assert mean_images.shape == (18, 8, 8)
    assert not np.isnan(mean_images).any()
    np.testing.assert_allclose(mean_images[0], image_stack[10])
    assert "Angle 190 not found in the image stack" in caplog.text
