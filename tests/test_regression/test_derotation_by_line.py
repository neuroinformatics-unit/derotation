from pathlib import Path

import numpy as np
import pytest
from assertions import compare_images
from PIL import Image

from derotation.analysis.full_derotation_pipeline import FullPipeline
from tests.test_regression.recreate_target.derotation_by_line import (
    get_angles,
    get_image_stack,
    get_len_stack,
    get_n_lines,
    get_n_total_lines,
)

#  These fixtures are used only in this module. They aim to recreate the
#  conditions used to generate the target images.


@pytest.fixture
def image_stack(len_stack):
    return get_image_stack(len_stack)


@pytest.fixture
def len_stack():
    return get_len_stack()


@pytest.fixture
def n_lines(image_stack):
    return get_n_lines(image_stack)


@pytest.fixture
def n_total_lines(n_lines, image_stack):
    return get_n_total_lines(image_stack, n_lines)


def test_derotation_by_line(n_lines, n_total_lines, len_stack, image_stack):
    pipeline = FullPipeline.__new__(FullPipeline)
    pipeline.image_stack = image_stack

    for kind in ["uniform", "sinusoidal"]:
        pipeline.rot_deg_line = get_angles(kind, n_lines, n_total_lines)
        pipeline.num_lines_per_frame = n_lines
        pipeline.center_of_rotation = (n_lines // 2, n_lines // 2)
        pipeline.hooks = {}
        pipeline.debugging_plots = False

        derotated_images = pipeline.derotate_frames_line_by_line()

        assert len(derotated_images) == len_stack
        assert derotated_images[0].shape == (n_lines, n_lines)

        for i, image in enumerate(derotated_images):
            target_image = Image.open(
                "tests/test_regression/images/"
                + f"{kind}_rotation/rotated_dog_{i + 1}.png"
            )
            target_image = np.array(target_image.convert("L"))

            compare_images(
                i,
                image,
                target_image,
                1,
                Path("tests/test_regression/images/"),
            )
