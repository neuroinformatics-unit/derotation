import numpy as np
import pytest
from PIL import Image

from derotation.analysis.derotation_pipeline import DerotationPipeline

lenna = Image.open("tests/test_regression/images/lenna.png").convert("L")


@pytest.fixture
def len_stack():
    return 10


@pytest.fixture
def image_stack(len_stack):
    image_stack = np.array([np.array(lenna) for _ in range(len_stack)])
    return image_stack


@pytest.fixture
def n_lines(image_stack):
    return image_stack.shape[1]


@pytest.fixture
def n_total_lines(image_stack, n_lines):
    return image_stack.shape[0] * n_lines


def get_angles(kind, n_lines, n_total_lines):
    if kind == "uniform":
        rotation = np.linspace(0, 360, n_total_lines - n_lines)
    elif kind == "sinusoidal":
        rotation = (
            np.sin(np.linspace(0, 2 * np.pi, n_total_lines - n_lines)) * 360
        )
    all_angles = np.zeros(n_total_lines)
    all_angles[n_lines // 2 : -n_lines // 2] = rotation

    return all_angles


def test_rotation_by_line(image_stack, n_lines, n_total_lines, len_stack):
    pipeline = DerotationPipeline.__new__(DerotationPipeline)
    pipeline.image_stack = image_stack

    for kind in ["uniform", "sinusoidal"]:
        pipeline.rot_deg_line = get_angles(kind, n_lines, n_total_lines)
        pipeline.num_lines_per_frame = n_lines

        rotated_images = pipeline.rotate_frames_line_by_line()

        assert len(rotated_images) == len_stack
        assert rotated_images[0].shape == (n_lines, n_lines)

        for i, image in enumerate(rotated_images):
            target_image = Image.open(
                "tests/test_regression/images/"
                + f"{kind}_rotation/rotated_lenna_{i + 1}.png"
            )
            target_image = np.array(target_image.convert("L"))
            assert np.allclose(
                image, target_image, atol=1
            ), f"Failed for {kind} rotation, image {i + 1}"
