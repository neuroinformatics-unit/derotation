import numpy as np
from PIL import Image

from derotation.analysis.full_derotation_pipeline import FullPipeline

dog = Image.open("images/dog.png").convert("L")


def get_len_stack():
    return 10


def get_image_stack(len_stack):
    image_stack = np.array([np.array(dog) for _ in range(len_stack)])
    return image_stack


def get_n_lines(image_stack):
    return image_stack.shape[1]


def get_n_total_lines(image_stack, n_lines):
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


def regenerate_images_for_testing(image_stack, n_lines, n_total_lines):
    pipeline = FullPipeline.__new__(FullPipeline)
    pipeline.image_stack = image_stack

    for kind in ["uniform", "sinusoidal"]:
        pipeline.rot_deg_line = get_angles(kind, n_lines, n_total_lines)
        pipeline.num_lines_per_frame = n_lines
        pipeline.center_of_rotation = (n_lines // 2, n_lines // 2)
        pipeline.debugging_plots = False
        pipeline.hooks = {}

        derotated_images = pipeline.derotate_frames_line_by_line()

        for i, image in enumerate(derotated_images):
            image = Image.fromarray(image.astype("uint8"))
            image.save(
                "tests/test_regression/images/"
                + f"{kind}_rotation/rotated_dog_{i + 1}.png"
            )


if __name__ == "__main__":
    stack_len = 10
    stack = get_image_stack(stack_len)
    lines_n = stack.shape[1]
    total_lines = stack.shape[0] * lines_n

    regenerate_images_for_testing(stack, lines_n, total_lines)
