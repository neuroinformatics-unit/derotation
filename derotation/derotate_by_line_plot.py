import copy

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import rotate


def create_gradient_cmap(hue, name="custom_cmap"):
    cdict = {
        "red": [(0.0, hue[0], hue[0]), (1.0, 1.0, 1.0)],
        "green": [(0.0, hue[1], hue[1]), (1.0, 1.0, 1.0)],
        "blue": [(0.0, hue[2], hue[2]), (1.0, 1.0, 1.0)],
        "alpha": [
            (0.0, 0.0, 0.0),  # Make NaNs transparent
            (0.01, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
    }
    return LinearSegmentedColormap(name, cdict)


# Create a list of hues for each row (following a rainbow scheme)
def generate_hues(n):
    # Create a gradient from purple (128, 0, 128) to green (0, 128, 0)
    start_color = np.array([70, 9, 91]) / 255  # Starting color
    end_color = np.array([33, 165, 133]) / 255  # Ending color

    hues = np.linspace(start_color, end_color, n)
    return hues


# Generate 256 hues
hues = generate_hues(256)


# def generate_rainbow_colormap():
#     cdict = {
#         'red':   [(0.0, 1.0, 1.0),  # Red
#                   (0.17, 1.0, 1.0), # Orange
#                   (0.33, 1.0, 1.0), # Yellow
#                   (0.5, 0.0, 0.0),  # Green
#                   (0.67, 0.0, 0.0), # Blue
#                   (0.83, 0.5, 0.5), # Indigo
#                   (1.0, 0.56, 0.56)], # Violet
#         'green': [(0.0, 0.0, 0.0),  # Red
#                   (0.17, 0.5, 0.5), # Orange
#                   (0.33, 1.0, 1.0), # Yellow
#                   (0.5, 1.0, 1.0),  # Green
#                   (0.67, 0.0, 0.0), # Blue
#                   (0.83, 0.0, 0.0), # Indigo
#                   (1.0, 0.37, 0.37)], # Violet
#         'blue':  [(0.0, 0.0, 0.0),  # Red
#                   (0.17, 0.0, 0.0), # Orange
#                   (0.33, 0.0, 0.0), # Yellow
#                   (0.5, 0.0, 0.0),  # Green
#                   (0.67, 1.0, 1.0), # Blue
#                   (0.83, 0.5, 0.5), # Indigo
#                   (1.0, 0.68, 0.68)] # Violet
#     }
#     return LinearSegmentedColormap('rainbow', cdict)

# def angle_to_rgb(angle, colormap):
#     normalized_angle = angle / 360
#     return colormap(normalized_angle)

# # Create the rainbow colormap
# rainbow_cmap = generate_rainbow_colormap()

# # Function to create a gradient colormap from a base hue
# def create_gradient_cmap(hue, name='custom_cmap'):
#     cdict = {
#         'red':   [(0.0, hue[0], hue[0]),
#                   (1.0, 1.0, 1.0)],
#         'green': [(0.0, hue[1], hue[1]),
#                   (1.0, 1.0, 1.0)],
#         'blue':  [(0.0, hue[2], hue[2]),
#                   (1.0, 1.0, 1.0)],
#         'alpha': [(0.0, 0.0, 0.0),  # Make NaNs transparent
#                   (0.01, 1.0, 1.0),
#                   (1.0, 1.0, 1.0)]
#     }
#     return LinearSegmentedColormap(name, cdict)


def rotate_an_image_array_line_by_line(
    image_stack: np.ndarray,
    rot_deg_line: np.ndarray,
    blank_pixels_value: float = 0,
) -> np.ndarray:
    """Rotates the image stack line by line, using the rotation angles
    provided.

    Description of the algorithm:
    - takes one line from the image stack
    - creates a new image with only that line
    - rotates the line by the given angle
    - substitutes the line in the new image
    - adds the new image to the rotated image stack

    Edge cases and how they are handled:
    - the rotation starts in the middle of the image -> the previous lines
    are copied from the first frame
    - the rotation ends in the middle of the image -> the remaining lines
    are copied from the last frame

    Parameters
    ----------
    image_stack : np.ndarray
        The image stack to be rotated.
    rot_deg_line : np.ndarray
        The rotation angles by line.

    Returns
    -------
    np.ndarray
        The rotated image stack.
    """

    num_lines_per_frame = image_stack.shape[1]
    rotated_image_stack = copy.deepcopy(image_stack)
    previous_image_completed = True
    rotation_completed = True

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.zeros_like(image_stack[0]), cmap="gray", vmin=0, vmax=1)

    for i, rotation in tqdm.tqdm(
        enumerate(rot_deg_line), total=len(rot_deg_line)
    ):
        line_counter = i % num_lines_per_frame
        image_counter = i // num_lines_per_frame

        is_rotating = np.absolute(rotation) > 0.00001
        image_scanning_completed = line_counter == (num_lines_per_frame - 1)
        if i == 0:
            rotation_just_finished = False
        else:
            rotation_just_finished = not is_rotating and (
                np.absolute(rot_deg_line[i - 1]) > np.absolute(rotation)
            )

        if is_rotating:
            if rotation_completed and (line_counter != 0):
                # fig, ax = plt.subplots(figsize=(10, 10))
                # ax.imshow(np.zeros_like(image_stack[0]), cmap='gray', vmin=0, vmax=1)

                # when starting a new rotation in the middle of the image
                rotated_filled_image = (
                    np.ones_like(image_stack[image_counter])
                    * blank_pixels_value
                )  # non sampled pixels are set to the min val of the image
                rotated_filled_image[:line_counter] = image_stack[
                    image_counter
                ][:line_counter]
            elif previous_image_completed:
                # fig, ax = plt.subplots(figsize=(10, 10))
                # ax.imshow(np.zeros_like(image_stack[0]), cmap='gray', vmin=0, vmax=1)

                rotated_filled_image = (
                    np.ones_like(image_stack[image_counter])
                    * blank_pixels_value
                )

            rotation_completed = False

            img_with_new_lines = image_stack[image_counter]
            line = img_with_new_lines[line_counter]

            image_with_only_line = np.zeros_like(img_with_new_lines)
            image_with_only_line[line_counter] = line

            rotated_line = rotate(
                image_with_only_line,
                rotation,
                reshape=False,
                order=0,
                mode="constant",
            )

            # if image_counter >= 122 and image_counter <= 143:
            if image_counter == 128:
                # hue = angle_to_rgb(rotation, rainbow_cmap)[:3]
                hue = hues[int((rotation + 180) * 255 / 360)]
                gradient_cmap = create_gradient_cmap(hue, name=f"row_cmap_{i}")
                ax.imshow(rotated_line, aspect="auto", cmap=gradient_cmap)

            rotated_filled_image = np.where(
                rotated_line == 0, rotated_filled_image, rotated_line
            )
            previous_image_completed = False
        if (
            image_scanning_completed and not rotation_completed
        ) or rotation_just_finished:
            if rotation_just_finished:
                rotation_completed = True

                rotated_filled_image[line_counter + 1 :] = image_stack[
                    image_counter
                ][line_counter + 1 :]

            # if image_counter >= 122 and image_counter <= 143:
            if image_counter == 128:
                ax.axis("off")
                plt.savefig(
                    f"examples/figures/pdf/rotate_{image_counter}.pdf", dpi=400
                )
                plt.close()

            rotated_image_stack[image_counter] = rotated_filled_image
            previous_image_completed = True

    return rotated_image_stack
