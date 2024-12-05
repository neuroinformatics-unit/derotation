import numpy as np
from matplotlib import pyplot as plt
from test_finding_center_of_rotation_by_joining_two_pipelines import (
    create_image_stack,
    create_rotation_angles,
    create_sample_image_with_two_cells,
)

from derotation.simulate.line_scanning_microscope import Rotator


def setup():
    plane_angle = 25
    num_frames = 50
    pad = 20

    #  create a sample image with two cells
    cells = create_sample_image_with_two_cells(lines_per_frame=100)
    cells = np.pad(cells, ((pad, pad), (pad, pad)), mode="constant")
    cells[cells == 0] = 80

    image_stack = create_image_stack(cells, num_frames=num_frames)
    print("Image stack shape:", image_stack.shape)

    _, angles = create_rotation_angles(image_stack.shape)
    print("Angles shape:", angles.shape)

    #  rotate the image stack
    rotator = Rotator(angles, image_stack, rotation_plane_angle=plane_angle)
    rotated_image_stack = rotator.rotate_by_line()

    print("Rotated image stack shape:", rotated_image_stack.shape)

    return image_stack, rotated_image_stack, rotator, num_frames


#  plot the original image, the rotated images
def make_plot(image_stack, rotated_image_stack, rotator, num_frames):
    row_n = 5
    fig, ax = plt.subplots(row_n, num_frames // row_n + 1, figsize=(40, 25))

    print(ax.shape)
    ax[0, 0].imshow(image_stack[0], cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Original image")
    ax[0, 0].axis("off")

    for n in range(1, len(rotated_image_stack) + 1):
        row = n % row_n
        col = n // row_n
        ax[row, col].imshow(
            rotated_image_stack[n - 1], cmap="gray", vmin=0, vmax=255
        )
        ax[row, col].set_title(n)

        # add information about the rotation angle
        angles = rotator.angles[n - 1]
        angle_range = f"{angles.min():.0f}-{angles.max():.0f}"
        ax[row, col].text(
            0.5,
            0.9,
            angle_range,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[row, col].transAxes,
            color="white",
        )

    #  last one, max projection
    ax[row_n - 1, num_frames // row_n].imshow(
        rotated_image_stack.max(axis=0), cmap="gray"
    )
    #  plot also an x in the center of the image in red
    ax[row_n - 1, num_frames // row_n].plot(
        rotated_image_stack.shape[2] / 2,
        rotated_image_stack.shape[1] / 2,
        "rx",
        markersize=10,
    )
    ax[row_n - 1, num_frames // row_n].set_title("Max projection")

    for a in ax.ravel():
        a.axis("off")

    #  save in debug folder
    plt.savefig("debug/rotation_out_of_plane.png")


# def test_max_projection_with_rotation_out_of_plane():
#     _, rotated_image_stack, *_ = setup()

#     max_projection = rotated_image_stack.max(axis=0)
#     edges = canny(max_projection, sigma=7)

#     fig, ax = plt.subplots(1, 3, figsize=(10, 5))
#     ax[0].imshow(max_projection, cmap="gray")
#     ax[0].set_title("Max projection")
#     ax[0].axis("off")

#     #  plot edges
#     ax[1].imshow(edges, cmap="gray")
#     ax[1].set_title("Edges")
#     ax[1].axis("off")

#     result = hough_ellipse(edges, accuracy=100, threshold=1,
#                            min_size=0, max_size=200)
#     result.sort(order="accumulator")

#     #  get the best ellipse
#     best = result[-1]
#     yc, xc, a, b = [int(x) for x in best[1:5]]

#     #  plot the ellipse
#     cy, cx = ellipse_perimeter(yc, xc, a, b)
#     max_projection[cy, cx] = 255

#     #  show ellipse
#     ax[2].imshow(max_projection, cmap="gray")
#     ax[2].set_title("Ellipse")
#     ax[2].axis("off")

#     plt.savefig("debug/rotation_out_of_plane_max_projection.png")


if __name__ == "__main__":
    image_stack, rotated_image_stack, rotator, num_frames = setup()
    make_plot(image_stack, rotated_image_stack, rotator, num_frames)
    # test_max_projection_with_rotation_out_of_plane()
