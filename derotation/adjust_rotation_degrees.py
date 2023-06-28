from pathlib import Path

import numpy as np
import scipy.optimize as opt
from find_centroid import detect_blobs, extract_blob_centers, preprocess_image
from matplotlib import pyplot as plt
from scipy.ndimage import rotate


def find_rotation_blocks(image_rotation_degree_per_frame):
    blocks = []
    indexes = []

    cumulative_index = 0
    # find idx of first non zero value
    while len(image_rotation_degree_per_frame) > 100:
        first_non_zero_idx = np.where(image_rotation_degree_per_frame != 0)[0][
            0
        ]
        len_first_group = np.where(
            image_rotation_degree_per_frame[first_non_zero_idx:] == 0
        )[0][0]
        blocks.append(
            image_rotation_degree_per_frame[
                first_non_zero_idx : first_non_zero_idx + len_first_group
            ]
        )
        image_rotation_degree_per_frame = image_rotation_degree_per_frame[
            first_non_zero_idx + len_first_group :
        ]

        indexes.append(
            np.arange(
                cumulative_index + first_non_zero_idx,
                cumulative_index + first_non_zero_idx + len_first_group,
            )
        )
        cumulative_index += first_non_zero_idx + len_first_group

    return blocks, indexes


def find_parametric_curves(blocks):
    parametric_curves = []
    for block in blocks:
        x = np.arange(len(block))
        y = block
        # fit a 4th order polynomial
        popt = np.polyfit(x, y, 4)
        parametric_curves.append(popt)

    return parametric_curves


def make_parametric_curve(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def get_rotation_degrees_from_a_curve(parametric_curve, frame_number):
    a, b, c, d, e = parametric_curve
    return (
        a * frame_number**4
        + b * frame_number**3
        + c * frame_number**2
        + d * frame_number
        + e
    )


def rotate_image(
    image,
    image_rotation_degree_per_frame,
):
    rotated_image = np.empty_like(image)
    for i in range(len(image)):
        rotated_image[i] = rotate(
            image[i], image_rotation_degree_per_frame[i], reshape=False
        )

    return rotated_image


def get_centers(image):
    img = preprocess_image(image)
    labels, properties = detect_blobs(img)
    centroids = extract_blob_centers(properties)

    return centroids


def get_mean_centroid(centroids):
    if len(centroids) > 0:
        x = []
        y = []
        for img_centroids in centroids[:10]:
            if len(img_centroids) >= 3:
                pass
            for c in img_centroids:
                if not_center_of_image(c) and in_region(c):
                    x.append(c[1])
                    y.append(c[0])

        mean_x = np.mean(x)
        mean_y = np.mean(y)

    return mean_x, mean_y


def not_center_of_image(c):
    return not (115 <= c[0] <= 135 and 115 <= c[1] <= 135)


def in_region(c):
    return 50 <= c[0] <= 200 and 50 <= c[1] <= 200


def optimize_image_rotation_degrees(
    _image, _image_rotation_degree_per_frame, use_curve_fit=False
):
    blocks, indexes = find_rotation_blocks(_image_rotation_degree_per_frame)
    if use_curve_fit:
        parametric_curves = find_parametric_curves(blocks)

    results = []
    for i in range(len(blocks)):
        frames_to_consider = indexes[i]
        images = _image[frames_to_consider]

        centers = []
        for img in images:
            centers.append(get_centers(img))
        mean_x, mean_y = 150, 160  # get_mean_centroid(centers)

        iteration = [0]  # use mutable object to hold the iteration number

        def callback(xk):
            iteration[0] += 1
            print(
                "Iteration: {0} - Function value: {1}".format(
                    iteration[0], f(xk)
                )
            )

        def f(parameters):
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(images[0], cmap="gist_ncar")
            ax[1].plot(blocks[i], marker=".", color="red")

            if use_curve_fit:
                rots = make_parametric_curve(
                    np.arange(len(images)), *parameters
                )
            else:
                rots = parameters
            ax[1].plot(rots, marker=".", color="black")

            rotated_image = rotate_image(images, rots)
            diff_x = []
            diff_y = []
            for img in rotated_image:
                try:
                    centers_rotated_image = get_centers(img)
                    center_dim_blob = mean_x, mean_y
                    for c in centers_rotated_image:
                        if not_center_of_image(c) and in_region(c):
                            center_dim_blob = c

                    ax[0].plot(
                        center_dim_blob[1],
                        center_dim_blob[0],
                        marker="*",
                        color="red",
                    )

                    diff_x.append(np.abs(center_dim_blob[1] - mean_x))
                    diff_y.append(np.abs(center_dim_blob[0] - mean_y))
                except IndexError:
                    pass

            path = Path(f"derotation/figures/block_{i}/")
            path.mkdir(parents=True, exist_ok=True)

            fig.savefig(f"{path}/iteration_{iteration[0]}.png")
            plt.close(fig)
            return np.mean(diff_x) + np.mean(diff_y)

        result = opt.minimize(
            f,
            parametric_curves[i] if use_curve_fit else blocks[i],
            method="Nelder-Mead",
            options={
                "xatol": 1e-2,
                "fatol": 1e-2,
                "maxiter": 500,
                "maxfev": 1000,
            },
            callback=callback,
        )
        results.append(result.x)

    return results
