import copy
import pickle
from pathlib import Path

import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from scipy.ndimage import rotate

from derotation.analysis.find_centroid import (
    detect_blobs,
    extract_blob_centers,
    find_centroid_pipeline,
    get_optimized_centroid_location,
    in_region,
    not_center_of_image,
    preprocess_image,
)


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


def rotate_all_images(
    image,
    image_rotation_degree_per_frame,
):
    rotated_images = np.empty_like(image)
    for i in range(len(image)):
        rotated_images[i] = rotate(
            image[i], image_rotation_degree_per_frame[i], reshape=False
        )

    return rotated_images


def get_centers(image):
    img = preprocess_image(image)
    labels, properties = detect_blobs(img)
    centroids = extract_blob_centers(properties)

    if (len(centroids) == 1) and (
        (not not_center_of_image(centroids[0]))
        or (not in_region(centroids[0]))
    ):
        print("only one blob found")

    return centroids


def optimize_image_rotation_degrees_with_centroids(
    _image, _image_rotation_degree_per_frame
):
    blocks, indexes = find_rotation_blocks(_image_rotation_degree_per_frame)

    results = []
    for i in range(len(blocks)):
        frames_to_consider = indexes[i]
        images = _image[frames_to_consider]

        centers = []
        optimized_parameters = []
        for img in images:
            c, x = get_optimized_centroid_location(img)
            centers.append(c)
            optimized_parameters.append(x)

        assert len(optimized_parameters) == len(images)

        mean_x, mean_y = 150, 160  # get_mean_centroid(centers)

        iteration = [0]  # use mutable object to hold the iteration number

        def cb(xk):
            iteration[0] += 1
            print(
                "============================================"
                + "\n"
                + "Iteration: {0} - Function value: {1}".format(
                    iteration[0], f(xk)
                )
            )

        def f(parameters):
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(images[0], cmap="gist_ncar")
            ax[1].plot(blocks[i], marker=".", color="red")

            rots = parameters
            ax[1].plot(rots, marker=".", color="black")

            rotated_images = rotate_all_images(images, rots)
            diff_x = []
            diff_y = []
            for k, img in enumerate(rotated_images):
                try:
                    centers_rotated_image = find_centroid_pipeline(
                        img, optimized_parameters[k]
                    )
                    center_dim_blob = mean_x, mean_y
                    for c in centers_rotated_image:
                        if not_center_of_image(c) and in_region(c):
                            center_dim_blob = c
                            break
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
            plt.close()

            # almost like arc length for small angles
            hypothenuse = [
                np.sqrt(x**2 + y**2) for x, y in zip(diff_x, diff_y)
            ]
            return np.sum(hypothenuse)

        result = opt.minimize(
            f,
            blocks[i],
            method="Nelder-Mead",
            options={
                "xatol": 1e-5,
                "fatol": 1e-5,
                "maxiter": 140,
                "maxfev": 1000,
            },
            # callback=cb,
        )
        results.append(result.x)

    return results, indexes, optimized_parameters


def get_optimal_rotation_degs(image, image_rotation_degree_per_frame):
    try:
        with open("derotation/optimized_parameters.pkl", "rb") as f:
            optimized_parameters = pickle.load(f)
        with open("derotation/indexes.pkl", "rb") as f:
            indexes = pickle.load(f)
        with open("derotation/opt_result.pkl", "rb") as f:
            opt_result = pickle.load(f)
    except FileNotFoundError:
        (
            opt_result,
            indexes,
            optimized_parameters,
        ) = optimize_image_rotation_degrees_with_centroids(
            image, image_rotation_degree_per_frame
        )
        with open("derotation/optimized_parameters.pkl", "wb") as f:
            pickle.dump(optimized_parameters, f)
        with open("derotation/indexes.pkl", "wb") as f:
            pickle.dump(indexes, f)
        with open("derotation/opt_result.pkl", "wb") as f:
            pickle.dump(opt_result, f)

    return opt_result, indexes, optimized_parameters


def apply_new_rotations(opt_result, image_rotation_degree_per_frame, indexes):
    new_image_rotation_degree_per_frame = copy.deepcopy(
        image_rotation_degree_per_frame
    )
    for i, block in enumerate(opt_result):
        new_image_rotation_degree_per_frame[indexes[i]] = block

    return new_image_rotation_degree_per_frame
