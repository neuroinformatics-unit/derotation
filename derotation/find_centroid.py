import numpy as np
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter
from skimage import measure


def not_center_of_image(c):
    return not (110 <= c[0] <= 145 and 110 <= c[1] <= 145)


def in_region(c):
    return 50 <= c[0] <= 200 and 50 <= c[1] <= 200


def preprocess_image(image, lower_threshold=-2700, higher_threshold=-2600):
    image = np.where(
        (image >= lower_threshold) & (image <= higher_threshold), 255, 0
    ).astype(np.uint8)

    return image


def detect_blobs(image, sigma=2.5, threshold_value=32):
    filtered_image = gaussian_filter(image, sigma=sigma)

    # Apply thresholding to obtain a binary image
    binary_image = filtered_image > threshold_value

    # Perform blob detection using connected component labeling
    labels = measure.label(binary_image)
    properties = measure.regionprops(labels)

    # Return the labels and regions of the detected blobs
    return labels, properties


def extract_blob_centers(properties):
    centroids = [p.centroid for p in properties]

    return centroids


def find_centroid_pipeline(image, x):
    lower_threshold, higher_threshold, binary_threshold, sigma = x
    img = preprocess_image(image, lower_threshold, higher_threshold)
    labels, properties = detect_blobs(img, sigma, binary_threshold)
    centroids = extract_blob_centers(properties)
    return centroids


def get_optimized_centroid_location(image):
    # initial parameters
    lower_threshold = -2700
    higher_threshold = -2600
    binary_threshold = 32
    sigma = 2.5

    x = [lower_threshold, higher_threshold, binary_threshold, sigma]

    iteration = [0]  # use mutable object to hold the iteration number

    def cb(xk):
        iteration[0] += 1
        print(
            "Iteration: {0} - Function value: {1}".format(iteration[0], f(xk))
        )

    def f(x):
        centroids = find_centroid_pipeline(image, x)
        count_valid_centroids = 0
        for c in centroids:
            if not_center_of_image(c) and in_region(c):
                count_valid_centroids += 1

        if count_valid_centroids == 1:
            return 0
        else:
            return 1

    res = opt.minimize(
        f,
        x,
        method="nelder-mead",
        options={"xtol": 1e-8, "disp": True},
        # callback=cb
    )

    # now find the centroid
    centroids = find_centroid_pipeline(image, res.x)

    return centroids, res.x
