import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure


def preprocess_image(image):
    lower_threshold = -2700
    higher_threshold = -2400
    image = np.where(
        (image >= lower_threshold) & (image <= higher_threshold), 255, 0
    ).astype(np.uint8)

    return image


def detect_blobs(image):
    # Convert the image to grayscale
    # gray = image.mean().astype(np.uint8)

    # Apply the Laplacian of Gaussian (LoG) filter
    # filtered_image = gaussian_laplace(image, sigma=8)
    filtered_image = gaussian_filter(image, sigma=8)

    # Apply thresholding to obtain a binary image
    threshold_value = 32  # Adjust this value according to your image
    binary_image = filtered_image > threshold_value

    # Perform blob detection using connected component labeling
    labels = measure.label(binary_image)
    properties = measure.regionprops(labels)

    # Return the labels and regions of the detected blobs
    return labels, properties


def extract_blob_centers(properties):
    centroids = [p.centroid for p in properties]

    return centroids
