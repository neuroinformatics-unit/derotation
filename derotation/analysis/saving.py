import numpy as np


def add_circle_mask(img_stack):
    img_height = img_stack.shape[1]
    xx, yy = np.mgrid[:img_height, :img_height]
    circle = (xx - img_height / 2) ** 2 + (yy - img_height / 2) ** 2
    mask = circle < (img_height / 2) ** 2

    masked_img_array = []
    for img in img_stack:
        masked_img_array.append(np.where(mask, img, np.nan))

    return masked_img_array
