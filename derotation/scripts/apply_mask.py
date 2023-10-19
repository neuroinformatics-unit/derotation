from pathlib import Path

import numpy as np
from tifffile import imread, imsave


def apply_mask(path_of_video):
    img_stack = imread(path_of_video)
    img_height = img_stack.shape[1]
    xx, yy = np.mgrid[:img_height, :img_height]
    circle = (xx - img_height / 2) ** 2 + (yy - img_height / 2) ** 2
    mask = circle < (img_height / 2) ** 2
    img_min = np.nanmin(img_stack)
    masked_img_array = []
    for img in img_stack:
        masked_img_array.append(np.where(mask, img, img_min))
    masked_img_array = np.array(masked_img_array)
    return masked_img_array


if "__main__" == __name__:
    # path_of_video = Path(sys.argv[1])
    path_of_video = Path(
        "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/imaging/translation2_00001_ce.tif"
    )
    masked_video = apply_mask(path_of_video)
    saving_path = path_of_video.parent / f"masked_raw_{path_of_video.name}"
    imsave(saving_path, masked_video)

    print(f"Masked video saved at {saving_path}")
