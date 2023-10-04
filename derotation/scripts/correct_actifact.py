from pathlib import Path

import tifffile as tiff

path_tif = Path(
    "/Users/lauraporta/local_data/rotation/230818_pollen_rotation/masked_no_rotation.tif"
)

image = tiff.imread(path_tif)
print(image.shape)
remove_last_bit = image[:1500, :, :]
print(remove_last_bit.shape)

#  save
path_tif = Path(
    "/Users/lauraporta/local_data/rotation/230818_pollen_rotation/masked_no_rotation_no_last_bit.tif"
)
tiff.imsave(path_tif, remove_last_bit)
