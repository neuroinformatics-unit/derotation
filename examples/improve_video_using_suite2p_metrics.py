from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from suite2p.io import BinaryFile
from suite2p.registration.nonrigid import make_blocks, spatial_taper

# derotator = IncrementalPipeline("incremental_rotation")
# derotator()


# # extract luminance variations across angles
# zero_rotation_mean_image = derotator.get_target_image(derotator.masked)
# mean_images_across_angles = derotator.calculate_mean_images(derotator.masked)


#  load registered bin file of suite2p
path_to_bin_file = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/incremental/derotated/suite2p/plane0/data.bin"
)
shape_image = [256, 256]
registered = BinaryFile(
    Ly=shape_image[0], Lx=shape_image[0], filename=path_to_bin_file
).file

time, x, y = registered.shape
registered = registered.reshape(time, x * y)
model = PCA(n_components=10, random_state=0).fit(registered)


print("debug")


path_options = Path("/Users/lauraporta/local_data/laura_ops.npy")
ops = np.load(path_options, allow_pickle=True).item()

bin_size = int(
    max(1, ops["nframes"] // ops["nbinned"], np.round(ops["tau"] * ops["fs"]))
)

with BinaryFile(filename=path_to_bin_file, Ly=ops["Ly"], Lx=ops["Lx"]) as f:
    registered = f.bin_movie(
        bin_size=bin_size,
        bad_frames=ops.get("badframes"),
        y_range=ops["yrange"],
        x_range=ops["xrange"],
    )

mov_mean = registered.mean(axis=0)
registered -= mov_mean
model = PCA(n_components=8, random_state=0).fit(registered)

result = (registered @ model.components_.T) @ model.components_


# use PCA as in the suite2p code - this takes in account for non rigid
# registration, that we are not going to do
path_options = Path("/Users/lauraporta/local_data/laura_ops.npy")
ops = np.load(path_options, allow_pickle=True).item()

mov_mean = registered.mean(axis=0)
registered -= mov_mean

block_size = [ops["block_size"][0] // 2, ops["block_size"][1] // 2]
nframes, Ly, Lx = registered.shape
yblock, xblock, _, block_size, _ = make_blocks(Ly, Lx, block_size=block_size)
nblocks = len(yblock)
Lyb, Lxb = block_size
n_comps_frac = 0.5
n_comps = int(min(min(Lyb * Lxb, nframes), min(Lyb, Lxb) * n_comps_frac))
maskMul = spatial_taper(Lyb // 4, Lyb, Lxb)
block_re = np.zeros((nblocks, nframes, Lyb * Lxb))
norm = np.zeros((Ly, Lx), np.float32)

for i in range(nblocks):
    block = registered[
        :, yblock[i][0] : yblock[i][-1], xblock[i][0] : xblock[i][-1]
    ].reshape(-1, Lyb * Lxb)
    model = PCA(n_components=n_comps, random_state=0).fit(block)
    block_re[i] = (block @ model.components_.T) @ model.components_
    norm[yblock[i][0] : yblock[i][-1], xblock[i][0] : xblock[i][-1]] += maskMul

reconstruction = np.zeros_like(registered)

block_re = block_re.reshape(nblocks, nframes, Lyb, Lxb)
block_re *= maskMul
for i in range(nblocks):
    reconstruction[
        :, yblock[i][0] : yblock[i][-1], xblock[i][0] : xblock[i][-1]
    ] += block_re[i]
reconstruction /= norm
reconstruction += mov_mean

plt.imshow(norm)
plt.show()


print("debug")
