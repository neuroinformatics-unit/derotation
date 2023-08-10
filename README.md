# Derotation
Rotate line by line images from 2- and 3-photon calcium imaging experiments.

## Installation
Create your environment and install the package from the branch `napari-matplotlib`:
```shell
conda create -n derotation python=3.10
conda activate derotation
git clone git+https://github.com/neuroinformatics-unit/derotation
git checkout napari-matplotlib
pip install -e .
```
## Edit configuration file
The configuration file is located in `derotation/config.yaml`.
In there you can specify the path in which to find the `tif` file, the `aux_stim` file and the `stumulus_randperm.mat` file, and most importantly your `dataset-folder` path.
The derotated tiffs will be saved as `{dataset-folder}/derotated/masked.tif`.

## Run with a script
`python3 derotation/derotate.py`` will run the derotation based on the configuration file.

## Run with Napari
```shell
napari
```
In the GUI, click on `Plugins` and then `NIU derotation plotting`.
The data will be automatically loaded based on the configuration file.
On the right you will see three buttons:
- `Run analysis`: to run the pre-processing of the analog signal. After pressing this button the plot underneath will display the calculated derotation angles. If you navigate the video using the slider on the bottom, the vertical red line will show you where in the angle trace you are.
- `Rotate by line`: to rotate the video by line. This will take a long while. Check the console to know at which frame did the computation arrive. The derotated video will be saved in `{dataset-folder}/derotated/masked.tif`.
