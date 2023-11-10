# Derotation

The Derotation package offers a robust solution for reconstructing rotated images, particularly those acquired during experimental preparations in the field of neuroscience research. Although it's tailored for calcium imaging data, this versatile tool is adaptable for any dataset where rotation angles are recorded, making it an essential utility for a wide array of image processing applications.

The core algorithm, `rotate_an_image_array_line_by_line` can be also used as a standalone function to deform images by a given angle.

## Derotation example: a grid
Derotate an image of a rotating grid using motor feedback.
On the left, the original image. On the right, the derotated image.
![Derotation example: a grid](
  ./images/rotation_by_line.png)

## Deformation example: any image stack, any rotation angle array
With the same algorithm, you can deform any image stack by any rotation angle array.
Here an example of the deformation of a stack of images of dogs, by using a linearly increasing rotation angle array.
![Deformation example: any image stack, any rotation angle array](
  ./images/dog_rotated.png)
It has been created with the script `examples/deformation_by_line_of_a_dog.py`.

## Introduction: Derotation in Neuroscience
In neuroscientific experiments, particularly those involving calcium imaging, precise data reconstruction is paramount. Derotation is designed to address this need, reconstructing the imaging data from experiments where the sample undergoes known rotational patterns.

### Experimental Protocol

The package is built to accommodate a specific experimental protocol which includes two primary recording phases:

- **Incremental Rotation**: Samples undergo a rotation of 10 degrees every 2 seconds. This phase is crucial for sampling luminance changes due to varying positions.
- **Full Rotation**: In this key phase, the sample's activity is recorded over a complete 360-degree rotation at varying speeds and directions, following a pseudo-randomized sequence.

## Prerequisites

To utilize the Derotation package, the following files are required:

- A `tif` file with the image data.
- An `aux_stim` file containing the analog signals from:
  - Rotation on signal (indicating when rotation is active)
  - Motor ticks (for stepper motors, one tick per angle increment)
  - Frame clock (one tick per new frame acquisition)
  - Line clock (one tick per new line acquisition)
- A `stumulus_randperm.mat` file detailing the stimuli randomization (including speed and direction variations).

## Installation

Install the Derotation package and its dependencies using pip with the following commands:

```shell
git clone https://github.com/neuroinformatics-unit/derotation.git
cd derotation
pip install .
```

## Configuration
Navigate to the `derotation/config/` directory to access and edit the configuration files. You'll find two examples: `full_rotation.yml` for full rotations and `incremental_rotation.yml` for incremental rotations.

### Setting up paths
Within these configuration files, specify the paths for your data under `paths_read` for the `tif`, `aux_stim`, and `stumulus_randperm`.mat files. The paths_write key allows you to define where the derotated TIFFs, debugging plots, and logs will be saved.

### Config Parameters
Here's a quick rundown of the configuration parameters you'll need to adjust:

* `channel_names`: List the names of the signals in the aux_stim file.
* `rotation_increment`: Set the motor's angle increment.
* `adjust_increment`: Enable this to adjust the rotation_increment if the motor ticks don't sum to exactly 360 degrees.
* `rot_deg`: Define a full rotation degree count.
* `debugging_plots`: Toggle this to save debugging plots.
* `analog_signals_processing`: Configure parameters for analog signal processing, including peak finding and pulse processing.
* `interpolation`: Settings related to how the rotation angle is determined from the acquisition times.

## Usage
To run the derotation process, use one of the example scripts provided:

```shell
python3 examples/derotate.py  # For full rotation based on "full_rotation.yml"
python3 examples/derotate_incremental.py  # For incremental rotation based on "incremental_rotation.yml"
```

### What happens behind the scenes?
**Full rotation**:
The main experimental protocol is a full rotation, where the sample is rotated 360 degrees. The rotation is recorded by the motor ticks, which are then converted to angles. The rotation angle is then interpolated to the line clock, which is used to derotate the image. Several steps are taken to ensure the accuracy of the derotation, including:
- check the number of ticks
- check the number of rotations
- adjust the rotation increment if necessary
Debugging plots can be used to visually inspect the output of each step.

**Incremental rotation**:
The incremental rotation is a preparatory phase that precedes the full rotation. The sample is rotated 10 degrees every 2 seconds. The rotation is recorded by the motor ticks, which are then converted to angles. The rotation angle is then interpolated to the frame clock, which is used to derotate the image. The same steps are taken to ensure the accuracy of the derotation.
In addition, each frame position is corrected by using cross-correlation.

## Explanations
### Derotation by frame vs by line
A frame-based derotation is simply the rotation of the image as a whole and is achieved using `scipy.ndimage.rotate`. This is the default derotation method for incremental rotations.
The line-based derotation is more complex and is achieved by rotating each line of the image by a certain number of pixels. This is the default derotation method for full rotations.
