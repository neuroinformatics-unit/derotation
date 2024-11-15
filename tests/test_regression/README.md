# How do these regression tests work?

This directory contains regression tests for the `derotation` package.

Every test can be run directly via `pytest` and relies on a set of pre-computed images stored in the `images` directory. The current state of the package is compared against them.
If the computations change substantially and you think the new results are correct, you can update the reference images by running the scripts in the folder `recreate_target`.
The `recreate_target` scripts will generate the reference images in the `images` directory.

In this table you can find a summary of the test modules and the corresponding reference images and scripts to recreate them:

| Test module | Reference image | Recreate script | Notes |
|-------------|-----------------|-----------------| ----- |
| `test_basic_rotator_with_ different_center_of_rotation.py` | `images/rotator/ rotated_frame_{center}_{id}.png` | `recreate_target/ rotator_different_center.py` | Testing the rotator with different centers of rotation with a gray-striped square sample image. |
| `test_basic_rotator.py` | `images/rotator/ rotated_frame_{id}.png` | `recreate_target/ basic_rotator.py` | Testing the rotator with a gray-striped square sample image. |
| `test_derotate_with_ different_center_of_rotation.py` | `images/rotator_derotator/ derotated_frame_{center}_{id}.png` | `recreate_target/ derotate_different_center.py` | Testing the derotator with different centers of rotation with a gray-striped square sample image on images previously rotated. |
| `test_derotation_by_line.py` | `images/sinusoidal_rotation/ rotated_dog_{id}.png`, `images/uniform_rotation/ rotated_dog_{id}.png` | `recreate_target/ derotation_by_line.py` | Testing the derotator with a sinusoidal and uniform rotation with a dog sample image. |
| `test_rotation_and_derotation.py` | `images/rotator_derotator/ derotated_frame_{id}.png` | `recreate_target/ rotation_and_derotation.py` | Testing the derotator with a gray-striped square sample image on images previously rotated. |
