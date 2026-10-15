import numpy as np

from derotation.simulate.line_scanning_microscope import Rotator


def test_Rotator_constructor():
    # image stack with 3 frames of 100x100 pixels
    image_stack = np.zeros((3, 100, 100))
    # 300 angles, one per line per frame
    angles = np.arange(300)
    # create a Rotator object
    Rotator(angles, image_stack)


def test_failing_Rotator_instantiation():
    # image stack with 3 frames of 100x100 pixels
    image_stack = np.zeros((3, 100, 100))
    # 299 angles, one per line per frame
    angles = np.arange(299)
    # create a Rotator object
    try:
        Rotator(angles, image_stack)
    except AssertionError:
        assert True
    else:
        assert False


def test_Rotator_non_square_keeps_the_frame_shape():
    image_stack = np.zeros((3, 40, 70))
    angles = np.linspace(0, 30, 3 * 40)
    rotated = Rotator(angles, image_stack).rotate_by_line()
    assert rotated.shape == image_stack.shape


def test_Rotator_cropped_frame_matches_cropped_square_result():
    """A frame cropped symmetrically in x has the same center of rotation,
    so its rotation matches the cropped rotation of the full frame wherever
    the rotated pixels come from inside the cropped frame.
    """
    rng = np.random.default_rng(0)
    size, crop = 60, 12
    image_stack = rng.integers(1, 255, (2, size, size)).astype(np.float64)
    angles = np.linspace(0, 90, 2 * size)

    square = Rotator(angles, image_stack, blank_pixel_val=0).rotate_by_line()
    cropped = Rotator(
        angles, image_stack[:, :, crop:-crop], blank_pixel_val=0
    ).rotate_by_line()

    expected = square[:, :, crop:-crop]
    assert cropped.shape == expected.shape
    filled = cropped != 0
    assert filled.mean() > 0.6
    assert np.array_equal(cropped[filled], expected[filled])
