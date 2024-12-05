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
