import numpy as np

from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_adjust_rotation_increment_360(
    derotation_pipeline: DerotationPipeline,
):
    (
        new_increments,
        ticks_per_rotation,
    ) = derotation_pipeline.adjust_rotation_increment()

    new_increments = np.round(new_increments, 0)

    correct_increments = np.array(
        [10.0, 9.0, 11.0, 10.0, 11.0, 13.0, 14.0, 12.0, 10.0, 11.0]
    )
    correct_tick_number = np.array([35, 42, 33, 37, 32, 28, 25, 30, 36, 33])

    assert np.all(new_increments == correct_increments)
    assert np.all(ticks_per_rotation == correct_tick_number)


def test_adjust_rotation_increment_5(
    derotation_pipeline: DerotationPipeline,
):
    derotation_pipeline.rot_deg = 5

    (
        new_increments,
        _,
    ) = derotation_pipeline.adjust_rotation_increment()

    new_increments = np.round(new_increments, 3)

    correct_increments = np.array(
        [0.143, 0.119, 0.152, 0.135, 0.156, 0.179, 0.2, 0.167, 0.139, 0.152]
    )

    assert np.all(
        new_increments == correct_increments
    ), f"new_increments: {new_increments}"
