import numpy as np

from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_adjust_rotation_increment_360(
    derotation_pipeline: DerotationPipeline,
    corrected_increments,
    ticks_per_rotation_calculated,
):
    (
        new_increments,
        new_ticks_per_rotation,
    ) = derotation_pipeline.adjust_rotation_increment()

    new_increments = np.round(new_increments, 0)

    assert np.all(new_increments == corrected_increments), f"{new_increments}"
    assert np.all(
        new_ticks_per_rotation == ticks_per_rotation_calculated
    ), f"{new_ticks_per_rotation}"


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
        [0.139, 0.119, 0.152, 0.128, 0.147, 0.179, 0.2, 0.167, 0.139, 0.152]
    )

    assert np.all(
        new_increments == correct_increments
    ), f"new_increments: {new_increments}"
