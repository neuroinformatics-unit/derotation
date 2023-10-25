import numpy as np

from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_finding_correct_start_end_times_with_threshold(
    derotation_pipeline: DerotationPipeline,
    full_rotation: np.ndarray,
    k: int,
    number_of_rotations: int,
    rotation_len: int,
):
    start, end = derotation_pipeline.get_start_end_times_with_threshold(
        full_rotation, k
    )

    assert len(start) == len(end)
    assert start[0] < end[0]
    assert end[-1] > start[-1]
    assert len(start) == number_of_rotations
    assert start[0] == rotation_len - 1
