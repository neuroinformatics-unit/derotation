import numpy as np

from derotation.analysis.full_rotation_pipeline import FullPipeline


def test_finding_correct_start_end_times_with_threshold(
    derotation_pipeline: FullPipeline,
    full_rotation: np.ndarray,
    std_coef: int,
    number_of_rotations: int,
    rotation_len: int,
):
    start, end = derotation_pipeline.get_start_end_times(
        full_rotation, std_coef
    )

    assert len(start) == len(end)
    assert start[0] < end[0]
    assert end[-1] > start[-1]
    assert len(start) == number_of_rotations
    assert start[0] == rotation_len - 1
