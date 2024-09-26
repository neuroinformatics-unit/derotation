import numpy as np

from derotation.analysis.full_derotation_pipeline import FullPipeline


def test_create_signed_rotation_array_interleaved(
    derotation_pipeline: FullPipeline,
    start_end_times: tuple,
):
    start, end = start_end_times
    rotation_on = derotation_pipeline.create_signed_rotation_array()

    for idx in range(0, len(start), 2):
        assert np.all(rotation_on[start[idx] : end[idx]] == 1)
        assert np.all(rotation_on[start[idx + 1] : end[idx + 1]] == -1)


def test_create_signed_rotation_array_incremental(
    derotation_pipeline: FullPipeline,
    start_end_times: tuple,
    direction_incremental: np.ndarray,
):
    derotation_pipeline.direction = direction_incremental
    start, end = start_end_times
    rotation_on = derotation_pipeline.create_signed_rotation_array()

    for idx in range(0, 5):
        assert np.all(rotation_on[start[idx] : end[idx]] == 1)
        assert np.all(rotation_on[start[idx + 5] : end[idx + 5]] == -1)
