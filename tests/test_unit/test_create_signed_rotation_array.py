import numpy as np

from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_create_signed_rotation_array_interleaved(
    full_length, start_end_times, direction_interleaved
):
    start, end = start_end_times
    rotation_on = DerotationPipeline.create_signed_rotation_array(
        full_length,
        start,
        end,
        direction_interleaved,
    )

    for idx in range(0, len(start), 2):
        assert np.all(rotation_on[start[idx] : end[idx]] == 1)
        assert np.all(rotation_on[start[idx + 1] : end[idx + 1]] == -1)


def test_create_signed_rotation_array_incremental(
    full_length, start_end_times, direction_incremental
):
    start, end = start_end_times
    rotation_on = DerotationPipeline.create_signed_rotation_array(
        full_length,
        start,
        end,
        direction_incremental,
    )

    for idx in range(0, 5):
        assert np.all(rotation_on[start[idx] : end[idx]] == 1)
        assert np.all(rotation_on[start[idx + 5] : end[idx + 5]] == -1)
