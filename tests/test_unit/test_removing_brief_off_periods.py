import numpy as np

from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_removing_brief_off_periods(
    start_end_times,
    start_end_times_with_bug,
):
    inter_rotation_interval_min_len = 50
    start_buggy, end_buggy = start_end_times_with_bug
    start, end = start_end_times

    corrected = DerotationPipeline.correct_start_and_end_rotation_signal(
        inter_rotation_interval_min_len,
        start_buggy,
        end_buggy,
    )

    assert len(corrected["start"]) == len(corrected["end"])
    assert len(corrected["start"]) == len(start)
    assert len(corrected["end"]) == len(end)
    assert corrected["start"][0] == start[0]
    assert corrected["end"][0] == end[0]

    assert np.any(
        corrected["end"] - corrected["start"]
        >= inter_rotation_interval_min_len
    )
