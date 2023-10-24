from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_finding_correct_start_end_times_with_threshold(
    full_rotation, k, rotation_len, number_of_rotations
):
    start, end = DerotationPipeline.get_start_end_times_with_threshold(
        full_rotation, k
    )

    assert len(start) == len(end)
    assert start[0] < end[0]
    assert end[-1] > start[-1]
    assert len(start) == number_of_rotations
    assert start[0] == rotation_len - 1
