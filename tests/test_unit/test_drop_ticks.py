from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_drop_ticks_generated_randomly(
    rotation_ticks, start_end_times, full_length, number_of_rotations
):
    start, end = start_end_times
    cleaned_ticks = DerotationPipeline.drop_ticks_outside_of_rotation(
        rotation_ticks, start, end, full_length, number_of_rotations
    )

    assert len(cleaned_ticks) == 362
