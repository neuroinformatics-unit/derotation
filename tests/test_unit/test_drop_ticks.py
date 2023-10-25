from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_drop_ticks_generated_randomly(
    derotation_pipeline: DerotationPipeline,
):
    derotation_pipeline.drop_ticks_outside_of_rotation()

    assert len(derotation_pipeline.rotation_ticks_peaks) == 362
