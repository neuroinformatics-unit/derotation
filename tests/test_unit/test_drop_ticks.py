from derotation.analysis.full_derotation_pipeline import FullPipeline


def test_drop_ticks_generated_randomly(
    derotation_pipeline: FullPipeline,
):
    len_before = len(derotation_pipeline.rotation_ticks_peaks)
    derotation_pipeline.drop_ticks_outside_of_rotation()
    len_after = len(derotation_pipeline.rotation_ticks_peaks)

    assert len_before > len_after
    assert len(derotation_pipeline.rotation_ticks_peaks) == 333
