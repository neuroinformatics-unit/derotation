import numpy as np
import pytest

from derotation.analysis.full_derotation_pipeline import FullPipeline


@pytest.fixture(autouse=True)
def random():
    np.random.seed(42)


@pytest.fixture
def number_of_rotations():
    return 10


@pytest.fixture
def rotation_len():
    return 100


@pytest.fixture
def full_length(number_of_rotations, rotation_len):
    return number_of_rotations * rotation_len * 3


@pytest.fixture
def direction_interleaved():
    direction = np.zeros(10)
    direction[::2] = 1
    direction[1::2] = -1
    return direction


@pytest.fixture
def direction_incremental():
    direction_1 = np.ones(5)
    direction_2 = np.ones(5) * -1
    direction = np.concatenate((direction_1, direction_2))
    return direction


@pytest.fixture
def intervals():
    return [0, 1, 0]


@pytest.fixture
def dummy_signal(intervals, number_of_rotations, rotation_len):
    dummy_signal = [
        num
        for _ in range(number_of_rotations)
        for num in intervals
        for _ in range(rotation_len)
    ]
    return dummy_signal


@pytest.fixture
def full_rotation(number_of_rotations, rotation_len, intervals, dummy_signal):
    dummy_signal += np.random.normal(
        0, 0.1, rotation_len * number_of_rotations * len(intervals)
    )

    return dummy_signal


@pytest.fixture
def start_end_times(number_of_rotations, rotation_len):
    start = np.arange(
        rotation_len - 1,
        rotation_len - 1 + number_of_rotations * rotation_len * 3,
        3 * rotation_len,
    )
    end = np.arange(
        rotation_len - 1 + rotation_len,
        2 * rotation_len - 1 + number_of_rotations * rotation_len * 3,
        3 * rotation_len,
    )

    return start, end


@pytest.fixture
def start_end_times_with_bug(start_end_times):
    start, end = start_end_times
    fictional_end = 130
    fictional_start = 140

    start = np.insert(start, 1, fictional_start)
    end = np.insert(end, 0, fictional_end)

    return start, end


@pytest.fixture
def direction():
    direction = np.zeros(10)
    direction[::2] = 1
    direction[1::2] = -1
    return direction


@pytest.fixture
def std_coef():
    return 0.2


@pytest.fixture
def ticks_per_rotation():
    return 100


@pytest.fixture
def rotation_ticks(
    full_length,
    ticks_per_rotation,
    number_of_rotations,
):
    correct_number_of_ticks = ticks_per_rotation * number_of_rotations
    number_of_ticks = correct_number_of_ticks + np.random.randint(0, 10)

    # distribute number_of_ticks in poisson indices
    ticks = np.random.choice(
        range(full_length), size=number_of_ticks, replace=False
    )
    ticks = np.sort(ticks)
    return ticks


@pytest.fixture
def corrected_increments():
    return np.asarray([10, 9, 11, 9, 11, 13, 14, 12, 10, 11])


@pytest.fixture
def ticks_per_rotation_calculated():
    return np.asarray([36, 42, 33, 39, 34, 28, 25, 30, 36, 33])


@pytest.fixture
def derotation_pipeline(
    rotation_ticks,
    start_end_times,
    full_length,
    number_of_rotations,
    full_rotation,
    direction,
    corrected_increments,
    ticks_per_rotation_calculated,
    dummy_signal,
):
    pipeline = FullPipeline.__new__(FullPipeline)

    pipeline.inter_rotation_interval_min_len = 50
    pipeline.rotation_ticks_peaks = rotation_ticks
    pipeline.rot_blocks_idx = {
        "start": start_end_times[0],
        "end": start_end_times[1],
    }
    pipeline.number_of_rotations = number_of_rotations
    pipeline.direction = direction
    pipeline.total_clock_time = full_length
    pipeline.full_rotation = full_rotation
    pipeline.rot_deg = 360
    pipeline.rotation_on = dummy_signal

    return pipeline


@pytest.fixture
def len_stack():
    return 3  # Number of frames in the stack


@pytest.fixture
def test_image():
    """Create a single frame image with a gradient square."""
    image = np.zeros((100, 100))
    gray_values = [i % 5 * 60 + 15 for i in range(100)]
    for i in range(100):
        image[i] = gray_values[i]
    image[:20] = 0
    image[-20:] = 0
    image[:, :20] = 0
    image[:, -20:] = 0

    return image


@pytest.fixture
def image_stack(len_stack, test_image):
    """Create a stack of frames with the same test image."""
    return np.array([test_image for _ in range(len_stack)])


@pytest.fixture
def angles(image_stack):
    """Generate a simple set of rotation angles."""
    n_total_lines = image_stack.shape[0] * image_stack.shape[1]
    return np.arange(n_total_lines)
