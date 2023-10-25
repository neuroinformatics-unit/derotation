import numpy as np
import pytest


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
def full_rotation(number_of_rotations, rotation_len):
    sequence = [0, 1, 0]
    dummy_signal = [
        num
        for _ in range(number_of_rotations)
        for num in sequence
        for _ in range(rotation_len)
    ]
    dummy_signal += np.random.normal(
        0, 0.1, rotation_len * number_of_rotations * len(sequence)
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
def k():
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
