import numpy as np
import pytest


@pytest.fixture
def number_of_rotations():
    return 10


@pytest.fixture
def rotation_len():
    return 100


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
    return 0
