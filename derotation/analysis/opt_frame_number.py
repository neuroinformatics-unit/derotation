import numpy as np
from scipy.optimize import bisect


def count_frames(k, frame_clock, target_len):
    # Calculate the threshold using a percentile of the total signal
    mean = np.mean(frame_clock)
    std = np.std(frame_clock)
    threshold = mean + k * std

    frames_start = np.where(np.diff(frame_clock) > threshold)[0]
    return len(frames_start) - target_len


def find_best_k(clock, image, clock_type):
    target_len = len(image) if clock_type == "frame" else len(image) * 256
    result = bisect(count_frames, -4, 4, args=(clock, target_len))  # -1, 1,
    best_k = result

    # Check if the best value of k satisfies the assertions
    threshold = np.mean(clock) + best_k * np.std(clock)
    start = np.where(np.diff(clock) > threshold)[0]

    assert len(start) == target_len, f"{len(start)} != {target_len}"

    return best_k
