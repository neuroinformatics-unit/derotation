import numpy as np
from scipy.optimize import bisect

def count_frames(k, frame_clock, image):
    # Calculate the threshold using a percentile of the total signal
    mean = np.mean(frame_clock)
    std = np.std(frame_clock)
    threshold = mean + k * std

    frames_start = np.where(np.diff(frame_clock) > threshold)[0]
    return len(frames_start) - len(image)

def find_best_k(frame_clock, image):
    result = bisect(count_frames, -1, 1, args=(frame_clock, image))
    best_k = result

    # Check if the best value of k satisfies the assertions
    threshold = np.mean(frame_clock) + best_k * np.std(frame_clock)
    frames_start = np.where(np.diff(frame_clock) > threshold)[0]

    assert len(frames_start) == len(image), f"{len(frames_start)} != {len(image)}"

    return best_k