import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from derotation.analysis.full_rotation_pipeline import FullPipeline

derotate = FullPipeline("full_rotation")
derotate.process_analog_signals()

#  get angular velocity from array of angles


def get_angular_velocity(angles, sampling_rate):
    #  angles are in degrees
    # angles = np.deg2rad(angles)
    diffs = np.diff(angles)
    #  remove outliers
    diffs = np.where(np.abs(diffs) > 0.05, 0, diffs)
    angular_velocity = diffs / sampling_rate

    return angular_velocity, diffs


angles_array = derotate.interpolated_angles
sampling_rate = 1 / 179200  # 1 / Hz

angular_velocity, diffs = get_angular_velocity(angles_array, sampling_rate)

#  if the angular velocity is 0, and the previous value is not 0, replace with the previous value
for i in tqdm(range(1, len(angular_velocity) // 2)):
    if i != 0 or i <= len(angular_velocity) - 2:
        if (
            angular_velocity[i] == 0
            and angular_velocity[i + 1] != 0
            and angular_velocity[i - 1] != 0
        ):
            angular_velocity[i] = angular_velocity[i - 1]

plt.plot(angular_velocity[: len(angular_velocity) // 2])
# plt.plot(angles_array)
# plt.plot(diffs)
plt.show()


print("debug")
