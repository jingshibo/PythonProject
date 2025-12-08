##
import numpy as np
import pandas as pd
from Gesture_Classification import Plot


## Read data
subject = 'Number1'
mode = 'fist'


# Read IMU data
with open(f'D:\Data\Gesture_Classification\subject_{subject}\\raw_data\\IMU\{mode}.dat', 'rb') as file:
    # Read the entire file into a numpy array of type 'float32'
    arr = np.fromfile(file, dtype=np.float32)
    imu_data = pd.DataFrame(np.reshape(arr[:], newshape=(-1, 22)))

# ## check data loss
diff = np.diff(imu_data, axis=0)
rows_with_zero = np.where(diff[:, 0] != 1)[0]

# plot the pulses of imu and emg for alignment
Plot.plotImuData(imu_data, start_index=0, end_index=-1)



