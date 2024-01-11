##
import numpy as np
import pandas as pd
from Bipolar_EMG.Utility_Functions import Emg_Imu_Preprocessing


## Read data
subject = 'Number1'
mode = 'standing'

# Read bipolar EMG data
emg_data = pd.read_csv(f'D:\Data\Bipolar_Data\subject_{subject}\\raw_data\\bipolarEMG\{mode}.csv', delimiter=';')
emg_data.drop(emg_data.columns[[3, 4]], axis=1, inplace=True)
emg_data.columns = range(emg_data.shape[1])
# Read IMU data
with open(f'D:\Data\Bipolar_Data\subject_{subject}\\raw_data\\IMU\{mode}.dat', 'rb') as file:
    # Read the entire file into a numpy array of type 'float32'
    arr = np.fromfile(file, dtype=np.float32)
    imu_data = pd.DataFrame(np.reshape(arr[:], newshape=(-1, 22)))

# ## check data loss
# diff = np.diff(imu_data, axis=0)
# rows_with_zero = np.where(diff[:, 0] != 1)[0]


## plot the pulses of imu and emg for alignment
Emg_Imu_Preprocessing.plotEmgImu(emg_data, imu_data, start_index=0, end_index=-1)


## align the beginning of emg and imu data
emg_start_index = 9729
imu_start_index = 558
emg_begin_cropped, imu_begin_cropped = Emg_Imu_Preprocessing.alignEmgImuBeginIndex(emg_start_index, imu_start_index, emg_data, imu_data)
# plot begin-cropped emg and imu data for ending index alignment
Emg_Imu_Preprocessing.plotEmgImu(emg_begin_cropped, imu_begin_cropped, start_index=0, end_index=-1)


## align the end of emg and imu data
emg_end_index = 119802
imu_end_index = 4491
emg_aligned = emg_begin_cropped.iloc[:emg_end_index + 1, 1:].reset_index(drop=True)  # remove the first column indicating data number
imu_aligned = imu_begin_cropped.iloc[:imu_end_index + 1, 1:].reset_index(drop=True)  # remove the first column indicating data number
# upsamling insole data to match emg
imu_upsampled = Emg_Imu_Preprocessing.upsampleImuEqualToEmg(imu_aligned, emg_aligned)
Emg_Imu_Preprocessing.plotEmgImuAligned(emg_aligned, imu_upsampled, start_index=0, end_index=-1)


## save the aligned results (parameters and data)
Emg_Imu_Preprocessing.saveAlignParameters(subject, mode, emg_start_index, imu_start_index, emg_end_index, imu_end_index, project='Bipolar_Data')
Emg_Imu_Preprocessing.saveAlignedData(subject, mode, imu_aligned, emg_aligned, project='Bipolar_Data')


## read alignment parameters
emg_start_index, imu_start_index, emg_end_index, imu_end_index = Emg_Imu_Preprocessing.readAlignParameters(subject, mode, project='Bipolar_Data')
print(emg_start_index, imu_start_index, emg_end_index, imu_end_index)

