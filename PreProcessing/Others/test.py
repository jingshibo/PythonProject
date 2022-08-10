## import modules
import os
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import datetime
from PreProcessing.Recover_Insole import insertMissingRow
from PreProcessing import Align_Two_Insoles
from PreProcessing import Align_Insole_Emg
import csv

## initialization
data_dir = 'D:\Data\Insole_Emg'
data_file_name = 'subject0_20220805_185544'
left_insole_file = f'left_insole\left_{data_file_name}.csv'
right_insole_file = f'right_insole\\right_{data_file_name}.csv'
emg_file = f'emg\emg_{data_file_name}.csv'
left_insole_path = os.path.join(data_dir, left_insole_file)
right_insole_path = os.path.join(data_dir, right_insole_file)
emg_path = os.path.join(data_dir, emg_file)

raw_left_data = []
raw_right_data = []
raw_emg_data = []
insole_sampling_period = 25  # insole sampling period

## emg
now = datetime.datetime.now()
raw_emg_data = pd.read_csv(emg_path, sep=',', header=None, dtype='int16',
                           converters={0: str, 1: str, 2: str})  # change data type for faster reading
print(datetime.datetime.now() - now)
## left insole
raw_left_data = pd.read_csv(left_insole_path, sep=',', header=None)
recovered_left_data = insertMissingRow(raw_left_data, insole_sampling_period)  # add missing rows with NaN values

## right insole
raw_right_data = pd.read_csv(right_insole_path, sep=',', header=None)
recovered_right_data = insertMissingRow(raw_right_data, insole_sampling_period)  # add missing rows with NaN values

## align the begin of sensor data
# observe to get the beginning index
combined_insole_data = pd.concat([recovered_left_data, recovered_right_data], ignore_index=True).sort_values([0,3]) # sort according to column 0, then column 3
combined_insole_data = combined_insole_data.reset_index(drop=False)

## only keep insole data after the beginning index
left_start_timestamp = 100 # 81700
right_start_timestamp = 33825 # 85125
left_start_index = recovered_left_data.index[recovered_left_data[3] == left_start_timestamp].tolist()
right_start_index = recovered_right_data.index[recovered_right_data[3] == right_start_timestamp].tolist()
crop_left_begin = recovered_left_data.iloc[left_start_index[0]:, :].reset_index(drop=True)
crop_right_begin = recovered_right_data.iloc[right_start_index[0]:, :].reset_index(drop=True)

## align the end of sensor data
# add the number of data as a column for easier comparison
crop_left_begin.insert(loc=0, column='number', value=range(len(crop_left_begin)))
crop_right_begin.insert(loc=0, column='number', value=range(len(crop_right_begin)))
# observe to get the ending index
croped_insole_data = pd.concat([crop_left_begin, crop_right_begin], ignore_index=True).sort_values([0, 3]) # sort according to column 0, then column 3
croped_insole_data = croped_insole_data.reset_index(drop=False)

## only keep data before the ending index
left_end_timestamp = 28225  # 110750
right_end_timestamp = 61950  # 114175
left_end_index = crop_left_begin.index[crop_left_begin[3] == left_end_timestamp].tolist()
right_end_index = crop_right_begin.index[crop_right_begin[3] == right_end_timestamp].tolist()
crop_left_done = crop_left_begin.iloc[:left_end_index[0]+1, 1:].reset_index(drop=True) # remove the first column indicating data number
crop_right_done = crop_right_begin.iloc[:right_end_index[0]+1, 1:].reset_index(drop=True) # remove the first column indicating data number

## align EMG
# get the average beginning and ending insole timestamp
crop_left_done[0] = pd.to_datetime(crop_left_done[0], format='%Y-%m-%d_%H:%M:%S.%f')
crop_right_done[0] = pd.to_datetime(crop_right_done[0], format='%Y-%m-%d_%H:%M:%S.%f')
average_insole_begin = (crop_left_done.iloc[0,0] - crop_right_done.iloc[0,0]) / 2 + crop_right_done.iloc[0,0]
average_insole_end = (crop_left_done[0].iloc[-1] - crop_right_done[0].iloc[-1]) / 2 + crop_right_done[0].iloc[-1]
# only keep data between the beginning and ending index
raw_emg_data[0] = pd.to_datetime(raw_emg_data[0], format='%Y-%m-%d_%H:%M:%S.%f')
emg_start_index = (pd.to_datetime(raw_emg_data[0]) - average_insole_begin).abs().idxmin() # obtain the closet beginning timestamp
emg_end_index = (pd.to_datetime(raw_emg_data[0]) - average_insole_end).abs().idxmin() # obtain the closet ending timestamp
crop_emg_done = raw_emg_data.iloc[emg_start_index:emg_end_index+1, :].reset_index(drop=True)


## upsampling insole data
upsampled_left_insole = upsampleInsoleData(crop_left_done).reset_index()
upsampled_right_insole = upsampleInsoleData(crop_right_done).reset_index()


## filtering
# filter EMG signal
sos  = signal.butter(4, [20,400], fs = 2000, btype = "bandpass", output='sos')
emg_bandpass_filtered = signal.sosfiltfilt(sos, crop_emg_done.iloc[:,3:67], axis=0)
b, a  = signal.iircomb(50, 50, fs=2000, ftype='notch')
emg_notch_filtered = signal.filtfilt(b, a, pd.DataFrame(emg_bandpass_filtered), axis=0)

## filter insole signal after upsampling
sos  = signal.butter(4, [10], fs = 2000, btype = "lowpass", output='sos')
left_insole_filtered = signal.sosfiltfilt(sos, upsampled_left_insole.iloc[:,1:193], axis=0)
right_insole_filtered = signal.sosfiltfilt(sos, upsampled_right_insole.iloc[:,1:193], axis=0)

##
plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
pd.DataFrame(left_insole_filtered)[191].plot()
ax1 = plt.subplot(2, 1, 2)
upsampled_left_insole[195].plot()

## filter's frequency response plot
# sos  = signal.butter(4, [10], fs = 2000, btype = "lowpass", output='sos')
# w, h = signal.sosfreqz(sos, worN=2000)
# plt.plot(w, abs(h))
b, a  = signal.iircomb(50, 50, fs=2000, ftype='notch')
w, h = signal.freqz(b, a , worN=2000)
plt.plot(w, abs(h))



## insole
left_total_force = pd.DataFrame(left_insole_filtered)[191]
right_total_force = pd.DataFrame(right_insole_filtered)[191]
# sum_emg_data = crop_emg_done.iloc[:, 3:-4].sum(1)


# fig, axes = plt.subplots(nrows=3, ncols=1)
# left_total_force[5000:15000].plot(subplots = Ture, ax=axes[0,0])
# right_total_force[5000:15000].plot(subplots = Ture, ax=axes[1,0])
# sum_emg_data[5000:15000].plot(subplots = Ture, ax=axes[2,0])

sum_emg_data = pd.DataFrame(emg_notch_filtered).sum(1)
sum_emg_data.columns = ['emg value']

start = 6800
end = 7000

plt.figure(1)
ax1 = plt.subplot(3, 1, 1)
left_total_force.iloc[start:end].plot() #no need to specify for first axis
ax1.set_ylabel("force(kg)")
ax1.set_title('Left Insole Force')
ax2 = plt.subplot(3, 1, 2)
right_total_force.iloc[start:end].plot(ax=plt.gca())
ax2.set_ylabel("force(kg)")
ax2.set_title('Right Insole Force')
ax3 = plt.subplot(3, 1, 3)
sum_emg_data.iloc[start:end].plot(ax=plt.gca())
ax3.set_xlabel("Sample Number")
ax3.set_ylabel("Emg Value")
ax3.set_title('Emg Signal')


## emg
data_dir = 'D:\Data\Insole_Emg'
file_name = 'subject0_20220801_141429'
emg_file = f'emg\emg_{file_name}.csv'
emg_path = os.path.join(data_dir, emg_file)

async_emg_data = pd.read_csv(emg_path, sep=',', header=None, dtype='int16', converters={0: str, 1: str, 2: str})  # change data type for faster reading

async_emg_data = async_emg_data.iloc[:,3:-4]
async_emg_data.iloc[00000:20000,0:5].plot(subplots=True)
plt.tight_layout()
plt.show()

##
a = async_emg_data.sum(1)
plt.figure()
a.iloc[5000:55000].plot(subplots=True)
plt.show()

# ##
# data_dir = 'D:\Data\Insole_Emg'
# file_name = 'subject0_20220731_182004'
# emg_file = f'emg\emg_{file_name}.csv'
# emg_path = os.path.join(data_dir, emg_file)
#
# ot_emg_data = pd.read_csv(emg_path, sep=';', header=None)  # change data type for faster reading
# ot_emg_data = ot_emg_data.iloc[:,1:]
# ot_emg_data.iloc[00000:20000,0:5].plot(subplots=True)
# plt.tight_layout()
# plt.show()
#
# ##
# b = ot_emg_data.sum(1)
# plt.figure()
# b.iloc[5000:45000].plot(subplots=True)
# plt.show()