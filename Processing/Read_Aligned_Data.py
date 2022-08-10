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


## read raw sensor data
# initialization
data_dir = 'D:\Data\Insole_Emg'
data_file_name = "subject0_20220801_110815"
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

# emg
now = datetime.datetime.now()
raw_emg_data = pd.read_csv(emg_path, sep=',', header=None, dtype='int16',
                           converters={0: str, 1: str, 2: str})  # change data type for faster reading
print(datetime.datetime.now() - now)
# left insole
raw_left_data = pd.read_csv(left_insole_path, sep=',', header=None)
recovered_left_data = insertMissingRow(raw_left_data, insole_sampling_period)  # add missing rows with NaN values
# right insole
raw_right_data = pd.read_csv(right_insole_path, sep=',', header=None)
recovered_right_data = insertMissingRow(raw_right_data, insole_sampling_period)  # add missing rows with NaN values


## read alignment data
data_dir = 'D:\Data\Insole_Emg'
alignment_file_name = 'subject0.csv'
alignment_file_path = os.path.join(data_dir, alignment_file_name)
alignment_data = pd.read_csv(alignment_file_path, sep=',') # header exists

file_parameter = alignment_data.query('data_file_name == data_file_name')
align_parameter = file_parameter.iloc[[-1]]  # extract the last row (apply the newest parameters)
left_start_timestamp = align_parameter['left_start_timestamp'].iloc[0]
right_start_timestamp = align_parameter['right_start_timestamp'].iloc[0]
left_end_timestamp = align_parameter['left_end_timestamp'].iloc[0]
right_end_timestamp = align_parameter['right_end_timestamp'].iloc[0]
print(left_start_timestamp, right_start_timestamp, left_end_timestamp, right_end_timestamp)


## read aligned sensor data
_, left_cropped_begin, right_cropped_begin = Align_Two_Insoles.alignInsoleBegin(  # to view combine_cropped_begin data
                        left_start_timestamp, right_start_timestamp, recovered_left_data,  recovered_right_data)
left_insole_aligned, right_insole_aligned = Align_Two_Insoles.alignInsoleEnd(left_end_timestamp,
                                right_end_timestamp, left_cropped_begin, right_cropped_begin)
emg_aligned = Align_Insole_Emg.alignInsoleEmg(raw_emg_data, left_insole_aligned, right_insole_aligned)

## preprocess sensor data
# upsampling insole data
left_insole_upsampled, right_insole_upsampled = Align_Two_Insoles.upsampleInsole(left_insole_aligned, right_insole_aligned)
# filtering emg data
emg_filtered = Align_Insole_Emg.filterEmg(emg_aligned)