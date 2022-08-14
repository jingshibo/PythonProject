## import modules
import os
import pandas as pd
import datetime
from PreProcessing.Utility import Align_Two_Insoles, Align_Insole_Emg, Recover_Insole, Split_Insole_Data
import json

## read raw sensor data
# initialization
data_dir = 'D:\Data\Insole_Emg'
data_file_name = "subject0_20220810_205047"
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
# left insole
raw_left_data = pd.read_csv(left_insole_path, sep=',', header=None)
recovered_left_data = Recover_Insole.insertMissingRow(raw_left_data,
    insole_sampling_period)  # add missing rows with NaN values
# right insole
raw_right_data = pd.read_csv(right_insole_path, sep=',', header=None)
recovered_right_data = Recover_Insole.insertMissingRow(raw_right_data,
    insole_sampling_period)  # add missing rows with NaN values
print(datetime.datetime.now() - now)

## read alignment data
data_dir = 'D:\Data\Insole_Emg'
alignment_file_name = 'subject0.csv'
alignment_file_path = os.path.join(data_dir, alignment_file_name)
alignment_data = pd.read_csv(alignment_file_path, sep=',')  # header exists

file_parameter = alignment_data.query('data_file_name == data_file_name')
align_parameter = file_parameter.iloc[[-1]]  # extract the last row (apply the newest parameters)

left_start_timestamp = align_parameter['left_start_timestamp'].iloc[0]
right_start_timestamp = align_parameter['right_start_timestamp'].iloc[0]
left_end_timestamp = align_parameter['left_end_timestamp'].iloc[0]
right_end_timestamp = align_parameter['right_end_timestamp'].iloc[0]
print(left_start_timestamp, right_start_timestamp, left_end_timestamp, right_end_timestamp)

## read aligned sensor data
_, left_begin_cropped, right_begin_cropped = Align_Two_Insoles.alignInsoleBegin(  # to view combine_cropped_begin data
    left_start_timestamp, right_start_timestamp, recovered_left_data, recovered_right_data)
left_insole_aligned, right_insole_aligned = Align_Two_Insoles.alignInsoleEnd(left_end_timestamp, right_end_timestamp,
    left_begin_cropped, right_begin_cropped)
emg_aligned = Align_Insole_Emg.alignInsoleEmg(raw_emg_data, left_insole_aligned, right_insole_aligned)

## preprocess sensor data
# upsampling insole data
left_insole_upsampled, right_insole_upsampled = Align_Two_Insoles.upsampleInsole(left_insole_aligned,
    right_insole_aligned, emg_aligned)
# filtering emg data
emg_filtered = Align_Insole_Emg.filterEmg(emg_aligned)

## if have not splited data yet, plot sensor data to split gait cycles
start_index = 00000
end_index = 600000
left_force_baseline = 4.5
right_force_baseline = 4.5
Split_Insole_Data.plotSplitLine(emg_filtered, left_insole_upsampled, right_insole_upsampled, start_index, end_index,
    left_force_baseline, right_force_baseline)


## split result path
subject = 'Shibo'
data_dir = 'D:\Data\Insole_Emg'
data_file_name = f'subject_{subject}_split'
parameter_file = f'Subject_{subject}\{data_file_name}.json'
parameter_path = os.path.join(data_dir, parameter_file)

## read split results from json files
with open(parameter_path) as json_file:
    Shibo_preprocessed_results = json.load(json_file)

##
