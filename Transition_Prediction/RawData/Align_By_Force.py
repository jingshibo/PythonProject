"""
run the following code blocks in order:
based on insole force value to decide the start and end row index for both left and right insoles to align them.
based on sync force value to align emg with insole data
save the alignment results into disk for later use
"""

## import modules
import os
import pandas as pd
import datetime
from Transition_Prediction.RawData.Utility_Functions import Insole_Emg_Alignment, Two_Insoles_Alignment, Upsampling_Filtering, \
    Insole_Emg_Recovery

## initialization
subject = 'Number1'
version = 0
mode = 'up_down'
session = 2

# subject = 'Zehao'
# version = 0
# mode = 'up_down'
# session = 7

data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}'
data_file_name = f'subject_{subject}_Experiment_{version}_session_{session}_{mode}'

left_insole_file = f'raw_{mode}\left_insole\left_{data_file_name}.csv'
right_insole_file = f'raw_{mode}\\right_insole\\right_{data_file_name}.csv'
emg_file = f'raw_{mode}\emg\emg_{data_file_name}.csv'

left_insole_path = os.path.join(data_dir, left_insole_file)
right_insole_path = os.path.join(data_dir, right_insole_file)
emg_path = os.path.join(data_dir, emg_file)

raw_left_data = []
raw_right_data = []
raw_emg_data = []


## read and impute sensor data
insole_sampling_period = 25  # insole sampling period
now = datetime.datetime.now()

# left insole
raw_left_data = pd.read_csv(left_insole_path, sep=',', header=None)
inserted_left_data, appended_left_data = Insole_Emg_Recovery.insertInsoleMissingRow(raw_left_data, insole_sampling_period)  # add missing rows with NaN values
recovered_left_data = inserted_left_data.interpolate(method='pchip', limit_direction='forward', axis=0)  # interpolate Nan value

# right insole
raw_right_data = pd.read_csv(right_insole_path, sep=',', header=None)
inserted_right_data, appended_right_data = Insole_Emg_Recovery.insertInsoleMissingRow(raw_right_data, insole_sampling_period)  # add missing rows with NaN values
recovered_right_data = inserted_right_data.interpolate(method='pchip', limit_direction='forward', axis=0)  # interpolate Nan value

# emg
raw_emg_data = pd.read_csv(emg_path, sep=',', header=None, dtype='int16', converters={0: str, 1: str, 2: str})  # change data type for faster reading
wrong_timestamp_sync, wrong_timestamp_emg1, wrong_timestamp_emg2 = Insole_Emg_Recovery.findLostEmgData(raw_emg_data)

# for single emg device data
# sample_counter = raw_emg_data.iloc[:, -1].to_frame()  # this column is the sample counter (timestamp) for SyncStation
# sample_counter.columns = ["sample_count"]
# counter_diff = sample_counter["sample_count"].diff().to_frame()
# counter_diff.columns = ["count_difference"]
# wrong_timestamp = counter_diff.query('count_difference != 1 & count_difference != -65535')  # exclude 65535 as it is when the counter restarts

recovered_emg_data = raw_emg_data  # if there are no abnormal emg data numbers
print(datetime.datetime.now() - now)


## view raw emg and insole data
# plot emg and insole data
Insole_Emg_Alignment.plotAllSensorData(recovered_emg_data, recovered_left_data, recovered_right_data, 0, -1)
# check the number of emg data
emg_timestamp = pd.to_datetime(recovered_emg_data[0], format='%Y-%m-%d_%H:%M:%S.%f')
expected_number = (emg_timestamp.iloc[-10000] - emg_timestamp.iloc[10000]).total_seconds() * 1000 * 2 # the number of emg value expected within the period
real_number = len(emg_timestamp) - 20000
print("expected emg number:", expected_number, "real emg number:", real_number, "missing emg number:", expected_number - real_number)
# check missing channels (all values are zeros)
for column in recovered_emg_data:
    if (recovered_emg_data[column] == 0).all() and column != 69 and column != 70 and column != 139 and column != 140:
        print( "missing channels:", column)


## align two insoles based on the force value
# concat two insole data into one dataframe to check timestamps when necessary
combined_insole_data = Two_Insoles_Alignment.cancatInsole(recovered_left_data, recovered_right_data)
# plot the total force of two insoles for insole alignment
start_index = 0
end_index = -1
Two_Insoles_Alignment.plotBothInsoles(recovered_left_data, recovered_right_data, start_index, end_index)


## align the beginning of two insoles
# view the insole force plotted above to find an appropriate start index that matches most force pulses
# usually use the first pulse as the referenece. if the result is undesired, just add or minus one on the index to adjust, instead of selecting anather pulse
# Note: it is the alignment later with emg data that will decide whether add one or minus one is better. But here to align only two insoles, you can do it arbitrarily.
right_start_index = 155
left_start_index = 134
combine_begin_cropped, left_begin_cropped, right_begin_cropped = Two_Insoles_Alignment.alignInsoleBeginIndex(left_start_index,
    right_start_index, recovered_left_data, recovered_right_data)
# check the following timestamps in combined_insole_data table when necessary
print("right_start_timestamp:", recovered_right_data.loc[right_start_index, 3])
print("left_start_timestamp:", recovered_left_data.loc[left_start_index, 3])
# plot begin-cropped insole data for ending index alignment
start_index = 0
end_index = -1
Two_Insoles_Alignment.plotBothInsoles(left_begin_cropped, right_begin_cropped, start_index, end_index)


## align the ending of two insoles
# view the insole force plotted above to find an appropriate end index that matches most force pulses. usually use the last pulse as the referenece.
# Note: scale the figure up to large enough in order to look into the alignment result more clearly.
# it is the alignment later with emg data that will decide whether the end_index here should add or minus one to match the end of emg index better.
right_end_index = 7073
left_end_index = 7073
left_insole_aligned, right_insole_aligned = Two_Insoles_Alignment.alignInsoleEndIndex(left_end_index, right_end_index,
    left_begin_cropped, right_begin_cropped)
# display the eng_index before data cropping. this can be used to obtain the pulse number for insole/emg alignment
print("right_end_index_before_cropping:", right_end_index + right_start_index)
print("left_end_index_before_cropping:", left_end_index + left_start_index)
# plot insole and sync force data for insole/emg alignment.
start_index = 0
end_index = -1
Insole_Emg_Alignment.plotInsoleSyncForce(recovered_emg_data, recovered_left_data, recovered_right_data, start_index, end_index)


## align the insole and emg data based on force value
# view the sync force plotted above to find an appropriate start and end index for alignment of emg and insoles
# Note:  you may need to adjust the start and end index of insoles (treat two insoles as one) in order to match the corresponding emg index.
# Usually use the position of sync force as the true value, which is more accurate. The start index and end index of the insole pair should be decided separately.
emg_start_index = 11253  # select the index belonging to the pulse number where the insole start_index is
emg_end_index = 364804  # select the index belonging to the pulse number where the insole end_index is
emg_aligned = recovered_emg_data.iloc[emg_start_index:emg_end_index+1, :].reset_index(drop=True)
# convert the timestamp column to datetime type
emg_aligned[0] = pd.to_datetime(emg_aligned[0], format='%Y-%m-%d_%H:%M:%S.%f')

# plot the insole and emg alignment results
start_index = 0
end_index = -1
# upsampling aligned insole data (this is not for alignment. rather it is only for plotting to check the alignment result)
left_insole_upsampled, right_insole_upsampled = Upsampling_Filtering.upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned)
# plot emg and insole data
Insole_Emg_Alignment.plotInsoleAlignedEmg(emg_aligned, left_insole_upsampled, right_insole_upsampled, start_index, end_index, sync_force=True)


## save the aligned results (parameters and data)
Insole_Emg_Alignment.saveAlignParameters(subject, data_file_name, left_start_index, right_start_index, left_end_index,
    right_end_index, emg_start_index, emg_end_index)
Insole_Emg_Alignment.saveAlignedData(subject, session, mode, version, left_insole_aligned, right_insole_aligned, emg_aligned)


## read alignment parameters
right_start, left_start, right_end, left_end, emg_start, emg_end = Insole_Emg_Alignment.readAlignParameters(subject, session, mode, version)
print(right_start, left_start, right_end, left_end, emg_start, emg_end)


## align insole and EMG based on timestamps (if there is no sync force to use)
emg_aligned_2 = Insole_Emg_Alignment.alignInsoleEmgTimestamp(recovered_emg_data, left_insole_aligned, right_insole_aligned)
# upsampling and filtering aligned data
left_insole_upsampled, right_insole_upsampled = Upsampling_Filtering.upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned_2)
# left_insole_filtered, right_insole_filtered = Upsampling_Filtering.filterInsole(left_insole_upsampled, right_insole_upsampled)
emg_filtered = Upsampling_Filtering.filterEmg(emg_aligned_2.iloc[:, 3:67], notch=False, quality_factor=30)
# check the insole and emg alignment results
start_index = 0
end_index = -1
# plot the emg and insole data to check the alignment result
Insole_Emg_Alignment.plotInsoleAlignedEmg(emg_aligned_2, left_insole_upsampled, right_insole_upsampled, start_index, end_index,
    sync_force=False, emg_columns=range(3, 67))


## test
# Insole_Emg_Alignment.plotInsoleSyncForce(emg_aligned, recovered_left_data, recovered_right_data, 0, -1)


## recover emg signal
# check each row in wrong_timestamp variable to figure out the reason for the abnormal number

# 1. change column order
# start_index = 589999
# reordered_emg_data = Insole_Emg_Recovery.changeColumnOrder(raw_emg_data, start_index)
# start_index = 632360
# reordered_emg_data = Insole_Emg_Recovery.changeColumnOrder(reordered_emg_data, start_index)
# recovered_emg_data = reordered_emg_data
# wrong_timestamp_sync, wrong_timestamp_emg1, wrong_timestamp_emg2 = Insole_Emg_Recovery.findLostEmgData(recovered_emg_data)

# # 2. how to deal with duplicated data: delete all these data and use the first data and the last data as reference to count the number of
# # missing data, then insert Nan of this number
# now = datetime.datetime.now()
# # delete duplicated data
# start_index = 588863  # the index before the first duplicated data
# end_index = 590748  # the index after the last duplicated data.
# # Note: it only drop data between start_index+1 ~ end_index-1. the last index will not be dropped.
# dropped_emg_data = raw_emg_data.drop(raw_emg_data.index[range(start_index+1, end_index)])
# inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(dropped_emg_data, start_index, end_index)
#
# # 3. how to deal with lost data: calculate the number of missing data and then insert Nan of the same number
# start_index = 588775  # the last index before the missing data
# end_index = 588776  # the first index after the missing data
# inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(inserted_emg_data, start_index, end_index)
#
# # interpolate emg data
# reindex_emg_data = inserted_emg_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
# recovered_emg_data = reindex_emg_data.interpolate(method='cubic', limit_direction='forward', axis=0)
# print(datetime.datetime.now() - now)