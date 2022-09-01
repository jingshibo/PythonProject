"""
run the following code blocks in order:
input the start and end row index for both left and right insoles to align them.
the emg signal will be automatically aligned using the timestamp information
save the alignment parameters into a csv file for later use
"""

## import modules
import os
import pandas as pd
import datetime
from RawData.Utility_Functions import Two_Insoles_Alignment, Insole_Emg_Alignment, Insole_Emg_Recovery, Upsampling_Filtering

## initialization
subject = 'Test'
mode = 'up_down'
session = 5

data_dir = 'D:\Data\Insole_Emg'
data_file_name = f'subject_{subject}_session_{session}_{mode}'

left_insole_file = f'subject_{subject}\\raw_{mode}\left_insole\left_{data_file_name}.csv'
right_insole_file = f'subject_{subject}\\raw_{mode}\\right_insole\\right_{data_file_name}.csv'
emg_file = f'subject_{subject}\\raw_{mode}\emg\emg_{data_file_name}.csv'

left_insole_path = os.path.join(data_dir, left_insole_file)
right_insole_path = os.path.join(data_dir, right_insole_file)
emg_path = os.path.join(data_dir, emg_file)

raw_left_data = []
raw_right_data = []
raw_emg_data = []

## read and impute sensor data
insole_sampling_period = 20  # insole sampling period
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
wrong_timestamp = Insole_Emg_Recovery.findLostEmgData(raw_emg_data)
recovered_emg_data = raw_emg_data  # if there are no abnormal emg data numbers
print(datetime.datetime.now() - now)

## recover emg signal (both deleting duplicated data and add missing data)
# check each row in wrong_timestamp variable to see if there are lost data or duplicated data
# how to deal with duplicated data: delete all these data and use the first data and the last data as reference to count the number of
# missing data, then insert Nan of this number
# now = datetime.datetime.now()
# # delete duplicated data
# start_index = 588863  # the index before the first duplicated data
# end_index = 590748  # the index after the last duplicated data.
# # Note: it only drop data between start_index+1 ~ end_index-1. the last index will not be dropped.
# dropped_emg_data = raw_emg_data.drop(raw_emg_data.index[range(start_index+1, end_index)])
# inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(dropped_emg_data, start_index, end_index)
#
# # how to deal with lost data: calculate the number of missing data and then insert Nan of the same number
# start_index = 588775  # the last index before the missing data
# end_index = 588776  # the first index after the missing data
# inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(inserted_emg_data, start_index, end_index)
#
# # # interpolate emg data
# reindex_emg_data = inserted_emg_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
# recovered_emg_data = reindex_emg_data.interpolate(method='cubic', limit_direction='forward', axis=0)
# print(datetime.datetime.now() - now)


## align two insoles based on the force value
# concat two insole data into one dataframe to check timestamps when necessary
combined_insole_data = Two_Insoles_Alignment.cancatInsole(recovered_left_data, recovered_right_data)
# plot the total force of two insoles for alignment
start_index = 0
end_index = -1
Two_Insoles_Alignment.plotBothInsoles(recovered_left_data, recovered_right_data, start_index, end_index)


## align the beginning of insole data
# view the insole force plotted above to find an appropriate start index for alignment of two insoles
left_start_index = 579  # second pulse
right_start_index = 557
# view combine_cropped_begin data table to check timestamps when necessary
combine_begin_cropped, left_begin_cropped, right_begin_cropped = Two_Insoles_Alignment.alignInsoleBeginIndex(left_start_index,
    right_start_index, recovered_left_data, recovered_right_data)
# plot begin-cropped insole data to find an appropriate end index for alignment of two insoles
start_index = 0
end_index = -1
Two_Insoles_Alignment.plotBothInsoles(left_begin_cropped, right_begin_cropped, start_index, end_index)


## align the ending of insole data
# view the insole force plotted above for alignment
left_end_index = 15725  # third pulse
right_end_index = 15725
left_insole_aligned, right_insole_aligned = Two_Insoles_Alignment.alignInsoleEndIndex(left_end_index, right_end_index,
    left_begin_cropped, right_begin_cropped)
# plot insole and sync force data for alignment
start_index = 0
end_index = -1
Insole_Emg_Alignment.plotInsoleSyncForce(recovered_emg_data, recovered_left_data, recovered_right_data, start_index, end_index)


## align the insole and emg data based on force value
# view the sync force plotted above to find an appropriate start and index for alignment of insoles and emg
emg_start_index = 29355
emg_end_index = 658038
emg_aligned = recovered_emg_data.iloc[emg_start_index:emg_end_index+1, :].reset_index(drop=True)
emg_aligned[0] = pd.to_datetime(emg_aligned[0], format='%Y-%m-%d_%H:%M:%S.%f') # convert timestamp to datetime type


## upsampling aligned insole data
left_insole_upsampled, right_insole_upsampled = Upsampling_Filtering.upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned)
# plot the insole and emg alignment results
start_index = 0
end_index = -1
Insole_Emg_Alignment.plotInsoleEmg(emg_aligned, left_insole_upsampled, right_insole_upsampled, start_index, end_index, sync_force=True)


## align insole and EMG based on timestamps (if there is not sync force)
emg_aligned_2 = Insole_Emg_Alignment.alignInsoleEmgTimestamp(recovered_emg_data, left_insole_aligned, right_insole_aligned)
# upsampling and filtering aligned data
left_insole_upsampled, right_insole_upsampled = Upsampling_Filtering.upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned_2)
# left_insole_filtered, right_insole_filtered = Upsampling_Filtering.filterInsole(left_insole_upsampled, right_insole_upsampled)
emg_filtered = Upsampling_Filtering.filterEmg(emg_aligned_2, emg_id=1, notch=False, quality_factor=30)
# check the insole and emg alignment results
start_index = 0
end_index = -1
# plot the emg and insole data to check the alignment result
Insole_Emg_Alignment.plotInsoleEmg(emg_aligned_2, left_insole_upsampled, right_insole_upsampled, start_index, end_index, sync_force=True)

## save the align results
 # save the alignment parameters
Two_Insoles_Alignment.saveAlignParameters(subject, data_file_name, left_start_index, right_start_index, left_end_index,
    right_end_index, emg_start_index, emg_end_index)
# save the aligned data
Insole_Emg_Alignment.saveAlignedData(subject, session, mode, left_insole_aligned, right_insole_aligned, emg_aligned)
