"""
run the following code blocks in order:
input the start and end timestamp for both left and right insoles to align them.
the emg signal will be automatically aligned using this information
save the alignment parameters into disk for later use
"""

## import modules
import os
import pandas as pd
import datetime
from Transition_Prediction.RawData.Utility_Functions import Insole_Emg_Alignment, Two_Insoles_Alignment, Upsampling_Filtering, \
    Insole_Emg_Recovery

## initialization
subject = 'Shibo'
mode = 'up_down'
session = 1

data_dir = 'D:\Data\Insole_Emg'
data_file_name = f'subject_{subject}_session_{session}_{mode}'

# data_file_name = 'subject0_20220810_205047'
# left_insole_file = f'left_insole\left_{data_file_name}.csv'
# right_insole_file = f'right_insole\\right_{data_file_name}.csv'
# emg_file = f'emg\emg_{data_file_name}.csv'

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
insole_sampling_period = 25  # insole sampling period
now = datetime.datetime.now()

# left insole
raw_left_data = pd.read_csv(left_insole_path, sep=',', header=None)
recovered_left_data = Insole_Emg_Recovery.insertInsoleMissingRow(raw_left_data, insole_sampling_period)  # add missing rows with NaN values

# right insole
raw_right_data = pd.read_csv(right_insole_path, sep=',', header=None)
recovered_right_data = Insole_Emg_Recovery.insertInsoleMissingRow(raw_right_data, insole_sampling_period)  # add missing rows with NaN values

# emg
raw_emg_data = pd.read_csv(emg_path, sep=',', header=None, dtype='int16', converters={0: str, 1: str, 2: str})  # change data type for faster reading
wrong_timestamp = Insole_Emg_Recovery.findLostEmgData(raw_emg_data)  # check each row to see if there are lost data or duplicated data
print(datetime.datetime.now() - now)

## recover emg signal (both deleting duplicated data and add missing data)
# how to deal with duplicated data: delete all these data and use the first data and the last data as reference to count the number of
# missing data, then insert Nan of this number
now = datetime.datetime.now()
# delete duplicated data
start_index = 588863  # the index before the first duplicated data
end_index = 590748  # the index after the last duplicated data.
# Note: it only drop data between start_index+1 ~ end_index-1. the last index will not be dropped.
dropped_emg_data = raw_emg_data.drop(raw_emg_data.index[range(start_index+1, end_index)])
inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(dropped_emg_data, start_index, end_index)

# how to deal with lost data: calculate the number of missing data and then insert Nan of the same number
start_index = 588775  # the last index before the missing data
end_index = 588776  # the first index after the missing data
inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(inserted_emg_data, start_index, end_index)

# # interpolate emg data
reindex_emg_data = inserted_emg_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
recovered_emg_data = reindex_emg_data.interpolate(method='cubic', limit_direction='forward', axis=0)
print(datetime.datetime.now() - now)

## comcat two insole data into one dataframe
combined_insole_data = Two_Insoles_Alignment.cancatInsole(recovered_left_data, recovered_right_data)

## align the begin of sensor data
# view the combined_insole_data table in order to find the appropriate start timestamp for alignment
left_start_timestamp = 331775
right_start_timestamp = 313275
# to view combine_cropped_begin data
combine_cropped_begin, left_begin_cropped, right_begin_cropped = Two_Insoles_Alignment.alignInsoleBeginTimestamp(left_start_timestamp,
    right_start_timestamp, recovered_left_data, recovered_right_data)

## align the end of sensor data
# view the combine_cropped_begin table in order to find the appropriate end timestamp for alignment
left_end_timestamp = 748500
right_end_timestamp = 730025
left_insole_aligned, right_insole_aligned = Two_Insoles_Alignment.alignInsoleEndTimestamp(left_end_timestamp, right_end_timestamp,
    left_begin_cropped, right_begin_cropped)

## check the insole alignment results
start_index = 0
end_index = 600000
# plot the insole data to check the alignment result
Two_Insoles_Alignment.plotBothInsoles(left_insole_aligned, right_insole_aligned, start_index, end_index)

## align insole and EMG
emg_aligned = Insole_Emg_Alignment.alignInsoleEmgTimestamp(recovered_emg_data, left_insole_aligned, right_insole_aligned)

## upsampling and filtering aligned data
left_insole_upsampled, right_insole_upsampled = Upsampling_Filtering.upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned)
# left_insole_filtered, right_insole_filtered = Upsampling_Filtering.filterInsole(left_insole_upsampled, right_insole_upsampled)
emg_filtered = Upsampling_Filtering.filterEmg(emg_aligned.iloc[:, 3:67], notch=False, quality_factor=30)

## check the insole and emg alignment results
start_index = 00000
end_index = 600000
# plot the emg and insole data to check the alignment result
Insole_Emg_Alignment.plotInsoleAlignedEmg(emg_filtered, left_insole_upsampled, right_insole_upsampled, start_index, end_index)

## save the align results
#  save the alignment parameters
Two_Insoles_Alignment.saveAlignParameters(subject, data_file_name, left_start_timestamp, right_start_timestamp, left_end_timestamp,
    right_end_timestamp)
# save the aligned data
Insole_Emg_Alignment.saveAlignedData(subject, session, mode, left_insole_aligned, right_insole_aligned, emg_aligned)

# ## plot sensor data
# start_index = 00000
# end_index = 610000
#
# left_total_force = left_insole_upsampled[195]
# right_total_force = right_insole_upsampled[195]
# emg_data = emg_filtered.sum(1)
#
# # plot
# fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
# axes[0].plot(range(len(left_total_force.iloc[start_index:end_index])), left_total_force.iloc[start_index:end_index],
#              label="Left Insole Force")
# axes[0].plot(range(len(right_total_force.iloc[start_index:end_index])),
#              right_total_force.iloc[start_index:end_index], label="Right Insole Force")
# axes[1].plot(range(len(emg_data.iloc[start_index:end_index])), emg_data.iloc[start_index:end_index],
#              label="Emg Signal")
#
# axes[0].set(title="Insole Force", ylabel="force(kg)")
# axes[1].set(title="Emg Signal", xlabel="Sample Number", ylabel="Emg Value")
#
# axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels
#
# axes[0].legend(loc="upper right")
# axes[1].legend(loc="upper right")
#
#
#
# ## test
# data = emg_aligned
# # date1 = datetime.datetime.strptime(data.iloc[600, 0], "%Y-%m-%d_%H:%M:%S.%f")
# # date2 = datetime.datetime.strptime(data.iloc[-1, 0], "%Y-%m-%d_%H:%M:%S.%f")
# date1 = data.iloc[0, 0]
# date2 = data.iloc[-1, 0]
# print((date2 - date1).total_seconds() * 1000 * 2)


## check which part of the EMG data goes wrong
lower_limit = pd.DataFrame({'baseline': ['0:00:00.07']})
lower_limit = pd.to_datetime(lower_limit.iloc[:, 0]).to_frame().iloc[0, 0]
higher_limit = pd.DataFrame({'baseline': ['0:00:00.14']})
higher_limit = pd.to_datetime(higher_limit.iloc[:, 0]).to_frame().iloc[0, 0]

interval = pd.to_datetime(emg_aligned.iloc[:, 1]).to_frame()
interval.columns = ['interval']
unusual_lower_interval = interval.query('interval < @lower_limit')
unusual_higher_interval = interval.query('interval > @higher_limit')

## calcutate the expected number of emg data
first_index = unusual_higher_interval.index[0] - 1
last_index = unusual_lower_interval.index[-1] + 1
real_number = last_index - first_index - 1
expected_number = (emg_aligned.iloc[last_index, 0] - emg_aligned.iloc[first_index, 0]).total_seconds() * 1000 * 2
diff_number = real_number - expected_number



##
datetime_end = datetime.datetime.strptime(raw_emg_data.iloc[-1,0], '%Y-%m-%d_%H:%M:%S.%f')
datetime_start = datetime.datetime.strptime(raw_emg_data.iloc[400,0], '%Y-%m-%d_%H:%M:%S.%f')
expected_number = (datetime_end - datetime_start).total_seconds() * 1000 * 2
print(expected_number)
real_number = len(recovered_emg_data) - 400
print(real_number)
diff_number = real_number - expected_number
print(diff_number)