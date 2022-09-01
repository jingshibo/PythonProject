import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

## insert missing insole data
def insertInsoleMissingRow(raw_insole_data, sampling_period):
    time_stamp = raw_insole_data.iloc[:, 3]  # the forth column is the timestamp
    for i in range(len(time_stamp) - 1):  # loop over all rows
        num_to_insert = int((time_stamp[i + 1] - time_stamp[i]) / sampling_period - 1)  # number of rows to insert
        for j in range(num_to_insert):
            nan_to_filling = ((raw_insole_data.shape[1] - 4) * [np.NaN])  # insert NaN value to missing row
            row_to_insert = [raw_insole_data.iloc[i, 0], 0, raw_insole_data.iloc[i, 2], time_stamp[i] + sampling_period * (j + 1),
                             *nan_to_filling]  # construct a row list. the first four columns are not measurements that needs to interpolate
            raw_insole_data.loc[i + (j + 1) / (num_to_insert + 1)] = row_to_insert  # Append row at the bottom with a given index
    inserted_insole_data = raw_insole_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
    return inserted_insole_data, raw_insole_data

## upsample insole data to every 0.5ms to match EMG (abandoned method, because the insole sampling rate is not exact 40Hz)
def upsampleInsoleData(insole_data):
    only_measured_data = insole_data.iloc[:, 3:]  # extract only measurement value columns
    only_measured_data.iloc[:, 0] = pd.to_datetime(only_measured_data.iloc[:, 0], unit='ms') # convert timestamp string to datetime object
    only_measured_data = only_measured_data.set_index([3]) # set the timestamp column as datetime index
    upsampled_sensor_data = only_measured_data.resample('0.5ms').asfreq()  # insert row of NaN every 0.5ms
    upsampled_sensor_data = upsampled_sensor_data.interpolate(method='pchip', limit_direction='forward', axis=0) # impute the NaN missing values
    return upsampled_sensor_data

## upsample insole data to exactly the same number as EMG
def upsampleInsoleEqualToEMG(insole_data, emg_data):
    x = np.arange(len(insole_data))
    y = insole_data.iloc[:, 3:].to_numpy()  # only extract measurement value columns
    f = PchipInterpolator(x, y)
    x_upsampled = np.linspace(min(x), max(x), len(emg_data))
    y_upsampled = f(x_upsampled)
    insole_upsampled = pd.DataFrame(y_upsampled)
    return insole_upsampled

## find lost emg data according to the value in sample counter
def findLostEmgData(raw_emg_data):
    sample_counter = raw_emg_data.iloc[:, -1].to_frame()  # the final column is the sample counter (timestamp)
    sample_counter.columns = ["sample_counter"]
    counter_diff = sample_counter["sample_counter"].diff().to_frame()
    counter_diff.columns = ["counter_difference"]
    wrong_timestamp = counter_diff.query('abs(counter_difference) > 1 & abs(counter_difference) < 65535') # exclude 65535 as it is when the counter restarts
    return wrong_timestamp  # return the index of wrong timestamps

## insert value of Nan with the number of missing rows to EMG
def insertEmgNanRow(emg_data, start_index, end_index):
    # Note: here we use .loc instead of .iloc in order to use the original row index to reach the data.
    start_sample_count = emg_data.loc[start_index, 70]  # the column named 70 is timestamp.
    end_sample_count = emg_data.loc[end_index, 70]
    # Warning: the following code only applies to when the number of lost data is less than 131072 (65s), which is usually satisfied
    if end_sample_count > start_sample_count:
        number_to_insert = end_sample_count - start_sample_count - 1
    else:  # the sample counter starts over when the count number > 65536
        number_to_insert = end_sample_count - start_sample_count - 1 + 65536
    for j in range(int(number_to_insert)):
        nan_to_filling = ((emg_data.shape[1] - 3) * [np.NaN])  # insert NaN value to missing row
        row_to_insert = [emg_data.loc[start_index, 0], emg_data.loc[start_index, 1], emg_data.loc[start_index, 2], *nan_to_filling]  # construct a row list
        emg_data.loc[start_index+(j+1) / (number_to_insert+1)] = row_to_insert  # Append row at the bottom with a given index
    return emg_data

