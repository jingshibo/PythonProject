import pandas as pd
import numpy as np

## insert missing insole data
def insertMissingRow(raw_insole_data, sampling_period):
    time_stamp = raw_insole_data.iloc[:, 3]
    for i in range(len(time_stamp) - 1):  # loop over all rows
        num_to_insert = int((time_stamp[i + 1] - time_stamp[i]) / sampling_period - 1)  # number of rows to insert
        for j in range(num_to_insert):
            nan_to_filling = ((raw_insole_data.shape[1] - 4) * [np.NaN])  # insert NaN value to missing row
            row_to_insert = [raw_insole_data.iloc[i, 0], 0, raw_insole_data.iloc[i, 2], time_stamp[i] + sampling_period * (j + 1),
                             *nan_to_filling]  # construct a row list
            raw_insole_data.loc[i + (j + 1) / (num_to_insert + 1)] = row_to_insert  # Append row at the bottom with a given index
    inserted_left_data = raw_insole_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
    recovered_left_data = inserted_left_data.interpolate(method='pchip', limit_direction='forward', axis=0)
    return recovered_left_data

## upsample insole data
def upsampleInsoleData(insole_data):
    only_measured_data = insole_data.iloc[:, 3:] # extract only measurement value columns
    only_measured_data.iloc[:, 0] = pd.to_datetime(only_measured_data.iloc[:, 0], unit='ms') # convert timestamp string to datetime object
    only_measured_data = only_measured_data.set_index([3]) # set the timestamp column as datetime index
    upsampled_sensor_data = only_measured_data.resample('0.5ms').asfreq()  # upsampling to every 0.5ms to match EMG
    upsampled_sensor_data = upsampled_sensor_data.interpolate(method='pchip', limit_direction='forward', axis=0) # impute the NaN missing values
    return upsampled_sensor_data