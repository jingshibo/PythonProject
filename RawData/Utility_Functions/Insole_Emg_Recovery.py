import pandas as pd
import numpy as np

## insert missing insole data
def insertInsoleMissingRow(raw_insole_data, sampling_period):
    time_stamp = raw_insole_data.iloc[:, 3]  # the forth column is the timestamp
    total_num_inserted = 0
    for i in range(len(time_stamp) - 1):  # loop over all rows
        num_to_insert = int((time_stamp[i + 1] - time_stamp[i]) / sampling_period - 1)  # number of rows to insert
        total_num_inserted = total_num_inserted + num_to_insert
        for j in range(num_to_insert):
            nan_to_filling = ((raw_insole_data.shape[1] - 4) * [np.NaN])  # insert NaN value to missing row
            row_to_insert = [raw_insole_data.iloc[i, 0], 0, raw_insole_data.iloc[i, 2], time_stamp[i] + sampling_period * (j + 1),
                             *nan_to_filling]  # construct a row list. the first four columns are not measurements that needs to interpolate
            raw_insole_data.loc[i + (j + 1) / (num_to_insert + 1)] = row_to_insert  # Append row at the bottom with a given index
    print("missing insole data:", total_num_inserted / len(raw_insole_data))  # display the percentage of missing insole data
    inserted_insole_data = raw_insole_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
    return inserted_insole_data, raw_insole_data


## find lost emg data according to the value in sample counter
def findLostEmgData(raw_emg_data):
    sample_counter = raw_emg_data.iloc[:, -1].to_frame()  # this column is the sample counter (timestamp) for SyncStation
    sample_counter.columns = ["sample_count"]
    counter_diff = sample_counter["sample_count"].diff().to_frame()
    counter_diff.columns = ["count_difference"]
    wrong_timestamp = counter_diff.query('count_difference != 1 & count_difference != -65535') # exclude 65535 as it is when the counter restarts
    sample_counter = raw_emg_data.iloc[:, 72].to_frame()  # this column is the sample counter (timestamp) for sessantaquattro 1
    sample_counter.columns = ["sample_count"]
    counter_diff = sample_counter["sample_count"].diff().to_frame()
    counter_diff.columns = ["count_difference"]
    wrong_timestamp1 = counter_diff.query('count_difference != 1 & count_difference != -65535') # exclude 65535 as it is when the counter restarts
    sample_counter = raw_emg_data.iloc[:, 142].to_frame()  # the final column is the sample counter (timestamp) for sessantaquattro 2
    sample_counter.columns = ["sample_count"]
    counter_diff = sample_counter["sample_count"].diff().to_frame()
    counter_diff.columns = ["count_difference"]
    wrong_timestamp2 = counter_diff.query('count_difference != 1 & count_difference != -65535')  # exclude 65535 as it is when the counter restarts
    return wrong_timestamp, wrong_timestamp1, wrong_timestamp2  # return the index of wrong timestamps

## find missing emg data index to indicator where the emg data is lost during wifi transimission
def findMissingEngIndex(wrong_timestamp_emg1, wrong_timestamp_emg2, data_length):
    emg1_wrong_index = wrong_timestamp_emg1.index[wrong_timestamp_emg1['count_difference'] == 0].tolist()
    emg1_missing_indicator = pd.Series(np.zeros(data_length))
    emg1_missing_indicator.values[emg1_wrong_index] = 1  # set the value to 1 at the points where the data are lost
    emg2_wrong_index = wrong_timestamp_emg2.index[wrong_timestamp_emg2['count_difference'] == 0].tolist()
    emg2_missing_indicator = pd.Series(np.zeros(data_length))
    emg2_missing_indicator.values[emg2_wrong_index] = 1  # set the value to 1 at the points where the data are lost
    return emg1_missing_indicator, emg2_missing_indicator

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

## recover emg column order
def changeColumnOrder(raw_emg_data, start_index):
    num_row = raw_emg_data.shape[0]
    num_column = raw_emg_data.shape[1]
    end_index = num_row  # the last row
    column_to_reorder = raw_emg_data.loc[start_index:, :]  # keep rows of emg data where the order needs to be changed
    drop_col = column_to_reorder.pop(3)  # delete the column first
    column_to_reorder.insert(num_column - 1, num_column, drop_col)  # then add the column to the end
    columnNames = list(range(0, num_column))
    column_to_reorder.columns = columnNames
    dropped_emg_data = raw_emg_data.drop(raw_emg_data.index[range(start_index, end_index)])  # delete the rows after the index
    reordered_emg_data = pd.concat([dropped_emg_data, column_to_reorder], ignore_index=True)  # combine the two parts together
    return reordered_emg_data