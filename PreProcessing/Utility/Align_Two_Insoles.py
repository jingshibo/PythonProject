##
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from PreProcessing.Utility import Recover_Insole
import os
import datetime
import csv

## comcat two insole data into one dataframe
def cancatInsole(recovered_left_data, recovered_right_data):
    combined_insole_data = pd.concat([recovered_left_data, recovered_right_data], ignore_index=True).sort_values([0, 3]) # sort according to column 0, then column 3
    combined_insole_data = combined_insole_data.reset_index(drop=False)
    return combined_insole_data

## align the begin of sensor data
def alignInsoleBegin(left_start_timestamp, right_start_timestamp, recovered_left_data, recovered_right_data):
    # observe to get the beginning index
    left_start_index = recovered_left_data.index[recovered_left_data[3] == left_start_timestamp].tolist()
    right_start_index = recovered_right_data.index[recovered_right_data[3] == right_start_timestamp].tolist()

    # only keep insole data after the start index
    left_cropped_begin = recovered_left_data.iloc[left_start_index[0]:, :].reset_index(drop=True)
    right_cropped_begin = recovered_right_data.iloc[right_start_index[0]:, :].reset_index(drop=True)

    # add the sample number as a column for easier comparison and concat left and right insoles
    left_cropped_begin.insert(loc=0, column='number', value=range(len(left_cropped_begin)))
    right_cropped_begin.insert(loc=0, column='number', value=range(len(right_cropped_begin)))
    combine_cropped_begin = pd.concat([left_cropped_begin, right_cropped_begin], ignore_index=True).sort_values([0, 3]) # sort according to column 0, then column 3
    combine_cropped_begin = combine_cropped_begin.reset_index(drop=False)
    return combine_cropped_begin, left_cropped_begin, right_cropped_begin

## align the end of sensor data
def alignInsoleEnd(left_end_timestamp, right_end_timestamp, left_cropped_begin, right_cropped_begin):
    left_end_index = left_cropped_begin.index[left_cropped_begin[3] == left_end_timestamp].tolist()
    right_end_index = right_cropped_begin.index[right_cropped_begin[3] == right_end_timestamp].tolist()

    # only keep data before the ending index
    left_insole_aligned = left_cropped_begin.iloc[:left_end_index[0]+1, 1:].reset_index(drop=True) # remove the first column indicating data number
    right_insole_aligned = right_cropped_begin.iloc[:right_end_index[0]+1, 1:].reset_index(drop=True) # remove the first column indicating data number
    return left_insole_aligned, right_insole_aligned

## upsamling insole data to match emg
def upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned):
    # upsample insole data to every 5ms (abandoned)
    # upsampled_left_insole = upsampleInsoleData(left_insole_aligned).reset_index()
    # upsampled_right_insole = upsampleInsoleData(right_insole_aligned).reset_index()

    # check if there are emg data lost
    emg_timestamp = pd.to_datetime(emg_aligned[0])
    expected_number = (emg_timestamp.iloc[-1] - emg_timestamp.iloc[0]).total_seconds() * 1000 * 2
    if abs(expected_number - len(emg_timestamp)) >= 50:
        raise Exception("EMG Data Lost")  # then you need to recover the lost data in EMG
    else:
        # upsample insole data to 2000Hz
        upsampled_left_insole = Recover_Insole.upsampleInsoleEqualToEMG(left_insole_aligned, emg_aligned)
        upsampled_right_insole = Recover_Insole.upsampleInsoleEqualToEMG(right_insole_aligned, emg_aligned)
        return upsampled_left_insole, upsampled_right_insole

## filtering insole data
def filterInsole(upsampled_left_insole, upsampled_right_insole):
    # filtering insole signal after upsampling
    sos  = signal.butter(4, [20], fs=2000, btype = "lowpass", output='sos')
    left_insole_filtered = signal.sosfiltfilt(sos, upsampled_left_insole.iloc[:,1:193], axis=0)
    right_insole_filtered = signal.sosfiltfilt(sos, upsampled_right_insole.iloc[:,1:193], axis=0)
    left_insole_filtered = pd.DataFrame(left_insole_filtered)
    left_insole_filtered.insert(0, "timestamp", upsampled_left_insole.iloc[:, 0]) # add timestamp column
    right_insole_filtered = pd.DataFrame(right_insole_filtered)
    right_insole_filtered.insert(0, "timestamp", upsampled_right_insole.iloc[:, 0]) # add timestamp column
    return left_insole_filtered, right_insole_filtered

## plot insole data
def plotAlignedInsole(left_insole_aligned, right_insole_aligned, start_index, end_index):
    left_total_force = left_insole_aligned.iloc[:, 195]  # extract total force column from aligned insole data
    right_total_force = right_insole_aligned.iloc[:, 195]

    # plot
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].plot(range(len(left_total_force.iloc[start_index:end_index])), left_total_force.iloc[start_index:end_index],
                 label="Left Insole Force")
    axes[1].plot(range(len(right_total_force.iloc[start_index:end_index])),
                 right_total_force.iloc[start_index:end_index], label="Right Insole Force")

    axes[0].set(title="Left Insole Force", ylabel="force(kg)")
    axes[1].set(title="right Insole Force", ylabel="force(kg)")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")

## save the alignment data
def saveAlignData(subject, data_file_name, left_start_timestamp, right_start_timestamp, left_end_timestamp, right_end_timestamp):
    # save file path
    data_dir = 'D:\Data\Insole_Emg'
    alignment_save_file = f'subject_{subject}.csv'
    alignment_save_path = os.path.join(data_dir, alignment_save_file)

    # alignment parameters to save
    columns = ['data_file_name', 'alignment_save_date', 'left_start_timestamp', 'right_start_timestamp',
        'left_end_timestamp', 'right_end_timestamp']
    save_parameters = [data_file_name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), left_start_timestamp,
        right_start_timestamp, left_end_timestamp, right_end_timestamp]

    with open(alignment_save_path, 'a+') as file:
        if os.stat(alignment_save_path).st_size == 0:  # if the file is new created
            print("Created file.")
            write = csv.writer(file)
            write.writerow(columns)  # write the column fields
            write.writerow(save_parameters)
        else:
            write = csv.writer(file)
            write.writerow(save_parameters)

## read the alignment data
def readAlignData(subject, data_file_name):
    data_dir = 'D:\Data\Insole_Emg'
    alignment_file_name = f'subject_{subject}.csv'
    alignment_file_path = os.path.join(data_dir, alignment_file_name)
    alignment_data = pd.read_csv(alignment_file_path, sep=',')  # header exists

    file_parameter = alignment_data.query('data_file_name == @data_file_name') # use @ to cite variable values
    align_parameter = file_parameter.iloc[[-1]]  # extract the last row (apply the newest parameters)

    left_start_timestamp = align_parameter['left_start_timestamp'].iloc[0]
    right_start_timestamp = align_parameter['right_start_timestamp'].iloc[0]
    left_end_timestamp = align_parameter['left_end_timestamp'].iloc[0]
    right_end_timestamp = align_parameter['right_end_timestamp'].iloc[0]
    return left_start_timestamp, right_start_timestamp, left_end_timestamp, right_end_timestamp