##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PreProcessing.Recover_Insole import upsampleInsoleData

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

## upsamling insole data
def upsampleInsole(left_insole_aligned, right_insole_aligned):
    # upsampling insole data
    upsampled_left_insole = upsampleInsoleData(left_insole_aligned).reset_index()
    upsampled_right_insole = upsampleInsoleData(right_insole_aligned).reset_index()
    return upsampled_left_insole, upsampled_right_insole

## filtering insole data
def filterInsole(upsampled_left_insole, upsampled_right_insole):
    # filtering insole signal after upsampling
    sos  = signal.butter(4, [20], fs = 2000, btype = "lowpass", output='sos')
    left_insole_filtered = signal.sosfiltfilt(sos, upsampled_left_insole.iloc[:,1:193], axis=0)
    right_insole_filtered = signal.sosfiltfilt(sos, upsampled_right_insole.iloc[:,1:193], axis=0)
    return pd.DataFrame(left_insole_filtered), pd.DataFrame(right_insole_filtered)

## plot insole data
def plotAlignedInsole(left_insole_dataframe, right_insole_dataframe, start_index, end_index):
    left_total_force = left_insole_dataframe[195]  # extract total force column
    right_total_force = right_insole_dataframe[195]

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
