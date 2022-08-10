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
import csv

## initialization
data_dir = 'D:\Data\Insole_Emg'
data_file_name = 'subject0_20220805_175751'
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


## read and impute sensor data
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


## comcat two insole data into one dataframe
combined_insole_data = Align_Two_Insoles.cancatInsole(recovered_left_data, recovered_right_data)  # to view combined_insole_data data


## align the begin of sensor data
left_start_timestamp = 100
right_start_timestamp = 350
combine_cropped_begin, left_cropped_begin, right_cropped_begin = Align_Two_Insoles.alignInsoleBegin(  # to view combine_cropped_begin data
                        left_start_timestamp, right_start_timestamp, recovered_left_data,  recovered_right_data)


## align the end of sensor data
left_end_timestamp = 304550
right_end_timestamp = 304800
left_insole_aligned, right_insole_aligned = Align_Two_Insoles.alignInsoleEnd(left_end_timestamp,
                                right_end_timestamp, left_cropped_begin, right_cropped_begin)

## upsampling and filtering aligned insole data
left_insole_upsampled, right_insole_upsampled = Align_Two_Insoles.upsampleInsole(left_insole_aligned, right_insole_aligned)
# left_insole_filtered, right_insole_filtered = Align_Two_Insoles.filterInsole(left_insole_upsampled, right_insole_upsampled)

## check the insole alignment results
start_index = 0
end_index = 40000
Align_Two_Insoles.plotAlignedInsole(left_insole_upsampled, right_insole_upsampled, start_index, end_index)



## align insole and EMG
emg_aligned = Align_Insole_Emg.alignInsoleEmg(raw_emg_data, left_insole_aligned, right_insole_aligned)
emg_filtered = Align_Insole_Emg.filterEmg(emg_aligned, notch=False, quality_factor=30)


## check the insole and emg alignment results
start_index = 00000
end_index = 610000
Align_Insole_Emg.plotInsoleEmg(emg_filtered, left_insole_upsampled, right_insole_upsampled, start_index, end_index)


## save the alignment parameters
# save file path
subject = 0
data_dir = 'D:\Data\Insole_Emg'
alignment_save_file = f'subject{subject}.csv'
alignment_save_path = os.path.join(data_dir, alignment_save_file)

# alignment parameters to save
columns = ['alignment_save_date', 'data_file_name', 'left_start_timestamp', 'right_start_timestamp',
           'left_end_timestamp', 'right_end_timestamp']
save_parameters = [datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), data_file_name, left_start_timestamp,
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





## plot sensor data
start_index = 00000
end_index = 610000

left_total_force = left_insole_upsampled[195]
right_total_force = right_insole_upsampled[195]
emg_data = emg_filtered.sum(1)

# plot
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
axes[0].plot(range(len(left_total_force.iloc[start_index:end_index])), left_total_force.iloc[start_index:end_index],
             label="Left Insole Force")
axes[0].plot(range(len(right_total_force.iloc[start_index:end_index])),
             right_total_force.iloc[start_index:end_index], label="Right Insole Force")
axes[1].plot(range(len(emg_data.iloc[start_index:end_index])), emg_data.iloc[start_index:end_index],
             label="Emg Signal")

axes[0].set(title="Insole Force", ylabel="force(kg)")
axes[1].set(title="Emg Signal", xlabel="Sample Number", ylabel="Emg Value")

axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels

axes[0].legend(loc="upper right")
axes[1].legend(loc="upper right")


## test
date1 = datetime.datetime.strptime(combine_cropped_begin.iat[0,2], "%Y-%m-%d_%H:%M:%S.%f")
date2 = datetime.datetime.strptime(combine_cropped_begin.iat[1014,2], "%Y-%m-%d_%H:%M:%S.%f")
delta = (date2 - date1).total_seconds() * 1000
number = delta / 25
print(number)