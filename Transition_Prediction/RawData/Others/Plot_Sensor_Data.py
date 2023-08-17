## import modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from Transition_Prediction.RawData.Utility_Functions.Insole_Emg_Recovery import insertInsoleMissingRow

# initialization
data_dir = 'D:\Data\Insole_Emg'
data_file_name = 'subject0_20220803_165712'
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


## emg
now = datetime.datetime.now()
raw_emg_data = pd.read_csv(emg_path, sep=',', header=None, dtype='int16',
                           converters={0: str, 1: str, 2: str})  # change data type for faster reading
print(datetime.datetime.now() - now)


## left insole
raw_left_data = pd.read_csv(left_insole_path, sep=',', header=None)
recovered_left_data = insertInsoleMissingRow(raw_left_data, insole_sampling_period)  # add missing rows with NaN values

left_total_force = recovered_left_data[195]  # extract total force column
big_toes = recovered_left_data[191]
smaller_toes = recovered_left_data[192]
inner_heel = recovered_left_data[185]
outer_heel = recovered_left_data[184]

front = recovered_left_data[189]
front_2 = recovered_left_data[190]
mid = recovered_left_data[186]
arch = recovered_left_data[187]

plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
left_total_force.iloc[0:1000].plot()  # no need to specify for first axis
ax1.set_ylabel('force(kg)')
ax1.set_title('Left Insole Force')
ax2 = plt.subplot(2, 1, 2)
left_total_force.iloc[0:1000].plot(ax=plt.gca())
ax2.set_ylabel('force(kg)')
ax2.set_title('Left Insole Force')

## right insole
raw_right_data = pd.read_csv(right_insole_path, sep=',', header=None)
recovered_right_data = insertInsoleMissingRow(raw_right_data, insole_sampling_period)  # add missing rows with NaN values

right_total_force = recovered_right_data[195]  # extract total force column

left_total_force = recovered_right_data[195]  # extract total force column
big_toes = recovered_right_data[191]
smaller_toes = recovered_right_data[192]
inner_heel = recovered_right_data[185]
outer_heel = recovered_right_data[184]

front = recovered_right_data[189]
front_2 = recovered_right_data[190]
mid = recovered_right_data[186]
arch = recovered_right_data[187]

plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
right_total_force.iloc[0:1100].plot()  # no need to specify for first axis
ax1.set_ylabel('force(kg)')
ax1.set_title('Left Insole Force')
ax2 = plt.subplot(2, 1, 2)
right_total_force.iloc[0:1000].plot(ax=plt.gca())
ax2.set_ylabel('force(kg)')
ax2.set_title('Left Insole Force')

