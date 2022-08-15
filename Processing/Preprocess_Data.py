## import modules
import os
import pandas as pd
import datetime
from PreProcessing.Utility import Split_Insole_Data
from Processing.Utility import Read_Aligned_Data


## storing multiple session preprocessed data
left_updown_list = []
right_updown_list = []
emg_updown_list = []
left_downup_list = []
right_downup_list = []
emg_downup_list = []


## read aligned and filtered data
subject = 'Shibo'
modes = ['up_down', 'down_up']
sessions = range(2)

modes = ['up_down']
sessions = [0]

now = datetime.datetime.now()
# extract sensor data
for mode in modes:
    for session in sessions:
        try:
            preprocessed_data_list = Read_Aligned_Data.readAlignedData(subject, session, mode) # return a tuple including EMG and insole data
        except Exception as e:
            print(e)
        # put multiple session data into a list
        if mode == "up_down":
            left_updown_list.append(preprocessed_data_list[0])
            right_updown_list.append(preprocessed_data_list[1])
            emg_updown_list.append(preprocessed_data_list[2])
        elif mode == "down_up":
            left_downup_list.append(preprocessed_data_list[0])
            right_downup_list.append(preprocessed_data_list[1])
            emg_downup_list.append(preprocessed_data_list[2])
print(datetime.datetime.now() - now)


## if have not splited data yet, plot sensor data to split gait cycles
start_index = 00000
end_index = 600000
left_force_baseline = 4.5
right_force_baseline = 4.5

session = 0
left_insole_upsampled = left_updown_list[session]
right_insole_upsampled = right_updown_list[session]
emg_filtered = emg_updown_list[session]

Split_Insole_Data.plotSplitLine(left_insole_upsampled, right_insole_upsampled, emg_filtered, start_index, end_index,
    left_force_baseline, right_force_baseline)


## read split results from json files
split_results = Split_Insole_Data.readSplitData(subject)
# convert split results to dataframe
split_updown_data = [pd.DataFrame(value) for session, value in split_results["up_to_down"].items()]
split_downup_data = [pd.DataFrame(value) for session, value in split_results["down_to_up"].items()]
