## import modules
import pandas as pd
import datetime
from RawData.Utility_Functions import Align_Insole_Emg, Upsampling_Filtering, Split_Insole_Data

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
sessions = list(range(1))

modes = ['up_down']
sessions = [0]

# read aligned sensor data
now = datetime.datetime.now()
for mode in modes:
    for session in sessions:
        # read aligned data
        now = datetime.datetime.now()
        left_insole_aligned, right_insole_aligned, emg_aligned = Align_Insole_Emg.readAlignedData(subject, session, mode)
        # upsampling and filtering data
        left_insole_preprocessed, right_insole_preprocessed, emg_preprocessed = Upsampling_Filtering.preprocessSensorData(
            left_insole_aligned, right_insole_aligned, emg_aligned, filterInsole=False, notchEMG=False, quality_factor=10)

        # put multiple session data into a list
        if mode == "up_down":
            left_updown_list.append(left_insole_preprocessed)
            right_updown_list.append(right_insole_preprocessed)
            emg_updown_list.append(emg_preprocessed)
        elif mode == "down_up":
            left_downup_list.append(left_insole_preprocessed)
            right_downup_list.append(right_insole_preprocessed)
            emg_downup_list.append(emg_preprocessed)
print(datetime.datetime.now() - now)



## read split results from json files
split_parameters = Split_Insole_Data.readSplitParameters(subject)
# convert split results to dataframe
split_updown_data = [pd.DataFrame(value) for session, value in split_parameters["up_to_down"].items()]
split_downup_data = [pd.DataFrame(value) for session, value in split_parameters["down_to_up"].items()]
