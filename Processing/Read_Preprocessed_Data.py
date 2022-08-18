## import modules
import pandas as pd
import datetime
from RawData.Utility_Functions import Align_Insole_Emg, Upsampling_Filtering, Split_Insole_Data
from Processing.Utility_Functions import Seperate_Data


## basic information
subject = 'Shibo'
modes = ['up_down', 'down_up']
sessions = list(range(10))

# modes = ['up_down']
# sessions = [0]


## read split data from json files
split_parameters = Split_Insole_Data.readSplitParameters(subject)
# convert split results to dataframe
split_up_down_list = [pd.DataFrame(value) for session, value in split_parameters["up_to_down"].items()]
split_down_up_list = [pd.DataFrame(value) for session, value in split_parameters["down_to_up"].items()]
# put split results into a dict
split_data = {"up_down": split_up_down_list, "down_up": split_down_up_list}


## read and combine aligned sensor data
now = datetime.datetime.now()
combined_emg_data = {}
for mode in modes:
    for session in sessions:
        # read aligned data
        left_insole_aligned, right_insole_aligned, emg_aligned = Align_Insole_Emg.readAlignedData(subject, session, mode)
        # upsampling and filtering data
        left_insole_preprocessed, right_insole_preprocessed, emg_preprocessed = Upsampling_Filtering.preprocessSensorData(
            left_insole_aligned, right_insole_aligned, emg_aligned, filterInsole=False, notchEMG=False, quality_factor=10)
        # devide the gait into subcycles
        gait_event_devision = Seperate_Data.seperateGait(split_data[mode][session], window_size=512)
        # divide the emg data into subcycles
        separated_emg_data = Seperate_Data.seperateEmgdata(emg_preprocessed, gait_event_devision)
        # combine emg data from all sessions together
        for key, value in separated_emg_data.items():
            if key in combined_emg_data:
                combined_emg_data[key].extend(value)
            else:
                combined_emg_data[key] = value

print(datetime.datetime.now() - now)


