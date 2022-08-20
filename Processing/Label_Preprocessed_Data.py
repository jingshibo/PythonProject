## import modules
import pandas as pd
import numpy as np
import datetime
from RawData.Utility_Functions import Insole_Emg_Alignment, Upsampling_Filtering, Insole_Data_Splition
from Processing.Utility_Functions import Data_Separation, Electrode_Reordering, Feature_Extraction


## basic information
subject = 'Shibo'
modes = ['up_down', 'down_up']
sessions = list(range(10))

modes = ['up_down']
sessions = [0]


## read split data from json files
split_parameters = Insole_Data_Splition.readSplitParameters(subject)
# convert split results to dataframe
split_up_down_list = [pd.DataFrame(value) for session, value in split_parameters["up_to_down"].items()]
split_down_up_list = [pd.DataFrame(value) for session, value in split_parameters["down_to_up"].items()]
# put split results into a dict
split_data = {"up_down": split_up_down_list, "down_up": split_down_up_list}


## read and label aligned sensor data
now = datetime.datetime.now()

combined_emg_labelled = {}
for mode in modes:
    for session in sessions:
        # read aligned data
        left_insole_aligned, right_insole_aligned, emg_aligned = Insole_Emg_Alignment.readAlignedData(subject, session, mode)
        # upsampling and filtering data
        left_insole_preprocessed, right_insole_preprocessed, emg_preprocessed = Upsampling_Filtering.preprocessSensorData(
            left_insole_aligned, right_insole_aligned, emg_aligned, filterInsole=False, notchEMG=False, quality_factor=10)
        # recover mission emg channels

        # adjust electrode order to match the physical EMG grid
        emg_reordered = Electrode_Reordering.reorderElectrodes(emg_preprocessed)
        # separate the gait event with labelling
        gait_event_label = Data_Separation.seperateGait(split_data[mode][session], window_size=512)
        # use the gait event timestamp to label emg data
        labelled_emg_data = Data_Separation.seperateEmgdata(emg_reordered, gait_event_label)
        # combine emg data from all sessions together
        for key, value in labelled_emg_data.items():
            if key in combined_emg_labelled:
                combined_emg_labelled[key].extend(value)
            else:
                combined_emg_labelled[key] = value

print(datetime.datetime.now() - now)


## feature labelling
window_size = 512
increment = 32
now = datetime.datetime.now()

emg_features_labelled = {}
for key, gait_event in combined_emg_labelled.items():
    emg_window_features = []

    event =  datetime.datetime.now()
    for emg_session_data in gait_event:

        session = datetime.datetime.now()
        for i in range(0, len(emg_session_data) - window_size + 1, increment):
            emg_window_data = emg_session_data[i:i + window_size, :]
            emg_window_features.append(Feature_Extraction.calcuEmgFeatures(emg_window_data))

        print("session:", datetime.datetime.now() - session)

    emg_features_labelled[f"{key}_features"] = np.array(emg_window_features)
    print("event:", datetime.datetime.now() - event)

print(datetime.datetime.now() - now)