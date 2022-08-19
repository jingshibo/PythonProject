## import modules
import pandas as pd
import numpy as np
import datetime
from RawData.Utility_Functions import Align_Insole_Emg, Upsampling_Filtering, Split_Insole_Data
from Processing.Utility_Functions import Seperate_Data, Reorder_Electrodes, Calculate_Features


## basic information
subject = 'Shibo'
modes = ['up_down', 'down_up']
sessions = list(range(10))

modes = ['up_down']
sessions = [0]


## read split data from json files
split_parameters = Split_Insole_Data.readSplitParameters(subject)
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
        left_insole_aligned, right_insole_aligned, emg_aligned = Align_Insole_Emg.readAlignedData(subject, session, mode)
        # upsampling and filtering data
        left_insole_preprocessed, right_insole_preprocessed, emg_preprocessed = Upsampling_Filtering.preprocessSensorData(
            left_insole_aligned, right_insole_aligned, emg_aligned, filterInsole=False, notchEMG=False, quality_factor=10)
        # recover mission emg channels

        # adjust electrode order to match the physical EMG grid
        emg_reordered = Reorder_Electrodes.reorderElectrodes(emg_preprocessed)
        # separate the gait event with labelling
        gait_event_label = Seperate_Data.seperateGait(split_data[mode][session], window_size=512)
        # use the gait event timestamp to label emg data
        labelled_emg_data = Seperate_Data.seperateEmgdata(emg_reordered, gait_event_label)
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
    for emg_session_data in gait_event:
        for i in range(0, len(emg_session_data) - window_size + 1, increment):
            emg_window_data = emg_session_data[i:i + window_size, :]
            emg_window_features.append(Calculate_Features.calcuEmgFeatures(emg_window_data))
    emg_features_labelled[f"{key}_features"] = np.array(emg_window_features)

print(datetime.datetime.now() - now)

## feature calculation


##
from math import sqrt
now = datetime.datetime.now()
a = [sqrt(i ** 2) for i in range(100000000)]
print(datetime.datetime.now() - now)

##
from math import sqrt
from joblib import Parallel, delayed
now = datetime.datetime.now()
Parallel(n_jobs=8)(delayed(sqrt)(i ** 2) for i in range(100000000))
print(datetime.datetime.now() - now)