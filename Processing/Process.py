## import modules
import pandas as pd
import numpy as np
import datetime
from RawData.Utility_Functions import Insole_Emg_Alignment, Upsampling_Filtering, Insole_Data_Splition
from Processing.Utility_Functions import Data_Separation, Electrode_Reordering, Feature_Extraction
import concurrent.futures

## feature labelling\

if __name__ == '__main__':


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
            gait_event = Data_Separation.seperateGait(split_data[mode][session], window_size=512)
            # use the gait event timestamp to label emg data
            emg_labelled= Data_Separation.seperateEmgdata(emg_reordered, gait_event)
            # combine emg data from all sessions together
            for gait_event_label, gait_event_emg in emg_labelled.items():
                if gait_event_label in combined_emg_labelled:
                    combined_emg_labelled[gait_event_label].extend(gait_event_emg)
                else:
                    combined_emg_labelled[gait_event_label] = gait_event_emg

    print(datetime.datetime.now() - now)


    ## calculate and label emg features
    window_size = 512
    increment = 32

    now = datetime.datetime.now()
    combined_emg_features = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(Feature_Extraction.labelEmgFeatures, key, gait_event) for key, gait_event in combined_emg_labelled.items()]
        for f in concurrent.futures.as_completed(futures):
            combined_emg_features.append(f.result())

    emg_features = {}
    for gait_event_features in combined_emg_features:
        key = list(gait_event_features.keys())[0]
        emg_features[key] = gait_event_features[key]

    print(datetime.datetime.now() - now)


## balance emg data
# banlanced_emg = {}
# for key, value in combined_emg_labelled.items():
#     base_number = len(combined_emg_labelled["emg_SSLW"])
#     to_devide = int(len(value) / base_number)
#     for i in range(to_devide):
#         banlanced_emg[f"emg_{key}_{i}"] = value[i*base_number:(i+1)*base_number]