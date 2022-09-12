"""
read split parameters to separate emg data and then calculate emg features for each gait event. the results are stored in a dict.
"""

## import modules
import pandas as pd
import datetime
from RawData.Utility_Functions import Insole_Emg_Alignment, Upsampling_Filtering, Insole_Data_Splition
from Processing.Utility_Functions import Data_Separation, Feature_Calculation, Feature_Storage
import concurrent.futures


## read split parameters from json files
def readSplitData(subject):
    split_parameters = Insole_Data_Splition.readSplitParameters(subject)
    # convert split results to dataframe
    split_up_down_list = [pd.DataFrame(value) for session, value in split_parameters["up_to_down"].items()]
    split_down_up_list = [pd.DataFrame(value) for session, value in split_parameters["down_to_up"].items()]
    # put split results into a dict
    split_data = {"up_down": split_up_down_list, "down_up": split_down_up_list}
    return split_data

## read, preprocess and label aligned sensor data
def labelSensorData(subject, modes, sessions, split_data):
    now = datetime.datetime.now()
    combined_emg_labelled = {}
    for mode in modes:
        for session in sessions:
            # read aligned data
            left_insole_aligned, right_insole_aligned, emg_aligned = Insole_Emg_Alignment.readAlignedData(subject, session, mode)
            # upsampling, filtering and reordering data
            left_insole_preprocessed, right_insole_preprocessed, emg_preprocessed = Upsampling_Filtering.preprocessSensorData(
                left_insole_aligned, right_insole_aligned, emg_aligned, filterInsole=False, notchEMG=False, quality_factor=10)
            # separate the gait event using timestamps
            gait_event_timestamp = Data_Separation.seperateGait(split_data[mode][session], window_size=512)
            # use the gait event timestamps to label emg data
            emg_labelled = Data_Separation.seperateEmgdata(emg_preprocessed, gait_event_timestamp)
            # combine the emg data from all sessions of the same gait event into the same key of a dict
            for gait_event_label, gait_event_emg in emg_labelled.items():
                if gait_event_label in combined_emg_labelled:  # check if there is already the key in the dict
                    combined_emg_labelled[gait_event_label].extend(gait_event_emg)
                else:
                    combined_emg_labelled[gait_event_label] = gait_event_emg
    print(datetime.datetime.now() - now)
    return combined_emg_labelled


## calculate and label emg features
def extractEmgFeatures(combined_emg_labelled, window_size=512, increment=32):
    now = datetime.datetime.now()
    combined_emg_features = []
    # calculate emg features using multiprocessing. there is balance of CPU number, not more is better as numpy auto parallel to some extent
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(Feature_Calculation.labelEmgFeatures, gait_event_label, gait_event_emg, window_size, increment) for
            gait_event_label, gait_event_emg in combined_emg_labelled.items()]  # parallel calculate features in multiple gait events
        for future in concurrent.futures.as_completed(futures):
            combined_emg_features.append(future.result())
    print(datetime.datetime.now() - now)
    # reorganize the calculated features with labelling
    emg_features = {}
    for gait_event_features in combined_emg_features:
        gait_event_label = list(gait_event_features.keys())[0]
        emg_features[gait_event_label] = gait_event_features[gait_event_label]
    return emg_features

## read sensor data and extract features with labeling
if __name__ == '__main__':
    # basic information

    # subject = 'Shibo'
    # modes = ['up_down', 'down_up']
    # sessions = list(range(10))
    subject = 'Shibo'
    modes = ['up_down']
    sessions = [0]

    # Feature extraction
    split_data = readSplitData(subject)
    combined_emg_labelled = labelSensorData(subject, modes, sessions, split_data)
    emg_features = extractEmgFeatures(combined_emg_labelled, window_size=512, increment=32)
    Feature_Storage.saveEmgFeatures(subject, emg_features)

## balance emg data
# banlanced_emg = {}
# for key, value in combined_emg_labelled.items():
#     base_number = len(combined_emg_labelled["emg_SSLW"])
#     to_devide = int(len(value) / base_number)
#     for i in range(to_devide):
#         banlanced_emg[f"emg_{key}_{i}"] = value[i*base_number:(i+1)*base_number]
# emg_features = extractEmgFeatures(banlanced_emg, window_size=512, increment=32)
