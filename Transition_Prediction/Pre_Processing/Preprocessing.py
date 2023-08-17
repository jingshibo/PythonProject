"""
read split parameters to separate emg data and then calculate emg features for each gait event. the results are stored in a dict.
"""

## import modules
import pandas as pd
import datetime
from Transition_Prediction.RawData.Utility_Functions import Insole_Emg_Alignment, Insole_Data_Splition, Upsampling_Filtering
from Transition_Prediction.Pre_Processing.Utility_Functions import Data_Separation, Feature_Calculation, Feature_Storage
import concurrent.futures


## read split parameters from json files
def readSplitParameters(subject, version):
    split_parameters = Insole_Data_Splition.readSplitParameters(subject)
    # convert split results to dataframe
    split_up_down_list = {session: pd.DataFrame(value) for session, value in split_parameters[f"experiment_{version}"]["up_to_down"].items()}
    split_down_up_list = {session: pd.DataFrame(value) for session, value in split_parameters[f"experiment_{version}"]["down_to_up"].items()}
    # put split results into a dict
    split_parameters = {"up_down": split_up_down_list, "down_up": split_down_up_list}
    return split_parameters


## read, preprocess and label aligned sensor data
def labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-1024, end_position=1024, lower_limit=20,
        higher_limit=400, envelope_cutoff=10, notchEMG=False, median_filtering=True, reordering=True, envelope=False):
    now = datetime.datetime.now()
    combined_emg_labelled = {}

    mode_session = zip(modes, sessions)
    for mode, sessions_in_mode in mode_session:
        for session in sessions_in_mode:
            # read aligned data
            left_insole_aligned, right_insole_aligned, emg_aligned = Insole_Emg_Alignment.readAlignedData(subject, session, mode, version)
            # upsampling, filtering and reordering data
            left_insole_preprocessed, right_insole_preprocessed, emg_filtered, emg_reordered, emg_envelope = \
                Upsampling_Filtering.preprocessSensorData(
                left_insole_aligned, right_insole_aligned, emg_aligned, lower_limit=lower_limit, higher_limit=higher_limit,
                envelope_cutoff=envelope_cutoff, notchEMG=notchEMG, median_filtering=median_filtering)
            if envelope == True:  # if emg envelope is needed
                emg_preprocessed = emg_envelope
            elif reordering == True:  # if reordering emg is needed
                emg_preprocessed = emg_reordered
            else:  # if only filtering
                emg_preprocessed = emg_filtered
            # separate the gait event using timestamps
            gait_event_timestamp = Data_Separation.seperateGait(split_parameters[mode][f'session{session}'], start_position, end_position)
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
        futures = [executor.submit(Feature_Calculation.labelEmgFeatures, gait_event_label, gait_event_emg, window_size, increment)
            for gait_event_label, gait_event_emg in combined_emg_labelled.items()]  # parallel calculate features in multiple gait events
        for future in concurrent.futures.as_completed(futures):
            combined_emg_features.append(future.result())
    print(datetime.datetime.now() - now)
    # reorganize and label the calculated features
    emg_features = {}
    for gait_event_features in combined_emg_features:
        gait_event_label = list(gait_event_features.keys())[0]
        emg_features[gait_event_label] = gait_event_features[gait_event_label]
    return emg_features


## read sensor data and extract features with labeling
if __name__ == '__main__':

    # # basic information
    # subject = 'Shibo'
    # version = 3  # the data from which experiment version to process
    # modes = ['up_down', 'down_up']
    # up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # sessions = [up_down_session, down_up_session]

    subject = 'Number2'
    version = 0  # the data from which experiment version to process
    modes = ['up_down', 'down_up']
    up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    down_up_session = [10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
    sessions = [up_down_session, down_up_session]

    # Feature extraction
    split_parameters = readSplitParameters(subject, version)
    combined_emg_labelled = labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-900, end_position=800)
    emg_features = extractEmgFeatures(combined_emg_labelled, window_size=700, increment=40)

    # store features
    feature_set = 0  # there may be multiple sets of features to be calculated for comparison
    Feature_Storage.saveEmgFeatures(subject, emg_features, version, feature_set)

    subject = 'Number1'
    version = 0  # the data from which experiment version to process
    modes = ['up_down', 'down_up']
    up_down_session = [0, 1, 4, 5, 6, 7, 8, 9, 10]
    down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    sessions = [up_down_session, down_up_session]

    # Feature extraction
    split_parameters = readSplitParameters(subject, version)
    combined_emg_labelled = labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-900, end_position=800)
    emg_features = extractEmgFeatures(combined_emg_labelled, window_size=700, increment=40)

    # store features
    feature_set = 0  # there may be multiple sets of features to be calculated for comparison
    Feature_Storage.saveEmgFeatures(subject, emg_features, version, feature_set)


## balance emg data
# banlanced_emg = {}
# for key, value in combined_emg_labelled.items():
#     base_number = len(combined_emg_labelled["emg_SSLW"])
#     to_devide = int(len(value) / base_number)
#     for i in range(to_devide):
#         banlanced_emg[f"emg_{key}_{i}"] = value[i*base_number:(i+1)*base_number]
# emg_features = extractEmgFeatures(banlanced_emg, window_size=512, increment=32)
