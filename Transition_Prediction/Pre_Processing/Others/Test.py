## import modules
import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.ar_model import AutoReg
from Transition_Prediction.RawData.Utility_Functions import Insole_Emg_Alignment, Insole_Data_Splition, Upsampling_Filtering
from Transition_Prediction.Pre_Processing.Utility_Functions import Data_Separation, Data_Reshaping
import concurrent.futures
import multiprocessing
import functools

##
def ar(emg_window_data, num_coeff, i):
    ar_model = AutoReg(emg_window_data[:, i], lags=num_coeff).fit()
    ar_para = ar_model.params
    return ar_para[1], ar_para[2], ar_para[3]

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
            emg_reordered = Data_Reshaping.reorderElectrodes(emg_preprocessed)
            # separate the gait event with labelling
            gait_event_key = Data_Separation.seperateGait(split_data[mode][session], start_position=512)
            # use the gait event timestamp to label emg data
            labelled_emg_data = Data_Separation.seperateEmgdata(emg_reordered, gait_event_key)
            # combine emg data from all sessions together
            for key, value in labelled_emg_data.items():
                if key in combined_emg_labelled:
                    combined_emg_labelled[key].extend(value)
                else:
                    combined_emg_labelled[key] = value
    print(datetime.datetime.now() - now)

    ##
    window_size = 512
    increment = 32
    emg_features = []

    now = datetime.datetime.now()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        emg_features_labelled = {}
        for gait_event_key, gait_event_emg in combined_emg_labelled.items():
            emg_feature_labelled = {}
            emg_window_features = []
            event = datetime.datetime.now()
            for session_emg in gait_event_emg:
                session = datetime.datetime.now()
                for i in range(0, len(session_emg) - window_size + 1, increment):
                    emg_window_data = session_emg[i:i + window_size, :]

                    sample_number = emg_window_data.shape[0]
                    channel_number = emg_window_data.shape[1]
                    # mean absolute value
                    MAV = np.mean(np.abs(emg_window_data), axis=0)
                    # root mean square
                    RMS = np.sqrt(np.sum(emg_window_data * emg_window_data, axis=0) / sample_number)
                    # waveform length
                    WL = np.sum(np.abs(np.diff(emg_window_data, axis=0)), axis=0)
                    # slope sign changes
                    SSC = np.zeros((sample_number - 4, channel_number))  # preallocate memory
                    for i in np.arange(0, sample_number - 4):
                        d1 = emg_window_data[i, :] - emg_window_data[i + 1, :]  # difference of sample i and sample i+1
                        d2 = emg_window_data[i + 1, :] - emg_window_data[i + 2, :]  # difference of sample i+1 and sample i+2
                        product_12 = d1 * d2
                        for j in np.arange(0, channel_number):
                            if product_12[j] < 0:  # check if point i+2 starts to turn [must not include equal'=' here]
                                # if this is true, we need to check if the next point [i+3] is also a turning point
                                d3 = emg_window_data[i + 2, j] - emg_window_data[i + 3, j]
                                product_23 = d2[j] * d3
                                if product_23 < 0:  # if this is true, it means point [i+3] is a turning point, and point [i+2] is noisy
                                    # then we need to see if the next next point [i+4] is also a turning point
                                    d4 = emg_window_data[i + 3, j] - emg_window_data[i + 4, j]
                                    product_34 = d3 * d4
                                    if product_34 < 0:  # if this is true, it means point [i+4] is also a turning point
                                        # if [i+4] is also a turning point, we think point [i+2] is noisy, but it is still a turning point
                                        SSC[i, j] = 1
                                        # because turning point [i+2] is a noisy value, we need to reset it value to be more correct
                                        if d1[j] > 0:  # the previous slope is decrease, and turning point [i+2] is the bottom
                                            emg_window_data[i + 2, j] = np.minimum(emg_window_data[i + 1, j], emg_window_data[i + 3, j])
                                        else:  # the previous slope is increase, and turning point [i+2] is the peak
                                            emg_window_data[i + 2, j] = np.maximum(emg_window_data[i + 1, j], emg_window_data[i + 3, j])
                                    else:  # if [i+4] is not a turning point, we think point [i+2] is noisy, and it is not a turning point
                                        SSC[i, j] = 0
                                        # because non-turning point [i+2] is a noisy value, we need to reset it value to be more correct
                                        if d1[j] > 0:
                                            emg_window_data[i + 2, j] = np.maximum(emg_window_data[i + 1, j], emg_window_data[i + 3, j])
                                        else:
                                            emg_window_data[i + 2, j] = np.minimum(emg_window_data[i + 1, j], emg_window_data[i + 3, j])
                                else:  # if product_23[j] >= 0, it means the next point [i+3] is not a turning point
                                    SSC[i, j] = 1  # point [i+2] is a stable turing point, and thus slope sign changes
                            else:  # if product_12[j] >= 0, it means point [i+2] is not a turning point
                                SSC[i, j] = 0  # slope sign does not change
                    SSCn = np.sum(SSC, axis=0)
                    # zero crossing number
                    ZC = np.zeros((sample_number - 2, channel_number))  # preallocate memory
                    for i in np.arange(0, sample_number - 2):
                        product_1 = emg_window_data[i, :] * emg_window_data[i + 1, :]  # product of sample i and sample i+1
                        for j in np.arange(0, channel_number):
                            if product_1[
                                j] <= 0:  # if sign change is detected [must include equal'=' here, in case there are two consective zero
                                # values]
                                product_2 = emg_window_data[i + 1, j] * emg_window_data[i + 2, j]  # product of sample i+1 and sample i+2
                                if product_2 >= 0:
                                    ZC[i, j] = 1  # it is a stable change, record this change
                                else:
                                    ZC[i, j] = 0  # it is a noisy change, ignore this change
                                    emg_window_data[i + 1, j] = - emg_window_data[i + 1, j]  # take the opposite to reset the noise value
                            else:
                                ZC[i, j] = 0  # no sign change recorded
                    ZCn = np.sum(ZC, axis=0)
                    # autoregression coeffecients
                    AR=[]
                    AR_1 = np.zeros(channel_number)
                    AR_2 = np.zeros(channel_number)
                    AR_3 = np.zeros(channel_number)
                    # AR_4 = np.zeros(channel_number)
                    # AR_5 = np.zeros(channel_number)
                    # AR_6 = np.zeros(channel_number)
                    num_coeff = 3
                    results = executor.map(functools.partial(ar, emg_window_data, num_coeff), np.arange(0, channel_number))
                    for result in results:
                        AR.append(result)

                    emg_window_features.append((np.concatenate([MAV, RMS, WL, SSCn, ZCn]), AR))
                print("session:", multiprocessing.current_process().name, datetime.datetime.now() - session)
            emg_feature_labelled[f"{gait_event_key}_features"] = emg_window_features
            print(f"event:{gait_event_key}", multiprocessing.current_process().name, datetime.datetime.now() - event)
            emg_features_labelled.update(emg_feature_labelled)

    print(datetime.datetime.now() - now)

