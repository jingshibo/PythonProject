##
import numpy as np
import datetime
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import skew, kurtosis
import multiprocessing


## calculate emg features within a window
def calcuEmgFeatures(emg_window_data):
    sample_number = emg_window_data.shape[0]
    channel_number = emg_window_data.shape[1]

    # mean absolute value
    MAV = np.mean(np.abs(emg_window_data), axis=0)
    # root mean square
    RMS = np.sqrt(np.sum(emg_window_data * emg_window_data, axis=0) / sample_number)
    # waveform length
    WL = np.sum(np.abs(np.diff(emg_window_data, axis=0)), axis=0)

    # # skewness
    # SK = skew(emg_window_data, axis=0)
    # # kurtosis
    # KU = kurtosis(emg_window_data, axis=0)

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
            if product_1[j] <= 0:  # if sign change is detected [must include equal'=' here, in case there are two consective zero values]
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
    AR_1 = np.zeros(channel_number)
    AR_2 = np.zeros(channel_number)
    AR_3 = np.zeros(channel_number)
    num_coeff = 3
    for i in np.arange(0, channel_number):
        ar_model = AutoReg(emg_window_data[:, i], lags=num_coeff).fit()
        ar_para = ar_model.params  # ar_para[0] is the constant
        AR_1[i] = ar_para[1]
        AR_2[i] = ar_para[2]
        AR_3[i] = ar_para[3]


    return np.concatenate([MAV, RMS, WL, SSCn, ZCn, AR_1, AR_2, AR_3])
    # return np.concatenate([MAV, RMS, WL, SSCn, ZCn])


## calculate imu features within a window
def calcuImuFeatures(imu_window_data):
    abs_imu_data = np.abs(imu_window_data)

    max_values = np.max(imu_window_data, axis=0)
    min_values = np.min(imu_window_data, axis=0)
    mean_values = np.mean(imu_window_data, axis=0)
    std_values = np.std(imu_window_data, axis=0)  # this is included in feature set 0, but excluded in feature set 1
    return np.concatenate([max_values, min_values, mean_values, std_values])


## extract features for emg data in an experiment round, which contains multiple windows
def labelEmgFeatures(gait_event_label, gait_event_emg, window_size, increment):
    emg_feature_labelled = {}
    emg_repetition_features = {}

    event_time = datetime.datetime.now()
    for repetition_number, emg_per_repetition in enumerate(gait_event_emg):  # each gait event contains multiple repetitions of experiment data
        emg_window_features = []  # reset the feature values for a new repetition
        repetition_time = datetime.datetime.now()
        for i in range(0, len(emg_per_repetition) - window_size + 1, increment):  # if window_size=512, increment=32, the sample number is 17 per repetition
            emg_window_data = emg_per_repetition[i:i + window_size, :]
            emg_window_features.append(calcuEmgFeatures(emg_window_data).tolist())  # convert numpy to list for dict storage
        emg_repetition_features[f"repetition_{repetition_number}_features"] = emg_window_features  # add the repetition results to a dict
        print(f"repetition time:{gait_event_label}", multiprocessing.current_process().name, datetime.datetime.now() - repetition_time)

    # the features are stored in the dict below, which can also be regarded as a labeling process
    emg_feature_labelled[f"{gait_event_label}_features"] = emg_repetition_features
    print(f"event time:{gait_event_label}", multiprocessing.current_process().name, datetime.datetime.now() - event_time)
    return emg_feature_labelled


# ## old version
# def labelEmgFeatures(gait_event_label, gait_event_emg, window_size, increment):
#     emg_feature_labelled = {}
#     emg_window_features = []
#
#     event_time = datetime.datetime.now()
#     for per_repetition_emg in gait_event_emg:  # each gait event contains multiple repetitions of experiment data
#         round_time = datetime.datetime.now()
#         for i in range(0, len(per_repetition_emg) - window_size + 1, increment):  # if window_size=512, increment=32, the sample number is 17 per repetition
#             emg_window_data = per_repetition_emg[i:i + window_size, :]
#             emg_window_features.append(calcuEmgFeatures(emg_window_data))
#         print(f"round time:{gait_event_label}", multiprocessing.current_process().name, datetime.datetime.now() - round_time)
#
#     # the features are stored in the dict below, which can also be regarded as labeling
#     emg_feature_labelled[f"{gait_event_label}_features"] = np.array(emg_window_features)
#     print(f"event time:{gait_event_label}", multiprocessing.current_process().name, datetime.datetime.now() - event_time)
#     return emg_feature_labelled
