import os
import pandas as pd
from Transition_Prediction.RawData.Utility_Functions import Insole_Emg_Alignment, Two_Insoles_Alignment, Insole_Emg_Recovery


## read align parameters and reconstruct aligned data (abandoned as it is too slow)
def readAlignedData(subject, session, mode):
    data_dir = 'D:\Data\Insole_Emg'
    data_file_name = f'subject_{subject}_session_{session}_{mode}'

    # data_file_name = "subject0_20220810_205047"
    # left_insole_file = f'left_insole\left_{data_file_name}.csv'
    # right_insole_file = f'right_insole\\right_{data_file_name}.csv'
    # emg_file = f'emg\emg_{data_file_name}.csv'

    left_insole_file = f'subject_{subject}\\raw_{mode}\left_insole\left_{data_file_name}.csv'
    right_insole_file = f'subject_{subject}\\raw_{mode}\\right_insole\\right_{data_file_name}.csv'
    emg_file = f'subject_{subject}\\raw_{mode}\emg\emg_{data_file_name}.csv'

    left_insole_path = os.path.join(data_dir, left_insole_file)
    right_insole_path = os.path.join(data_dir, right_insole_file)
    emg_path = os.path.join(data_dir, emg_file)

    ## read and impute sensor data
    insole_sampling_period = 25  # insole sampling period

    # emg
    raw_emg_data = pd.read_csv(emg_path, sep=',', header=None, dtype='int16',
        converters={0: str, 1: str, 2: str})  # change data type for faster reading

    # left insole
    raw_left_data = pd.read_csv(left_insole_path, sep=',', header=None)
    recovered_left_data = Insole_Emg_Recovery.insertInsoleMissingRow(raw_left_data, insole_sampling_period)  # add missing rows with NaN values

    # right insole
    raw_right_data = pd.read_csv(right_insole_path, sep=',', header=None)
    recovered_right_data = Insole_Emg_Recovery.insertInsoleMissingRow(raw_right_data, insole_sampling_period)  # add missing rows with NaN values

    ## read alignment parameters
    left_start_timestamp, right_start_timestamp, left_end_timestamp, right_end_timestamp, emg_start_timestamp, emg_end_timestamp = \
        Insole_Emg_Alignment.readAlignParameters(subject, session, mode)
    print(left_start_timestamp, right_start_timestamp, left_end_timestamp, right_end_timestamp, emg_start_timestamp, emg_end_timestamp)

    ## get aligned sensor data
    _, left_begin_cropped, right_begin_cropped = Two_Insoles_Alignment.alignInsoleBeginTimestamp(  # to view combine_cropped_begin data
        left_start_timestamp, right_start_timestamp, recovered_left_data, recovered_right_data)
    left_insole_aligned, right_insole_aligned = Two_Insoles_Alignment.alignInsoleEndTimestamp(left_end_timestamp, right_end_timestamp,
        left_begin_cropped, right_begin_cropped)
    emg_aligned = Insole_Emg_Alignment.alignInsoleEmgTimestamp(raw_emg_data, left_insole_aligned, right_insole_aligned)

    return left_insole_aligned, right_insole_aligned, emg_aligned
