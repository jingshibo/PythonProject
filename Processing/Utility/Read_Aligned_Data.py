import os
import pandas as pd
from PreProcessing.Utility import Align_Two_Insoles, Align_Insole_Emg, Recover_Insole

def readAlignedData(subject, session, mode):
    data_dir = 'D:\Data\Insole_Emg'
    data_file_name = f'subject_{subject}_session_{session}_{mode}'

    # data_file_name = "subject0_20220810_205047"
    # left_insole_file = f'left_insole\left_{data_file_name}.csv'
    # right_insole_file = f'right_insole\\right_{data_file_name}.csv'
    # emg_file = f'emg\emg_{data_file_name}.csv'

    left_insole_file = f'subject_{subject}\{mode}\left_insole\left_{data_file_name}.csv'
    right_insole_file = f'subject_{subject}\{mode}\\right_insole\\right_{data_file_name}.csv'
    emg_file = f'subject_{subject}\{mode}\emg\emg_{data_file_name}.csv'

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
    recovered_left_data = Recover_Insole.insertMissingRow(raw_left_data,
        insole_sampling_period)  # add missing rows with NaN values
    # right insole
    raw_right_data = pd.read_csv(right_insole_path, sep=',', header=None)
    recovered_right_data = Recover_Insole.insertMissingRow(raw_right_data,
        insole_sampling_period)  # add missing rows with NaN values


    ## read alignment parameters
    left_start_timestamp, right_start_timestamp, left_end_timestamp, right_end_timestamp = Align_Two_Insoles.readAlignData(
        subject, data_file_name)
    print(left_start_timestamp, right_start_timestamp, left_end_timestamp, right_end_timestamp)


    ## get aligned sensor data
    _, left_begin_cropped, right_begin_cropped = Align_Two_Insoles.alignInsoleBegin(  # to view combine_cropped_begin data
        left_start_timestamp, right_start_timestamp, recovered_left_data, recovered_right_data)
    left_insole_aligned, right_insole_aligned = Align_Two_Insoles.alignInsoleEnd(left_end_timestamp, right_end_timestamp,
        left_begin_cropped, right_begin_cropped)
    emg_aligned = Align_Insole_Emg.alignInsoleEmg(raw_emg_data, left_insole_aligned, right_insole_aligned)


    ## preprocess sensor data
    # upsampling insole data
    left_insole_upsampled, right_insole_upsampled = Align_Two_Insoles.upsampleInsole(left_insole_aligned,
        right_insole_aligned, emg_aligned)
    # left_insole_filtered, right_insole_filtered = Align_Two_Insoles.filterInsole(left_insole_upsampled, right_insole_upsampled)

    # filtering emg data
    emg_filtered = Align_Insole_Emg.filterEmg(emg_aligned, notch=False, quality_factor=10)

    return left_insole_upsampled, right_insole_upsampled, emg_filtered, left_insole_aligned, right_insole_aligned, emg_aligned