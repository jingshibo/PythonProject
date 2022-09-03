## import mudules
import pandas as pd
from scipy import signal, ndimage
from RawData.Utility_Functions import Insole_Emg_Recovery
from Processing.Utility_Functions import Data_Reshaping


## upsamling insole data to match emg
def upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned):
    # upsample insole data to every 5ms (abandoned)
    # upsampled_left_insole = upsampleInsoleData(left_insole_aligned).reset_index()
    # upsampled_right_insole = upsampleInsoleData(right_insole_aligned).reset_index()

    # check if there are emg data lost
    emg_timestamp = pd.to_datetime(emg_aligned.iloc[:, 0])
    expected_number = (emg_timestamp.iloc[-1] - emg_timestamp.iloc[0]).total_seconds() * 1000 * 2 # the number of emg value expected within the period
    real_number = len(emg_timestamp)
    print("expected emg number:", expected_number, "real emg number:", real_number, "number difference:", expected_number - real_number)

    if abs(expected_number - real_number) >= 20:  # 20 is a self-selected threshold
        raise Exception("Abnormal EMG Data Number")  # in this case, you need to recover the lost data in EMG
    else:
        # upsample insole data to 2000Hz same to EMG data
        upsampled_left_insole = Insole_Emg_Recovery.upsampleInsoleEqualToEMG(left_insole_aligned, emg_aligned)
        upsampled_right_insole = Insole_Emg_Recovery.upsampleInsoleEqualToEMG(right_insole_aligned, emg_aligned)
        return upsampled_left_insole, upsampled_right_insole


## filtering insole data
def filterInsole(upsampled_left_insole, upsampled_right_insole):
    # filtering insole signal after upsampling
    sos = signal.butter(4, [20], fs=2000, btype="lowpass", output='sos')
    left_insole_filtered = signal.sosfiltfilt(sos, upsampled_left_insole.iloc[:, 1:193], axis=0)  # only filter the force measurements
    right_insole_filtered = signal.sosfiltfilt(sos, upsampled_right_insole.iloc[:, 1:193], axis=0)
    left_insole_filtered = pd.DataFrame(left_insole_filtered)
    left_insole_filtered.insert(0, "timestamp", upsampled_left_insole.iloc[:, 0])  # add timestamp column to the beginning again
    right_insole_filtered = pd.DataFrame(right_insole_filtered)
    right_insole_filtered.insert(0, "timestamp", upsampled_right_insole.iloc[:, 0])  # add timestamp column to the beginning again
    return left_insole_filtered, right_insole_filtered


## filtering EMG data
def filterEmg(emg_measurements, notch=False, quality_factor=10):
    sos = signal.butter(4, [20, 400], fs=2000, btype="bandpass", output='sos')
    emg_bandpass_filtered = signal.sosfiltfilt(sos, emg_measurements, axis=0)  # only filter the emg measurements
    emg_filtered = emg_bandpass_filtered
    if notch:  # to remove power line interference
        b, a = signal.iircomb(50, quality_factor, fs=2000, ftype='notch')
        emg_notch_filtered = signal.filtfilt(b, a, pd.DataFrame(emg_bandpass_filtered), axis=0)
        emg_filtered = emg_notch_filtered
    # compensate bad emg channels (median filtering)
    emg_filtered = pd.DataFrame(ndimage.median_filter(emg_filtered, mode='nearest', size=3))
    return pd.DataFrame(emg_filtered)


## preprocess all sensor data
def preprocessSensorData(left_insole_aligned, right_insole_aligned, emg_aligned, filterInsole = False, notchEMG = False, quality_factor=10):
    # upsampling insole data
    left_insole_preprocessed, right_insole_preprocessed = upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned)
    if filterInsole:
        left_insole_preprocessed, right_insole_preprocessed = filterInsole(left_insole_preprocessed, right_insole_preprocessed)
    # filtering emg data
    if emg_aligned.shape[1] >= 64 and emg_aligned.shape[1] < 128:  # if one sessantaquattro data
        emg_filtered = filterEmg(emg_aligned.iloc[:, 3:67], notchEMG, quality_factor)
        emg_reordered = Data_Reshaping.reorderElectrodes(emg_filtered)
    elif emg_aligned.shape[1] >= 128 and emg_aligned.shape[1] < 192:  # if two sessantaquattro data
        emg1_preprocessed = filterEmg(emg_aligned.iloc[:, 3:67], notchEMG, quality_factor)  # extract only emg measurement data
        emg2_preprocessed = filterEmg(emg_aligned.iloc[:, 73:137], notchEMG, quality_factor)
        emg1_reordered = Data_Reshaping.reorderElectrodes(emg1_preprocessed)
        emg2_reordered = Data_Reshaping.reorderElectrodes(emg2_preprocessed)
        emg_reordered = pd.concat([emg1_reordered, emg2_reordered], axis=1, ignore_index=True)
    return left_insole_preprocessed, right_insole_preprocessed, emg_reordered



