## import mudules
import pandas as pd
from scipy import signal
from RawData.Utility_Functions import Recover_Insole

## upsamling insole data to match emg
def upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned):
    # upsample insole data to every 5ms (abandoned)
    # upsampled_left_insole = upsampleInsoleData(left_insole_aligned).reset_index()
    # upsampled_right_insole = upsampleInsoleData(right_insole_aligned).reset_index()

    # check if there are emg data lost
    emg_timestamp = pd.to_datetime(emg_aligned.iloc[:, 0])
    expected_number = (emg_timestamp.iloc[-1] - emg_timestamp.iloc[0]).total_seconds() * 1000 * 2
    if abs(expected_number - len(emg_timestamp)) >= 100:
        raise Exception("EMG Data Lost")  # then you need to recover the lost data in EMG
    else:
        # upsample insole data to 2000Hz
        upsampled_left_insole = Recover_Insole.upsampleInsoleEqualToEMG(left_insole_aligned, emg_aligned)
        upsampled_right_insole = Recover_Insole.upsampleInsoleEqualToEMG(right_insole_aligned, emg_aligned)
        return upsampled_left_insole, upsampled_right_insole

## filtering insole data
def filterInsole(upsampled_left_insole, upsampled_right_insole):
    # filtering insole signal after upsampling
    sos  = signal.butter(4, [20], fs=2000, btype = "lowpass", output='sos')
    left_insole_filtered = signal.sosfiltfilt(sos, upsampled_left_insole.iloc[:,1:193], axis=0)
    right_insole_filtered = signal.sosfiltfilt(sos, upsampled_right_insole.iloc[:,1:193], axis=0)
    left_insole_filtered = pd.DataFrame(left_insole_filtered)
    left_insole_filtered.insert(0, "timestamp", upsampled_left_insole.iloc[:, 0]) # add timestamp column
    right_insole_filtered = pd.DataFrame(right_insole_filtered)
    right_insole_filtered.insert(0, "timestamp", upsampled_right_insole.iloc[:, 0]) # add timestamp column
    return left_insole_filtered, right_insole_filtered

## filtering EMG data
def filterEmg(emg_aligned, notch = False, quality_factor = 30):
    sos = signal.butter(4, [20, 400], fs=2000, btype="bandpass", output='sos')
    emg_bandpass_filtered = signal.sosfiltfilt(sos, emg_aligned.iloc[:, 3:67], axis=0)
    emg_filtered = emg_bandpass_filtered
    if notch:
        b, a = signal.iircomb(50, quality_factor, fs=2000, ftype='notch')
        emg_notch_filtered = signal.filtfilt(b, a, pd.DataFrame(emg_bandpass_filtered), axis=0)
        emg_filtered = emg_notch_filtered
    return pd.DataFrame(emg_filtered)


## preprocess sensor data
def preprocessSensorData(left_insole_aligned, right_insole_aligned, emg_aligned, filterInsole = False, notchEMG = False, quality_factor=10):
    # upsampling insole data
    left_insole_preprocessed, right_insole_preprocessed = upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned)
    if filterInsole:
        left_insole_preprocessed, right_insole_preprocessed = filterInsole(left_insole_preprocessed, right_insole_preprocessed)
    # filtering emg data
    emg_preprocessed = filterEmg(emg_aligned, notchEMG, quality_factor)
    return left_insole_preprocessed, right_insole_preprocessed, emg_preprocessed

