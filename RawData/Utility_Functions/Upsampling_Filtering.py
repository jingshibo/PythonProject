## import mudules
import pandas as pd
import numpy as np
from scipy import signal, ndimage
from Pre_Processing.Utility_Functions import Data_Reshaping
from scipy.interpolate import PchipInterpolator


## upsample insole data to every 0.5ms to match EMG (abandoned method, because the insole sampling rate is not exact 40Hz)
def upsampleInsoleDataTo2000(insole_data):
    only_measured_data = insole_data.iloc[:, 3:]  # extract only measurement value columns
    only_measured_data.iloc[:, 0] = pd.to_datetime(only_measured_data.iloc[:, 0], unit='ms') # convert timestamp string to datetime object
    only_measured_data = only_measured_data.set_index([3])  # set the timestamp column as datetime index
    upsampled_sensor_data = only_measured_data.resample('0.5ms').asfreq()  # insert row of NaN every 0.5ms
    upsampled_sensor_data = upsampled_sensor_data.interpolate(method='pchip', limit_direction='forward', axis=0) # impute the NaN missing values
    return upsampled_sensor_data

## upsample insole data to exactly the same number as EMG
def upsampleInsoleEqualToEMG(insole_data, emg_data):
    x = np.arange(len(insole_data))
    y = insole_data.iloc[:, 3:].to_numpy()  # only extract measurement value columns
    f = PchipInterpolator(x, y)
    x_upsampled = np.linspace(min(x), max(x), len(emg_data))
    y_upsampled = f(x_upsampled)
    insole_upsampled = pd.DataFrame(y_upsampled)
    return insole_upsampled

## upsamling insole data to match emg
def upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned):
    # upsample insole data to every 5ms (abandoned)
    # upsampled_left_insole = upsampleInsoleData(left_insole_aligned).reset_index()
    # upsampled_right_insole = upsampleInsoleData(right_insole_aligned).reset_index()

    # check if there are emg data lost
    emg_timestamp = pd.to_datetime(emg_aligned.iloc[:, 0])
    expected_number = (emg_timestamp.iloc[-1] - emg_timestamp.iloc[0]).total_seconds() * 1000 * 2 # the number of emg value expected within the period
    real_number = len(emg_timestamp)
    print("expected emg number:", expected_number, "real emg number:", real_number, "missing emg number:", expected_number - real_number)

    # upsample insole data to 2000Hz same to EMG data
    upsampled_left_insole = upsampleInsoleEqualToEMG(left_insole_aligned, emg_aligned)
    upsampled_right_insole = upsampleInsoleEqualToEMG(right_insole_aligned, emg_aligned)

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
    left_insole_filtered.columns = list(range(0, left_insole_filtered.shape[1]))
    right_insole_filtered.columns = list(range(0, right_insole_filtered.shape[1]))
    return left_insole_filtered, right_insole_filtered


## filtering EMG data
def filterEmg(emg_measurements, notch=False, median_filtering=True, quality_factor=10):
    sos = signal.butter(4, [20, 400], fs=2000, btype="bandpass", output='sos')
    emg_bandpass_filtered = signal.sosfiltfilt(sos, emg_measurements, axis=0)  # only filter the emg measurements
    emg_filtered = emg_bandpass_filtered
    if notch:  # to remove power line interference
        b, a = signal.iirnotch(50, quality_factor, 2000)  # notch filter
        # b, a = signal.iircomb(50, quality_factor, fs=2000, ftype='notch')  # comb filter
        emg_notch_filtered = signal.filtfilt(b, a, pd.DataFrame(emg_bandpass_filtered), axis=0)
        emg_filtered = emg_notch_filtered
    # compensate bad emg channels (median filtering)
    if median_filtering:
        emg_filtered = pd.DataFrame(ndimage.median_filter(emg_filtered, mode='nearest', size=3))
    else:
        emg_filtered = emg_filtered
    return pd.DataFrame(emg_filtered)


## rectify emg and get envelope
def getEmgRectEnvelope(emg_reordered, cutoff=10):
    emg_rectfied = emg_reordered.abs().to_numpy()
    sos = signal.butter(4, cutoff, fs=2000, btype="lowpass", output='sos')
    emg_envelope = signal.sosfiltfilt(sos, emg_rectfied, axis=0)  # only filter the emg measurements
    return pd.DataFrame(emg_envelope)

## preprocess all sensor data
def preprocessSensorData(left_insole_aligned, right_insole_aligned, emg_aligned, envelope_cutoff=10, insoleFiltering=False, notchEMG=False,
        median_filtering=True, quality_factor=10):
    # upsampling insole data
    left_insole_preprocessed, right_insole_preprocessed = upsampleInsole(left_insole_aligned, right_insole_aligned, emg_aligned)
    if insoleFiltering:
        left_insole_preprocessed, right_insole_preprocessed = filterInsole(left_insole_preprocessed, right_insole_preprocessed)
    # filtering emg data
    if emg_aligned.shape[1] >= 64 and emg_aligned.shape[1] < 128:  # if one sessantaquattro data
        emg_filtered = filterEmg(emg_aligned.iloc[:, 3:67], notchEMG, median_filtering, quality_factor=quality_factor) # extract only emg measurement data
        emg_filtered = Data_Reshaping.insertElectrode(emg_filtered)
        emg_reordered = Data_Reshaping.reorderElectrodes(emg_filtered)
        emg_envelope = getEmgRectEnvelope(emg_reordered, cutoff=envelope_cutoff)  # rectify and envelope EMG
    elif emg_aligned.shape[1] >= 128 and emg_aligned.shape[1] < 192:  # if two sessantaquattro data
        emg1_filtered = filterEmg(emg_aligned.iloc[:, 3:67], notchEMG, median_filtering, quality_factor=quality_factor)
        emg2_filtered = filterEmg(emg_aligned.iloc[:, 73:137], notchEMG, median_filtering, quality_factor=quality_factor)
        emg1_filtered = Data_Reshaping.insertElectrode(emg1_filtered)
        emg2_filtered = Data_Reshaping.insertElectrode(emg2_filtered)
        emg_filtered = pd.concat([emg1_filtered, emg2_filtered], axis=1, ignore_index=True)
        # change emg channel order
        emg1_reordered = Data_Reshaping.reorderElectrodes(emg1_filtered)
        emg2_reordered = Data_Reshaping.reorderElectrodes(emg2_filtered)
        emg_reordered = pd.concat([emg1_reordered, emg2_reordered], axis=1, ignore_index=True)
        # get emg envelope
        emg1_envelope = getEmgRectEnvelope(emg1_reordered, cutoff=envelope_cutoff)
        emg2_envelope = getEmgRectEnvelope(emg2_reordered, cutoff=envelope_cutoff)
        emg_envelope = pd.concat([emg1_envelope, emg2_envelope], axis=1, ignore_index=True)
    return left_insole_preprocessed, right_insole_preprocessed, emg_filtered, emg_reordered, emg_envelope


