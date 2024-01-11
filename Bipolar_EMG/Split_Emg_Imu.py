## import modules
from Bipolar_EMG.Utility_Functions import Emg_Imu_Preprocessing
from Transition_Prediction.RawData.Utility_Functions import Upsampling_Filtering, Insole_Data_Splition
from scipy import signal
import pandas as pd


## read and preprocess aligned data
subject = 'Number1'
mode = 'standing'

# read and plot aligned data
imu_aligned, emg_aligned = Emg_Imu_Preprocessing.readAlignedData(subject, mode, project='Bipolar_Data')
# upsampling imu data
imu_upsampled = Emg_Imu_Preprocessing.upsampleImuEqualToEmg(imu_aligned, emg_aligned)
Emg_Imu_Preprocessing.plotEmgImuAligned(emg_aligned, imu_upsampled, start_index=0, end_index=-1)
# filtering aligned emg and imu data
emg_filtered = Upsampling_Filtering.filterEmg(emg_aligned.iloc[:, 1:7], lower_limit=20, higher_limit=400, median_filtering=False)  # filter only emg data
sos = signal.butter(4, [45], fs=2000, btype="lowpass", output='sos')
imu_filtered = pd.DataFrame(signal.sosfiltfilt(sos, imu_upsampled.iloc[:, 1:], axis=0))  # only filter the measurements

# calculate and plot MAV values of emg and imu data
Emg_Imu_Preprocessing.plotMeanAbsValue(emg_filtered, imu_filtered)


## split results stored in dict (the values refer to the separation of effective durations)
subject = 'Number1'
split_parameters = {
    'downslope': [(14000, 45420), (62540, 96550)],
    'downstairs': [(16000, 26000), (45000, 55000), (73000, 83000), (103500, 113500), (131000, 141000), (161000, 171000), (190000, 200000),
        (220000, 230000), (246500, 256500), (275000, 285000)],
    'level': [(20000, 120000)],
    'standing': [(40000, 80000)],
    'upslope': [(14080, 45520), (61280, 91010)],
    'upstairs': [(13000, 25000), (42400, 52400), (72400, 82400), (99500, 109500), (127500, 137500), (154500, 164500), (181000, 191000),
        (208500, 218500), (235400, 245400), (259300, 269300)], }

##
subject = 'Number2'
split_parameters = {
    'downslope': [(), ()],
    'downstairs': [(), (), (), (), (), (), (), (), (), ()],
    'level': [()],
    'standing': [()],
    'upslope': [(), ()],
    'upstairs': [(), (), (), (), (), (), (), (), (), ()],
}

##
subject = 'Number3'
split_parameters = {
    'downslope': [(), ()],
    'downstairs': [(), (), (), (), (), (), (), (), (), ()],
    'level': [()],
    'standing': [()],
    'upslope': [(), ()],
    'upstairs': [(), (), (), (), (), (), (), (), (), ()],
}

##
subject = 'Number5'
split_parameters = {
    'downslope': [(), ()],
    'downstairs': [(), (), (), (), (), (), (), (), (), ()],
    'level': [()],
    'standing': [()],
    'upslope': [(), ()],
    'upstairs': [(), (), (), (), (), (), (), (), (), ()],
}

##
subject = 'Number7'
split_parameters = {
    'downslope': [(), ()],
    'downstairs': [(), (), (), (), (), (), (), (), (), ()],
    'level': [()],
    'standing': [()],
    'upslope': [(), ()],
    'upstairs': [(), (), (), (), (), (), (), (), (), ()],
}

##
subject = 'Number9'
split_parameters = {
    'downslope': [(), ()],
    'downstairs': [(), (), (), (), (), (), (), (), (), ()],
    'level': [()],
    'standing': [()],
    'upslope': [(), ()],
    'upstairs': [(), (), (), (), (), (), (), (), (), ()],
}

##
subject = 'Number10'
split_parameters = {
    'downslope': [(), ()],
    'downstairs': [(), (), (), (), (), (), (), (), (), ()],
    'level': [()],
    'standing': [()],
    'upslope': [(), ()],
    'upstairs': [(), (), (), (), (), (), (), (), (), ()],
}

## save split parameters to json files
Insole_Data_Splition.saveSplitParameters(subject, split_parameters, project='Bipolar_Data')
