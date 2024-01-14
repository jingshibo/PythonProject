'''
    extract steady-state emg and imu data
'''


## import modules
from Bipolar_EMG.Utility_Functions import Emg_Imu_Preprocessing
from Transition_Prediction.RawData.Utility_Functions import Upsampling_Filtering, Insole_Data_Splition
from scipy import signal
import pandas as pd


## read and preprocess aligned data
subject = 'Number8'
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
    'downslope': [(15000, 45000), (68000, 98000)],
    'downstairs': [(15000, 25000), (46000, 56000), (74000, 84000), (102000, 112000), (129000, 139000), (156000, 166000), (183000, 193000),
        (210000, 220000), (237000, 247000), (264000, 274000)],
    'level': [(20000, 120000)],
    'standing': [(20000, 60000)],
    'upslope': [(15000, 45000), (62000, 92000)],
    'upstairs': [(14000, 24000), (40000, 50000), (67000, 77000), (95000, 105000), (120000, 130000), (152000, 162000), (180000, 190000),
        (210000, 220000), (237000, 247000), (264000, 274000)], }

##
subject = 'Number3'
split_parameters = {
    'downslope': [(15000, 45000), (65000, 95000)],
    'downstairs': [(15000, 25000), (43000, 53000), (76000, 86000), (107000, 117000), (138000, 148000), (170000, 180000), (200000, 210000),
        (230000, 240000), (260000, 270000), (290000, 300000)],
    'level': [(15000, 115000)],
    'standing': [(30000, 70000)],
    'upslope': [(24000, 54000), (70000, 100000)],
    'upstairs': [(15000, 25000), (43000, 53000), (73000, 83000), (103000, 113000), (135000, 145000), (168000, 178000), (200000, 210000),
        (233000, 243000), (265000, 275000), (296000, 306000)], }

##
subject = 'Number5'
split_parameters = {
    'downslope': [(15000, 45000), (65000, 95000)],
    'downstairs': [(12000, 22000), (42000, 52000), (75000, 85000), (105000, 115000), (133000, 143000), (164000, 174000), (193000, 203000)],
    'level': [(15000, 95000)],
    'standing': [(30000, 70000)],
    'upslope': [(20000, 50000), (65000, 95000)],
    'upstairs': [(14000, 24000), (44000, 54000), (70000, 80000), (97000, 107000), (125000, 135000), (155000, 165000), (185000, 195000)],
}

##
subject = 'Number7'
split_parameters = {
    'downslope': [(11000, 41000), (56000, 86000)],
    'downstairs': [(18000, 28000), (45000, 55000), (78000, 88000), (108000, 118000), (141000, 151000), (170000, 180000), (200000, 210000)],
    'level': [(10000, 70000)],
    'standing': [(40000, 80000)],
    'upslope': [(10000, 40000), (53000, 82000)],
    'upstairs': [(12000, 22000), (34000, 44000), (53000, 63000), (82000, 92000), (110000, 120000), (140000, 150000), (169000, 179000)],
}

##
subject = 'Number9'
split_parameters = {
    'downslope': [(14000, 44000), (70000, 100000)],
    'downstairs': [(14000, 24000), (47000, 57000), (84000, 94000), (117000, 127000), (150000, 160000), (183000, 193000), (213000, 223000),
        (243000, 253000), (272000, 282000), (300000, 310000)],
    'level': [(15000, 115000)],
    'standing': [(40000, 80000)],
    'upslope': [(13000, 43000), (64000, 94000)],
    'upstairs': [(33000, 43000), (60000, 70000), (86000, 96000), (115000, 125000), (141000, 151000), (168000, 178000), (195000, 205000),
        (222000, 232000), (250000, 260000), (275000, 285000)],
}

##
subject = 'Number10'
split_parameters = {'downslope': [(14000, 44000), (66000, 96000)],
    'downstairs': [(13000, 23000), (38000, 48000), (65000, 75000), (95000, 105000), (128000, 138000), (158000, 168000), (188000, 198000),
        (221000, 231000), (251000, 261000)], 'level': [(15000, 110000)], 'standing': [(40000, 80000)],
    'upslope': [(16000, 46000), (62000, 92000)],
    'upstairs': [(16000, 26000), (42000, 52000), (69000, 79000), (95000, 105000), (122000, 132000), (150000, 160000), (180000, 190000),
        (208000, 218000), (238000, 248000)], }

##
subject = 'Number4'
split_parameters = {'downslope': [(20000, 90000)],
    'downstairs': [(11000, 21000), (39000, 49000), (68000, 78000), (98000, 108000), (129000, 139000), (155000, 165000), (185000, 195000),
        (212000, 222000), (240000, 250000), (268000, 278000)],
    'level': [(15000, 95000)],
    'standing': [(20000, 60000)],
    'upslope': [(15000, 75000)],
    'upstairs': [(14000, 24000), (42000, 52000), (74000, 84000), (105000, 115000), (136000, 146000), (166000, 176000), (195000, 205000),
        (224000, 234000), (250000, 260000), (280000, 290000)], }

##
subject = 'Number8'
split_parameters = {'downslope': [(10000, 55000)],
    'downstairs': [(10000, 20000), (36500, 46500), (62000, 72000), (91000, 101000), (121000, 131000), (152000, 162000), (182000, 192000)],
    'level': [(20000, 90000)],
    'standing': [(30000, 70000)],
    'upslope': [(10000, 65000)],
    'upstairs': [(11000, 21000), (39000, 49000), (65000, 75000), (91000, 101000), (120000, 130000), (148000, 158000), (176000, 186000)]}


## save split parameters to json files
Insole_Data_Splition.saveSplitParameters(subject, split_parameters, project='Bipolar_Data')
