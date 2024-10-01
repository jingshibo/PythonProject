##
import copy
from Bipolar_EMG.Models import Dataset_Model, Results_Plot
import numpy as np


# ##  previous hdsemg-derived bipolar results
# bipolar_accuracy_from_hdsemg = [
#     67.9401, 87.6155, 93.0836, 90.4619, 95.3558, 95.0312, 98.2272, 99.0012, 98.9263, 99.6754,
#     78.583, 85.8704, 89.7368, 97.004, 97.085, 94.2915, 96.8826, 95.3441, 97.4696, 98.2996,
#     85.9954, 86.5741, 93.8889, 93.6574, 93.7037, 92.9861, 95.4398, 96.1343, 98.7731, 99.375,
#     59.8616, 76.2399, 82.9527, 92.4798, 93.0565, 86.4821, 95.6863, 93.4256, 94.6251, 97.8085,
#     58.8948, 77.1945, 86.6738, 91.4984, 94.5377, 96.6631, 96.3656, 95.2816, 96.9394, 98.9586,
#     77.0613, 77.8436, 91.7125, 87.8224, 91.649, 90.6554, 93.5307, 95.3277, 94.9049, 96.1311,
#     71.2804, 91.0375, 91.2362, 95.7174, 95.9382, 95.3642, 97.1744, 93.9514, 94.7682, 97.5717
# ]
# bipolar_accuracy_from_hdsemg_matrix = np.reshape(bipolar_accuracy_from_hdsemg, (7, 10))
# accuracy_mean = np.mean(bipolar_accuracy_from_hdsemg_matrix, axis=0)
# accuracy_std = np.std(bipolar_accuracy_from_hdsemg_matrix, axis=0)
# Results_Plot.plotOldBoxBipolar(bipolar_accuracy_from_hdsemg_matrix, RF_accuracy=accuracy_mean[0])


## model set
sensor_sets = {
    'emg_0': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM'],  # three front and back
    'emg_1': ['RF', 'BF', 'VM', 'TA', 'GM'],  # upper three + lower two
    'emg_2': ['RF', 'BF', 'TA', 'SL', 'GM'],  # upper two + lower three
    'emg_3': ['RF', 'BF', 'TA', 'GM'],  # upper two + lower two
    'emg_4': ['VM', 'BF', 'TA', 'SL'],  # another upper two + lower two
    'emg_5': ['VM', 'RF', 'GM', 'SL'],  # another upper two + lower two
    'emg_6': ['RF', 'TA', 'VM'],  # all front
    'emg_7': ['BF', 'SL', 'GM'],  # all back
    'emg_8': ['RF', 'BF', 'VM'],  # upper three
    'emg_9': ['TA', 'SL', 'GM'],  # lower three
    'emg_10': ['RF', 'TA'],  # front two
    'emg_11': ['BF', 'GM'],  # back two
    'emg_12': ['RF', 'BF'],  # upper two as agonist-antagonist pair
    'emg_13': ['TA', 'GM'],  # lower two as agonist-antagonist pair
    'emg_14': ['BF', 'TA'],  # front one + back one
    'emg_15': ['RF'],
    'emg_16': ['TA'],
    'emg_17': ['BF'],
    'emg_18': ['SL'],
    'emg_19': ['VM'],
    'emg_20': ['GM'],
    # # only imu
    # 'imu_0': ['LL', 'FT', 'UL'],
    # 'imu_1': ['FT', 'UL'],
    # 'imu_2': ['LL', 'UL'],
    # 'imu_3': ['LL'],
    # 'imu_4': ['FT'],
    # 'imu_5': ['UL'],
    # # emg+imu
    # 'emg_imu_0': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'LL', 'FT', 'UL'],
    # 'emg_imu_1': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'LL', 'UL'],
    # 'emg_imu_2': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'FT'],
    # 'emg_imu_3': ['RF', 'TA', 'BF', 'GM', 'LL', 'FT', 'UL'],
    # 'emg_imu_4': ['RF', 'TA', 'BF', 'GM', 'LL', 'UL'],
    # 'emg_imu_5': ['RF', 'TA', 'BF', 'GM', 'FT'],
}


##
# subjects = ['Number1', 'Number2', 'Number3', 'Number5', 'Number7', 'Number9', 'Number10', 'Number4', 'Number8']  # Add 'Number4' and 'Number8' if needed
subjects = ['Number2', 'Number3', 'Number5', 'Number7']
result_set = 0

all_subjects = {} # save all subject results
for subject in subjects:
    subject_results = {}
    for sensor_set in sensor_sets.values():
        model_type = '+'.join(sensor_set)
        subject_results[f'{model_type}_{result_set}'] = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
    all_subjects[subject] = subject_results


##  calculate accuracy across subjects for each bipolar EMG dataset
subjects_data = copy.deepcopy(all_subjects)
average_by_bipolar = Results_Plot.average_bipolar_mix_subject(subjects_data)
average_by_subject = Results_Plot.average_accuracy_by_subject(all_subjects)

##  plot box accuracy grouped by number of bipolarEMG
combined_by_bipolar = Results_Plot.aggregate_results(subjects_data)
rf_accuracy = average_by_bipolar[f'RF_{result_set}']['accuracy_mean']  # accuracy of rectus femoris bipolar EMG
combined_by_bipolar_group = Results_Plot.aggregate_by_bipolar_number(combined_by_bipolar)
Results_Plot.plotBipolarBoxAccuracy(combined_by_bipolar_group, RF_accuracy=rf_accuracy, RF_accuracy_old=68.1)

##  plot average accuracy grouped by number of bipolarEMG
results_by_muscle_number = Results_Plot.groupResultByMuscleNumber(average_by_bipolar, result_set)
average_by_bipolar_group = Results_Plot.calculate_bipolar_group_mean(combined_by_bipolar_group)
Results_Plot.calculateTtestValues(average_by_subject, average_by_bipolar_group)
Results_Plot.plotMeanAccuracy(average_by_bipolar_group, results_by_muscle_number, hdsemg_accuracy=95.4, derived_12_accuracy=92.8)

