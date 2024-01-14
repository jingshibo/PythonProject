##
from Bipolar_EMG.Models import Dataset_Model, Plotting_Process


## model set
sensor_sets = {
    'emg_0': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM'],  # three front and back
    'emg_1': ['RF', 'TA', 'BF', 'GM'],  # front two + back two
    'emg_2': ['RF', 'TA', 'VM'],  # all front
    'emg_3': ['BF', 'SL', 'GM'],  # all back
    'emg_4': ['RF', 'BF', 'VM'],  # upper three
    'emg_5': ['TA', 'SL', 'GM'],  # lower three
    'emg_6': ['RF', 'TA'],  # front two
    'emg_7': ['BF', 'GM'],  # back two
    'emg_8': ['RF', 'BF'],  # upper two
    'emg_9': ['TA', 'GM'],  # lower two
    'emg_10': ['RF'],
    'emg_11': ['TA'],
    'emg_12': ['BF'],
    'emg_13': ['SL'],
    'emg_14': ['VM'],
    'emg_15': ['GM'],
    # only imu
    'imu_0': ['LL', 'FT', 'UL'],
    'imu_1': ['FT', 'UL'],
    'imu_2': ['LL', 'UL'],
    'imu_3': ['LL'],
    'imu_4': ['FT'],
    'imu_5': ['UL'],
    # emg+imu
    'emg_imu_0': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'LL', 'FT', 'UL'],
    'emg_imu_1': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'LL', 'UL'],
    'emg_imu_2': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'FT'],
    'emg_imu_3': ['RF', 'TA', 'BF', 'GM', 'LL', 'FT', 'UL'],
    'emg_imu_4': ['RF', 'TA', 'BF', 'GM', 'LL', 'UL'],
    'emg_imu_5': ['RF', 'TA', 'BF', 'GM', 'FT'],
}
all_subjects = {}  # save all subject results
result_set = 0


##
subject = 'Number1'
subject_results = {}
for sensor_set in sensor_sets.values():
    model_type = '+'.join(sensor_set)
    subject_results[f'{model_type}_{result_set}'] = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
all_subjects[subject] = subject_results

##
subject = 'Number2'
subject_results = {}
for sensor_set in sensor_sets.values():
    model_type = '+'.join(sensor_set)
    subject_results[f'{model_type}_{result_set}'] = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
all_subjects[subject] = subject_results

##
subject = 'Number3'
subject_results = {}
for sensor_set in sensor_sets.values():
    model_type = '+'.join(sensor_set)
    subject_results[f'{model_type}_{result_set}'] = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
all_subjects[subject] = subject_results

##
subject = 'Number5'
subject_results = {}
for sensor_set in sensor_sets.values():
    model_type = '+'.join(sensor_set)
    subject_results[f'{model_type}_{result_set}'] = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
all_subjects[subject] = subject_results

##
subject = 'Number7'
subject_results = {}
for sensor_set in sensor_sets.values():
    model_type = '+'.join(sensor_set)
    subject_results[f'{model_type}_{result_set}'] = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
all_subjects[subject] = subject_results

##
subject = 'Number9'
subject_results = {}
for sensor_set in sensor_sets.values():
    model_type = '+'.join(sensor_set)
    subject_results[f'{model_type}_{result_set}'] = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
all_subjects[subject] = subject_results

##
subject = 'Number10'
subject_results = {}
for sensor_set in sensor_sets.values():
    model_type = '+'.join(sensor_set)
    subject_results[f'{model_type}_{result_set}'] = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
all_subjects[subject] = subject_results



##  average accuracy across subjects
average_results = Plotting_Process.calculate_metrics(all_subjects)
Plotting_Process.calculate_mean_bipolar(average_results, bipolar_types=['RF_0', 'BF_0', 'SL_0', 'VM_0', 'GM_0'])


