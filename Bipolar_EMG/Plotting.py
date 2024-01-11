##
from Bipolar_EMG.Models import Dataset_Model


## model set
all_subjects = {}  # save all subject results
model_type = '6emg+3imu'
result_set = 0


## get model results. Note: different from the GRU method, here you can decide the shift_unit as you want.
subject = 'Number1'
subject_results = Dataset_Model.loadResult(subject, model_type, result_set, project='Bipolar_Data')
all_subjects[subject] = subject_results



##  average accuracy across subjects
delay_groups = list(all_subjects['Shibo']['accuracy'].keys())  # list all transition types
average_accuracy = {delay_time: 0 for delay_time in delay_groups}  # initialize average accuracy list
average_cm = {delay_time: 0 for delay_time in delay_groups}  # initialize average cm list

for subject_number, subject_results in all_subjects.items():
    for key, value in subject_results.items():
        for delay_time, delay_value in value.items():
            if key == 'accuracy':
                average_accuracy[delay_time] = average_accuracy[delay_time] + delay_value
            elif key == 'cm_call':
                average_cm[delay_time] = average_cm[delay_time] + delay_value

for delay_time, delay_value in average_accuracy.items():
    average_accuracy[delay_time] = delay_value / len(all_subjects)
for delay_time, delay_value in average_cm.items():
    average_cm[delay_time] = delay_value / len(all_subjects)