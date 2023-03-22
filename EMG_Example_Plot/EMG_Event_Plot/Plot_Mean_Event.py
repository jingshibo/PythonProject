## import
from EMG_Example_Plot.EMG_Event_Plot.Functions import Raw_Emg_Manipulation
from collections import defaultdict

## save all subject results
all_subjects = {}


## load average event values
subject = 'Shibo'
version = 1
result_set = 1
all_subjects[subject] = Raw_Emg_Manipulation.loadAverageEvent(subject, version, result_set)


subject = 'Zehao'
version = 0
result_set = 1
all_subjects[subject] = Raw_Emg_Manipulation.loadAverageEvent(subject, version, result_set)


subject = 'Number1'
version = 0
result_set = 0
all_subjects[subject] = Raw_Emg_Manipulation.loadAverageEvent(subject, version, result_set)


subject = 'Number2'
version = 0
result_set = 0
all_subjects[subject] = Raw_Emg_Manipulation.loadAverageEvent(subject, version, result_set)


subject = 'Number3'
version = 0
result_set = 1
all_subjects[subject] = Raw_Emg_Manipulation.loadAverageEvent(subject, version, result_set)


subject = 'Number4'
version = 0
result_set = 1
all_subjects[subject] = Raw_Emg_Manipulation.loadAverageEvent(subject, version, result_set)


subject = 'Number5'
version = 0
result_set = 1
all_subjects[subject] = Raw_Emg_Manipulation.loadAverageEvent(subject, version, result_set)


##  calculate the average emg series value at each gait event
sum_emg_1 = defaultdict(float) # create a default dictionary with zero as default value
sum_emg_2 = defaultdict(float) # create a default dictionary with zero as default value
for subject_number, subject_value in all_subjects.items():
    for emg_number, emg_value in subject_value.items():
        for gait_event, event_value in emg_value.items():
            if emg_number == 'emg_1_mean_events':
                sum_emg_1[gait_event] += event_value
            elif emg_number == 'emg_2_mean_events':
                sum_emg_2[gait_event] += event_value
mean_emg_1 = {event: value / len(all_subjects) for event, value in dict(sum_emg_1).items()}
mean_emg_2 = {event: value / len(all_subjects) for event, value in dict(sum_emg_2).items()}


## plot the signals
Raw_Emg_Manipulation.plotMeanEvent(mean_emg_1)
