'''
plot the filtered emg data from each channel at each gait event
'''


## modules
from Transition_Prediction.Pre_Processing.Utility_Functions import Data_Reshaping, Feature_Storage
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Transition_Prediction.Pre_Processing import Preprocessing
from Conditional_GAN.Data_Procesing import Plot_Emg
import matplotlib.pyplot as plt
import pandas as pd
import copy


## input emg labelled series data
subject = 'Number4'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [0, 1, 2, 5, 6]
sessions = [up_down_session, down_up_session]

lower_limit = 20
higher_limit = 400
envelope_cutoff = 100
envelope = True  # the output will always be rectified if set True

# labelled emg series data
split_parameters = Preprocessing.readSplitParameters(subject, version)
# combined_emg_labelled = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-700,
#     end_position=800, lower_limit=20, higher_limit=400, envelope_cutoff=400, notchEMG=False, median_filtering=True, reordering=True,
#     envelope=True)
# old_emg_preprocessed = Data_Preparation.removeSomeSamples(combined_emg_labelled)
combined_emg_labelled = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-900,
    end_position=800, lower_limit=lower_limit, higher_limit=higher_limit, envelope_cutoff=envelope_cutoff, notchEMG=False,
    median_filtering=True, reordering=True, envelope=envelope)
old_emg_envelope = Data_Preparation.removeSomeSamples(combined_emg_labelled)


## read and filter new data
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [5, 6, 7, 8, 9]
down_up_session = [4, 5, 6, 8, 9]
# up_down_session = [0]
# down_up_session = [0]
sessions = [up_down_session, down_up_session]

# labelled emg series data
split_parameters = Preprocessing.readSplitParameters(subject, version)
# combined_emg_labelled = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-1000,
#     end_position=800, lower_limit=20, higher_limit=400, envelope_cutoff=400, notchEMG=False, median_filtering=True, reordering=True,
#     envelope=True)
# new_emg_preprocessed = Data_Preparation.removeSomeSamples(combined_emg_labelled)
combined_emg_labelled = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-900,
    end_position=800, lower_limit=lower_limit, higher_limit=higher_limit, envelope_cutoff=envelope_cutoff, notchEMG=False,
    median_filtering=True, reordering=True, envelope=envelope)
new_emg_envelope = Data_Preparation.removeSomeSamples(combined_emg_labelled)


## calculate average values
old_emg_value = Plot_Emg.averageEmgValues(old_emg_envelope)
new_emg_value = Plot_Emg.averageEmgValues(new_emg_envelope)
# Plot using a single line of code
Plot_Emg.plotAverageValue(old_emg_value['emg_1_repetition_list'], "emg_LWLW", title='old_emg_LWLW', ylim=(0, 1000))
Plot_Emg.plotAverageValue(old_emg_value['emg_1_repetition_list'], "emg_SASA", title='old_emg_SASA', ylim=(0, 1000))
Plot_Emg.plotAverageValue(old_emg_value['emg_1_repetition_list'], "emg_LWSA", title='old_emg_LWSA', ylim=(0, 1000))
Plot_Emg.plotAverageValue(new_emg_value['emg_1_repetition_list'], "emg_LWLW", title='new_emg_LWLW', ylim=(0, 1000))
Plot_Emg.plotAverageValue(new_emg_value['emg_1_repetition_list'], "emg_SASA", title='new_emg_SASA', ylim=(0, 1000))
Plot_Emg.plotAverageValue(new_emg_value['emg_1_repetition_list'], "emg_LWSA", title='new_emg_LWSA', ylim=(0, 1000))


## plot average channel value in a single plot
# List of datasets and titles
datasets1 = [(pd.DataFrame(old_emg_value['emg_1_repetition_list']["emg_LWLW"])).mean(axis=1),
    (pd.DataFrame(old_emg_value['emg_1_repetition_list']["emg_SDSD"])).mean(axis=1),
    (pd.DataFrame(old_emg_value['emg_1_repetition_list']["emg_LWSD"])).mean(axis=1)]
datasets2 = [(pd.DataFrame(new_emg_value['emg_1_repetition_list']["emg_LWLW"])).mean(axis=1),
    (pd.DataFrame(new_emg_value['emg_1_repetition_list']["emg_SDSD"])).mean(axis=1),
    (pd.DataFrame(new_emg_value['emg_1_repetition_list']["emg_LWSD"])).mean(axis=1)]
titles1 = ["old_emg_LWLW", "old_emg_SASA", "old_emg_LWSA"]
titles2 = ["new_emg_LWLW", "new_emg_SASA", "new_emg_LWSA"]

# Create a single plot for all six datasets with y-limit set to 1000
fig, ax = plt.subplots(figsize=(12, 6))
# Plot all 6 datasets in the single figure
for data, title in zip(datasets1 + datasets2, titles1 + titles2):
    data.plot(ax=ax, label=title)
# Add titles, labels, and legend
ax.set_title('All Six Datasets')
ax.set_xlabel('Row Index')
ax.set_ylabel('Average Value')
ax.set_ylim(0, 1000)
ax.legend()
plt.tight_layout()
plt.show()


##  plot summed series emg data
left_number = 000
right_number = 1800
emg_1_value = copy.deepcopy(old_emg_value['emg_1_event_mean'])
for event, value in old_emg_value['emg_1_event_mean'].items():
    emg_1_value[event] = value[left_number:right_number]
emg_2_value = copy.deepcopy(old_emg_value['emg_2_event_mean'])
for event, value in old_emg_value['emg_2_event_mean'].items():
    emg_2_value[event] = value[left_number:right_number]
pd.DataFrame(emg_1_value).plot(subplots=True, layout=(4, 4), title="EMG 1")
pd.DataFrame(emg_2_value).plot(subplots=True, layout=(4, 4), title="EMG 2")


##  plot summed series emg data by group
emd = emg_2_value
x = list(range(left_number-1000, right_number-1000))
plt.figure()
plt.plot(x, emd["emg_LWLW"], x, emd["emg_LWSA"], x, emd["emg_SASA"], x, emd["emg_SALW"])
plt.legend(['emg_LWLW', 'emg_LWSA', 'emg_SASA', 'emg_SALW'])
plt.title("LWSA")
plt.figure()
plt.plot(x, emd["emg_LWLW"], x, emd["emg_LWSD"], x, emd["emg_SDSD"], x, emd["emg_SDLW"])
plt.legend(['emg_LWLW', 'emg_LWSD', 'emg_SDSD', 'emg_SDLW'])
plt.title("LWSD")


##  plot summed series emg data by group
emd = emg_2_value
x = list(range(left_number-1000, right_number-1000))
plt.figure()
plt.plot(x, emd["emg_LWLW"], x, emd["emg_LWSA"], x, emd["emg_LWSD"], x, emd["emg_LWSS"])
plt.legend(['emg_LWLW', 'emg_LWSA', 'emg_LWSD', 'emg_LWSS'])
plt.title("LW")
# plt.figure()
# plt.plot(x, emd["emg_SASA"], x, emd["emg_SALW"], x, emd["emg_SASS"])
# plt.legend(['emg_SASA', 'emg_SALW', 'emg_SASS'])
# plt.title("SA")
# plt.figure()
# plt.plot(x, emd["emg_SDSD"], x, emd["emg_SDLW"], x, emd["emg_SDSS"])
# plt.legend(['emg_SDSD', 'emg_SDLW', 'emg_SDSS'])
# plt.title("SD")
# plt.figure()
# plt.plot(x, emd["emg_SSLW"], x, emd["emg_SSSA"], x, emd["emg_SSSD"])
# plt.legend(['emg_SSLW', 'emg_SSSA', 'emg_SSSD'])
# plt.title("SS")




## input emg feature data
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which feature set to use
emg_feature = Feature_Storage.readEmgFeatures(subject, version, feature_set)
# if you need to use CNN model, you need to reshape the data
emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_feature)
# reduce the sample number of certain modes to balance the dataset
emg_feature_sample_reduced = copy.deepcopy(emg_feature)
emg_feature_sample_reduced['emg_LW_features'] = emg_feature_sample_reduced['emg_LW_features'][
int(emg_feature['emg_LW_features'].shape[0] / 4): int(emg_feature['emg_LW_features'].shape[0] * 3 / 4), :]


