'''
plot the filtered emg data from each channel at each gait event
'''


## modules
from Transition_Prediction.Pre_Processing.Utility_Functions import Data_Reshaping, Feature_Storage
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Transition_Prediction.Pre_Processing import Preprocessing
from Conditional_GAN.Data_Procesing import Plot_Emg_Data
import numpy as np
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


## For demonstration purposes, we'll use data from result_dict_part1 for 'key_1'
old_emg_envelope_1, old_emg_envelope_2 = Plot_Emg_Data.averageChannelValues(old_emg_envelope)
new_emg_envelope_1, new_emg_envelope_2 = Plot_Emg_Data.averageChannelValues(new_emg_envelope)
# Plot using a single line of code
Plot_Emg_Data.plotAverageChannel(old_emg_envelope_1, "emg_LWLW", title='old_emg_LWLW', ylim=(0, 1000))
Plot_Emg_Data.plotAverageChannel(old_emg_envelope_1, "emg_SASA", title='old_emg_SASA', ylim=(0, 1000))
Plot_Emg_Data.plotAverageChannel(old_emg_envelope_1, "emg_LWSA", title='old_emg_LWSA', ylim=(0, 1000))
Plot_Emg_Data.plotAverageChannel(new_emg_envelope_1, "emg_LWLW", title='new_emg_LWLW', ylim=(0, 1000))
Plot_Emg_Data.plotAverageChannel(new_emg_envelope_1, "emg_SASA", title='new_emg_SASA', ylim=(0, 1000))
Plot_Emg_Data.plotAverageChannel(new_emg_envelope_1, "emg_LWSA", title='new_emg_LWSA', ylim=(0, 1000))


## plot event average value in a single plot
# List of datasets and titles
datasets1 = [(pd.DataFrame(old_emg_envelope_1["emg_LWLW"])).mean(axis=1), (pd.DataFrame(old_emg_envelope_1["emg_SDSD"])).mean(axis=1),
    (pd.DataFrame(old_emg_envelope_1["emg_LWSD"])).mean(axis=1)]
datasets2 = [(pd.DataFrame(new_emg_envelope_1["emg_LWLW"])).mean(axis=1), (pd.DataFrame(new_emg_envelope_1["emg_SDSD"])).mean(axis=1),
    (pd.DataFrame(new_emg_envelope_1["emg_LWSD"])).mean(axis=1)]
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
emg_1_channel_list, emg_2_channel_list, emg_1_event_mean, emg_2_event_mean = Plot_Emg_Data.averageRepetitionAndEventValues(old_emg_envelope)
left_number = 000
right_number = 1800
emg_1_value = copy.deepcopy(emg_1_event_mean)
for event, value in emg_1_event_mean.items():
    emg_1_value[event] = value[left_number:right_number]
emg_2_value = copy.deepcopy(emg_2_event_mean)
for event, value in emg_2_event_mean.items():
    emg_2_value[event] = value[left_number:right_number]
pd.DataFrame(emg_1_value).plot(subplots=True, layout=(4, 4), title="EMG 1")
pd.DataFrame(emg_2_value).plot(subplots=True, layout=(4, 4), title="EMG 2")


##  plot summed series emg data by group
emd_data = emg_2_value
x = list(range(left_number-1000, right_number-1000))
plt.figure()
plt.plot(x, emd_data["emg_LWLW_data"], x, emd_data["emg_LWSA_data"], x, emd_data["emg_SASA_data"], x, emd_data["emg_SALW_data"])
plt.legend(['emg_LWLW_data', 'emg_LWSA_data', 'emg_SASA_data', 'emg_SALW_data'])
plt.title("LWSA")
plt.figure()
plt.plot(x, emd_data["emg_LWLW_data"], x, emd_data["emg_LWSD_data"], x, emd_data["emg_SDSD_data"], x, emd_data["emg_SDLW_data"])
plt.legend(['emg_LWLW_data', 'emg_LWSD_data', 'emg_SDSD_data', 'emg_SDLW_data'])
plt.title("LWSD")


##  plot summed series emg data by group
emd_data = emg_2_value
x = list(range(left_number-1000, right_number-1000))
plt.figure()
plt.plot(x, emd_data["emg_LWLW_data"], x, emd_data["emg_LWSA_data"], x, emd_data["emg_LWSD_data"], x, emd_data["emg_LWSS_data"])
plt.legend(['emg_LWLW_data', 'emg_LWSA_data', 'emg_LWSD_data', 'emg_LWSS_data'])
plt.title("LW")
# plt.figure()
# plt.plot(x, emd_data["emg_SASA_data"], x, emd_data["emg_SALW_data"], x, emd_data["emg_SASS_data"])
# plt.legend(['emg_SASA_data', 'emg_SALW_data', 'emg_SASS_data'])
# plt.title("SA")
# plt.figure()
# plt.plot(x, emd_data["emg_SDSD_data"], x, emd_data["emg_SDLW_data"], x, emd_data["emg_SDSS_data"])
# plt.legend(['emg_SDSD_data', 'emg_SDLW_data', 'emg_SDSS_data'])
# plt.title("SD")
# plt.figure()
# plt.plot(x, emd_data["emg_SSLW_data"], x, emd_data["emg_SSSA_data"], x, emd_data["emg_SSSD_data"])
# plt.legend(['emg_SSLW_data', 'emg_SSSA_data', 'emg_SSSD_data'])
# plt.title("SS")





## organize channel summed emg data
emd_data = emg_1_channel_list
start_index = 0  # note: the first image is the mean value for the entire dataset of the gait event (emg_1_mean_events)
end_index = 30
horizontal = 6
vertical = 5

(pd.DataFrame(emd_data["emg_LWLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWLW")
(pd.DataFrame(emd_data["emg_LWSA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWSA")
(pd.DataFrame(emd_data["emg_SASA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SASA")
(pd.DataFrame(emd_data["emg_LWLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWLW")
(pd.DataFrame(emd_data["emg_LWSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWSD")
(pd.DataFrame(emd_data["emg_SDSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDSD")


# ## transit from LW
# (pd.DataFrame(emd_data["emg_LWLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWLW")
# (pd.DataFrame(emd_data["emg_LWSA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWSA")
# (pd.DataFrame(emd_data["emg_LWSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWSD")
# (pd.DataFrame(emd_data["emg_LWSS_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWSS")
# ## transit from SA
# (pd.DataFrame(emd_data["emg_SASA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SASA")
# (pd.DataFrame(emd_data["emg_SALW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SALW")
# (pd.DataFrame(emd_data["emg_SASS_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SASS")
# ## transit from SD
# (pd.DataFrame(emd_data["emg_SDSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDSD")
# (pd.DataFrame(emd_data["emg_SDLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDLW")
# (pd.DataFrame(emd_data["emg_SDSS_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDSS")
# ## transit from SS
# (pd.DataFrame(emd_data["emg_SSLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SSLW")
# (pd.DataFrame(emd_data["emg_SSSA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SSSA")
# (pd.DataFrame(emd_data["emg_SSSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SSSD")
# ## no transition
# (pd.DataFrame(emd_data["emg_LWLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWLW")
# (pd.DataFrame(emd_data["emg_SASA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SASA")
# (pd.DataFrame(emd_data["emg_SDSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDSD")





## input emg feature data
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which feature set to use
emg_feature_data = Feature_Storage.readEmgFeatures(subject, version, feature_set)
# if you need to use CNN model, you need to reshape the data
emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_feature_data)
# reduce the sample number of certain modes to balance the dataset
emg_feature_sample_reduced = copy.deepcopy(emg_feature_data)
emg_feature_sample_reduced['emg_LW_features'] = emg_feature_sample_reduced['emg_LW_features'][
int(emg_feature_data['emg_LW_features'].shape[0] / 4): int(emg_feature_data['emg_LW_features'].shape[0] * 3 / 4), :]





