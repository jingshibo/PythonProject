##
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing
from Transition_Prediction.Pre_Processing.Utility_Functions import Feature_Calculation
import matplotlib.pyplot as plt


'''calculate emg features'''
## extract data
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
modes = {'standing': 'SS', 'level': 'LW', 'upstairs': 'SA', 'downstairs': 'SD', 'upslope': 'RA', 'downslope': 'RD'}

for subject in subjects:
    emg_preprocessed = Emg_Preprocessing.preprocessEmgData(subject, modes)


    ## reorganize by sliding windows
    window_size = 512  # 256ms
    window_increment = 64  # 32ms
    emg_windowed = {}
    for mode, name in modes.items():
        emg_windowed[name] = Emg_Preprocessing.createWindows(emg_preprocessed[name], window_size, window_increment)


    ## calculate features for each window data
    emg_features = {}
    for mode, name in modes.items():
        emg_feature_list = []
        for emg_window_data in emg_windowed[name]:
            emg_feature = Feature_Calculation.calcuEmgFeatures(emg_window_data)
            emg_feature_list.append(emg_feature)
        emg_features[name] = np.vstack(emg_feature_list).tolist()  # convert numpy to list for dict storage


    ## save features
    feature_set = 'hdsemg'  # there may be multiple sets of features to be calculated for comparison
    Emg_Preprocessing.saveFeatures(subject, emg_features, feature_set)


    ''' interpolate emg features '''
    ## read emg features
    emg_features = Emg_Preprocessing.readFeatures(subject, 'hdsemg')
    emg_feature_array = {key: np.array(value) for key, value in emg_features.items()}

    ## reshape emg features
    emg_feature_map = {}
    for locomotion_mode, locomotion_value in emg_feature_array.items():
        emg_feature_map[locomotion_mode] = np.reshape(locomotion_value, (locomotion_value.shape[0], 13, 5, -1), 'F').astype(np.float32)

    ## interpolate emg features
    emg_feature_interp = {}
    for locomotion_mode, locomotion_value in emg_feature_map.items():
        emg_feature_interp[locomotion_mode] = zoom(locomotion_value, (1, 97 / 13, 33 / 5, 1), order=3)

    ## save interpolated features
    Emg_Preprocessing.saveInterpFeatures(subject, emg_feature_interp, feature_set)


## plot a single heatmap for interpolated emg features
subject = 'Number1'
emg_features_interp = Emg_Preprocessing.readInterpFeatures(subject, 'hdsemg')

selected_number = 500  # e.g., the first image
selected_channel = 7  # e.g., the first channel
interp_matrix = emg_features_interp['LW'][selected_number, :, :, selected_channel]

# Create the plot with increased figure size
plt.figure(figsize=(5, 13))
size = 30
# Display the heatmap
plt.imshow(interp_matrix, cmap='viridis', aspect='auto') # 'viridis' is a popular colormap
# Add a colorbar and set its label size
cbar = plt.colorbar(label='Intensity')
cbar.ax.tick_params(labelsize=size)  # Increase colorbar tick label size
# Set the title and axis labels with increased font sizes
# plt.title(f'Heatmap for Number {selected_number}, Channel {selected_channel}', fontsize=18)
# plt.xlabel('Width', fontsize=size)
# plt.ylabel('Length', fontsize=size)
# Set x-axis ticks at intervals of 10
x_ticks = np.arange(0, interp_matrix.shape[1], 10)  # Change interp_matrix.shape[1] to the number of columns in the matrix
plt.xticks(x_ticks, fontsize=size)
# Increase the x and y tick label sizes
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
# Show the plot
plt.show()