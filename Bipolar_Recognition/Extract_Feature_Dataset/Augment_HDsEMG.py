##
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing, Manipulate_Channels
import numpy as np


## load interpolated emg features
subject = 'Number1'
feature_set = 'HDsEMG'
interp_emg_features = Emg_Preprocessing.readInterpFeatures(subject, feature_set)
# keep only the central image parts
vertical_channels = slice(8, 89)
horizontal_channels = slice(0, 25)


## shift and clip emg feature images
clip_emg_h = {}  # horizontal shift
for locomotion_mode, locomotion_values in interp_emg_features.items():
    # horizontal shift (to left)
    shift_emg_h = np.array([Manipulate_Channels.shiftEmgImage(image, max_shift=8, direction='left') for image in locomotion_values])
    # clip only the central part
    clip_emg_h[locomotion_mode] = shift_emg_h[:, vertical_channels, horizontal_channels, :]

clip_emg_v = {}  # vertical shift
for locomotion_mode, locomotion_values in interp_emg_features.items():
    # vertical shift (to up)
    shift_emg_v = np.array([Manipulate_Channels.shiftEmgImage(image, max_shift=8, direction='up') for image in locomotion_values])
    # clip only the central part
    clip_emg_v[locomotion_mode] = shift_emg_v[:, vertical_channels, horizontal_channels, :]

clip_emg_o = {}  # original image (no shift, only truncate)
for locomotion_mode, locomotion_values in interp_emg_features.items():
    clip_emg_o[locomotion_mode] = locomotion_values[:, vertical_channels, horizontal_channels, :]


##
clip_emg_h = Manipulate_Channels.multipleShifts(interp_emg_features, direction='left', max_shift=8, repeat_number=3,
    vertical_channels=slice(8, 89), horizontal_channels=slice(0, 25))



## plot a single emg image
import matplotlib.pyplot as plt

selected_number = 1  # e.g., the first image
selected_channel = 1  # e.g., the first channel
matrix = emg_cross_validation_shift['shift_1']['group_0']['train_set']['SS'][selected_number][:, :, selected_channel]

plt.figure(figsize=(5, 13))
plt.imshow(matrix, cmap='viridis', aspect='auto')  # 'viridis' is a popular colormap
plt.colorbar(label='Intensity')
plt.title(f'Heatmap for Number {selected_number}, Channel {selected_channel}')
plt.xlabel('Width')
plt.ylabel('Length')
plt.show()

matrix = emg_cross_validation_original['group_0']['train_set']['SS'][selected_number][:, :, selected_channel]

plt.figure(figsize=(5, 13))
plt.imshow(matrix, cmap='viridis', aspect='auto')  # 'viridis' is a popular colormap
plt.colorbar(label='Intensity')
plt.title(f'Heatmap for Number {selected_number}, Channel {selected_channel}')
plt.xlabel('Width')
plt.ylabel('Length')
plt.show()