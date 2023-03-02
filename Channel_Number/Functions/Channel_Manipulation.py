import numpy as np
import copy
from skimage import restoration
from scipy.ndimage import median_filter


## select part of the channels to train the model for performance comparison
def selectSomeChannels(emg_preprocessed, channel_dataset):
    if len(channel_dataset) == 65 or len(channel_dataset) == 1:  # channel from one muscle
        channel_to_process = channel_dataset
    else:   # channel from two muscles
        channel_to_process = channel_dataset + [x + 65 for x in channel_dataset]
    emg_channel_selected = copy.deepcopy(emg_preprocessed)

    for transition_type, transition_value in emg_preprocessed.items():
        for number, value in enumerate(transition_value):
            if len(channel_to_process) == 4:  # select bipolar EMG from two muscles
                emg_channel_selected[transition_type][number] = value[:, [channel_to_process[0], channel_to_process[2]]] - \
                                                                value[:, [channel_to_process[1], channel_to_process[3]]]
            elif len(channel_to_process) == 1:  # select bipolar EMG from one muscle
                emg_channel_selected[transition_type][number] = value[:, channel_to_process[0]] - value[:, channel_to_process[0]+2]
            else:  # select monopolar EMG
                emg_channel_selected[transition_type][number] = value[:, channel_to_process]

    return emg_channel_selected


##  simulate the lost of some channels in the testset and convert the emg data into 2d images
def losingSomeTestChannels(cross_validation_groups, lost_channels):
    channel_lost = lost_channels + [x + 65 for x in lost_channels]
    emg_channel_lost = copy.deepcopy(cross_validation_groups)
    for group_number, group_value in emg_channel_lost.items():
        for set_type, set_value in group_value.items():
            for transition_type, transition_value in set_value.items():
                for number, value in enumerate(transition_value):
                    if set_type == 'test_set':
                        value[:, channel_lost] = 0
                        transition_value[number] = np.reshape(value, (value.shape[0], 13, 5, -1), 'F').astype(np.float32)
                    elif set_type == 'train_set':
                        transition_value[number] = np.reshape(value, (value.shape[0], 13, 5, -1), 'F').astype(np.float32)
    return emg_channel_lost


## recover the lost channels (inpaint + median filtering)
def inpaintImages(channel_lost_dataset, is_median_filtering=True):
    # using inpaint method to recover the image
    emg_inpainted = copy.deepcopy(channel_lost_dataset)
    for group_number, group_value in emg_inpainted.items():
        for set_type, set_value in group_value.items():
            for transition_type, transition_value in set_value.items():
                for number, value in enumerate(transition_value):
                    try:  # if it runs error, try another set of losing channels
                        # create a mask array
                        masks = value == 0  # create a boolean mask that is True for zero elements and False for non-zero elements
                        mask = masks.astype(int)[0, :, :, 0]  # convert false to 0, true to 1 and only keep the row and column of the array
                        # convert sample dimension to channel dimension so that all samples in this loop can be processed at once
                        images_transposed = np.reshape(np.transpose(value, (1, 2, 3, 0)), (value.shape[1], value.shape[2], -1), 'F')
                        image_inpainted = restoration.inpaint.inpaint_biharmonic(images_transposed, mask=mask, channel_axis=-1)  # inpaint the emg image
                        # median filtering
                        if is_median_filtering == True:
                            image_inpainted = median_filter(image_inpainted, size=3)
                        # convert the emg data to the original shape
                        transition_value[number] = np.transpose(
                            np.reshape(image_inpainted, (value.shape[1], value.shape[2], value.shape[3], -1), 'F'), (3, 0, 1, 2))
                        print(group_number, set_type, transition_type, number)
                    except Exception as e:
                        print(group_number, set_type, transition_type, number, e)

    return emg_inpainted


##  convert 2d emg shape back to previous 1d emg shape, and change the electrode order to align with the physical grid
def restoreEmgshape(emg_inpainted, emg_channel_lost):
    # restore inpainted emg shape
    emg_recovered = copy.deepcopy(emg_inpainted)
    for group_number, group_value in emg_recovered.items():
        for set_type, set_value in group_value.items():
            for transition_type, transition_value in set_value.items():
                for number, value in enumerate(transition_value):
                    reshaped_data = np.reshape(value, (value.shape[0], -1), 'F')  # back to 1d shape
                    reshaped_data[:, 13:26] = np.flip(reshaped_data[:, 13:26], axis=1)
                    reshaped_data[:, 39:52] = np.flip(reshaped_data[:, 39:52], axis=1)
                    reshaped_data[:, 13+65:26+65] = np.flip(reshaped_data[:, 13+65:26+65], axis=1)
                    reshaped_data[:, 39+65:52+65] = np.flip(reshaped_data[:, 39+65:52+65], axis=1)
                    transition_value[number] = reshaped_data
    # restore channel lost emg shape
    emg_unrecovered = copy.deepcopy(emg_channel_lost)
    for group_number, group_value in emg_unrecovered.items():
        for set_type, set_value in group_value.items():
            for transition_type, transition_value in set_value.items():
                for number, value in enumerate(transition_value):
                    reshaped_data = np.reshape(value, (value.shape[0], -1), 'F')  # back to 1d shape
                    reshaped_data[:, 13:26] = np.flip(reshaped_data[:, 13:26], axis=1)
                    reshaped_data[:, 39:52] = np.flip(reshaped_data[:, 39:52], axis=1)
                    reshaped_data[:, 13+65:26+65] = np.flip(reshaped_data[:, 13+65:26+65], axis=1)
                    reshaped_data[:, 39+65:52+65] = np.flip(reshaped_data[:, 39+65:52+65], axis=1)
                    if set_type == 'train_set':
                        reshaped_data = compensateZeroColumns(reshaped_data)
                    transition_value[number] = reshaped_data
    return emg_recovered, emg_unrecovered


## compensate the training set's zero columns using the mean value of nearby columns
def compensateZeroColumns(arr):
    # Get the indices of columns with all zeros
    zero_cols = np.where(np.all(arr == 0, axis=0))[0]

    # Loop through each zero column
    for col_idx in zero_cols:
        # Check if the column is at the first position
        if col_idx == 0 or col_idx == 65:
            # Calculate the average of the next column
            avg_val = arr[:, col_idx + 1]
        # Check if the column is at the last position
        elif col_idx == 64 or col_idx == arr.shape[1] - 1:
            # Calculate the average of the previous column
            avg_val = arr[:, col_idx - 1]
        else:
            # Calculate the average of the neighboring columns
            avg_val = np.mean(arr[:, [col_idx - 1, col_idx + 1]], axis=1)

        # Replace the values in the zero column with the calculated average value
        arr[:, col_idx] = avg_val
    return arr

