import numpy as np
import copy
from skimage.restoration import inpaint_biharmonic
from scipy.ndimage import median_filter


## select part of the channels to train the model for performance comparison
def selectSomeChannels(emg_preprocessed, channel_dataset):
    if len(channel_dataset) == 65:  # select one muscle
        channel_to_process = channel_dataset
    else:
        channel_to_process = channel_dataset + [x + 65 for x in channel_dataset]
    emg_channel_selected = copy.deepcopy(emg_preprocessed)
    for transition_type, transition_value in emg_preprocessed.items():
        for number, value in enumerate(transition_value):
            if len(channel_to_process) == 4:  # select a bipolar
                emg_channel_selected[transition_type][number] = value[:, [channel_to_process[0], channel_to_process[2]]] - \
                                                                value[:, [channel_to_process[1], channel_to_process[3]]]
            else:
                emg_channel_selected[transition_type][number] = value[:, channel_to_process]

    return emg_channel_selected


##  simulate the lost of some channels in the testset and convert the emg data into 2d images
def losingSomeTestChannels(cross_validation_groups, lost_channels):
    channel_lost = lost_channels + [x + 65 for x in lost_channels]
    channel_lost_testset = copy.deepcopy(cross_validation_groups)
    for group_number, group_value in channel_lost_testset.items():
        for set_type, set_value in group_value.items():
            for transition_type, transition_value in set_value.items():
                for number, value in enumerate(transition_value):
                    if set_type == 'test_set':
                        value[:, channel_lost] = 0
                        transition_value[number] = np.reshape(value, (value.shape[0], 13, 5, -1), 'F')
                    elif set_type == 'train_set':
                        transition_value[number] = np.reshape(value, (value.shape[0], 13, 5, -1), 'F')
    return channel_lost_testset


## recover the lost channels (inpaint + median filtering)
def inpaintImages(channel_lost_dataset):
    images_recovered = copy.deepcopy(channel_lost_dataset)
    for group_number, group_value in images_recovered.items():
        for set_type, set_value in group_value.items():
            for transition_type, transition_value in set_value.items():
                for number, value in enumerate(transition_value):
                    # create a mask array
                    masks = value == 0  # create a boolean mask that is True for zero elements and False for non-zero elements
                    mask = masks.astype(int)[0, :, :, 0]  # convert false to 0, true to 1 and only keep the row and column of the array
                    # convert sample dimension to channel dimension to increase the processing speed
                    images_transposed = np.reshape(np.transpose(value, (1, 2, 3, 0)), (value.shape[1], value.shape[2], -1), 'F')
                    image_inpainted = inpaint_biharmonic(images_transposed, mask=mask, channel_axis=-1)  # inpaint the emg image
                    image_median_filtered = median_filter(image_inpainted, size=3)  # median filtering
                    transition_value[number] = np.transpose(  # convert the emg data to the original shape
                        np.reshape(image_median_filtered, (value.shape[1], value.shape[2], value.shape[3], -1), 'F'), (3, 0, 1, 2))

                    print(group_number, set_type, transition_type, number)

    return images_recovered
