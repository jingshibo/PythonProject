import numpy as np
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data, Dtw_Similarity
import pandas as pd
from scipy import ndimage
from scipy import signal
import cv2

## slice a piece of emg data from a given time period
def sliceTimePeriod(emg_data, start, end, toeoff):
    '''

    :param emg_data:
    :param start:
    :param end:
    :param toeoff: the toeoff
    :return:
    '''
    sliced_emg_data = {}
    # Loop through each key-value pair in the original dictionary
    for key, array_list in emg_data.items():
        # Loop through each array in the list and select only the first 65 columns
        sliced_emg_data[key] = [np.copy(arr[start:end, :]) for arr in array_list]  # the slices are views into the original data,
        # not copies.
    window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=toeoff - start, endtime_after_toeoff_ms=end - toeoff,
        feature_window_ms=toeoff - start, predict_window_ms=toeoff - start)

    return sliced_emg_data, window_parameters


## split two emg grids into two
def separateEmgGrids(emg_dict, separate=False):
    # Initialize a list to store dictionaries for each grid
    emg_grids = {}
    if separate is False:  # treat all grids as a whole and thus return the original data
        emg_grids['grid_1'] = [emg_dict]
    elif separate is True:
        num_columns = emg_dict[next(iter(emg_dict))][0].shape[1]  # next(iter(emg_dict)) is used to get the first key in the dict
        num_grids = num_columns // 65  # Calculate the number of grids based on the number of columns
        # Initialize dictionaries for each electrode grid
        for grid_idx in range(num_grids):
            emg_grids[f'grid_{grid_idx+1}'] = {}
        # Loop through each key-value pair in the original dictionary
        for key, array_list in emg_dict.items():
            # Loop through each array in the list
            for arr in array_list:
                for grid_idx in range(num_grids):
                    start_col = grid_idx * 65
                    end_col = (grid_idx + 1) * 65
                    # Extract the columns corresponding to the current grid
                    sub_arr = arr[:, start_col:end_col]
                    # Add the sub-array to the corresponding dictionary
                    if key not in emg_grids[f'grid_{grid_idx+1}']:
                        emg_grids[f'grid_{grid_idx+1}'][key] = []
                    emg_grids[f'grid_{grid_idx+1}'][key].append(sub_arr)
    return emg_grids


## median filtering each emg grid separately and then combine them into one again
def spatialFilterModelInput(emg_data, kernel):
    filtered_emg = {}
    # Loop through each key-value pair in the original dictionary
    for key, array_list in emg_data.items():
        filtered_array_list = []
        for arr in array_list:
            num_columns = arr.shape[1]
            num_grids = num_columns // 65  # Assumes that each grid has 65 columns

            filtered_arr_parts = []
            for i in range(num_grids):  # automatically adapt to different num of grids
                start_col = i * 65
                end_col = (i + 1) * 65
                sub_arr = arr[:, start_col:end_col]
                # filtered_sub_arr = pd.DataFrame(ndimage.median_filter(sub_arr, mode='nearest', size=size)).to_numpy()
                filtered_sub_arr = pd.DataFrame(ndimage.gaussian_filter(sub_arr, mode='reflect', sigma=kernel, axes=[0, 1])).to_numpy()
                filtered_arr_parts.append(filtered_sub_arr)
            # Concatenate along columns to form the final filtered array
            filtered_arr = np.concatenate(filtered_arr_parts, axis=1)
            filtered_array_list.append(filtered_arr)

        # Add the new list of filtered arrays to the new dictionary
        filtered_emg[key] = filtered_array_list
    return filtered_emg

