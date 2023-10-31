import numpy as np
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data, Dtw_Similarity
import pandas as pd
from scipy import ndimage
from scipy import signal
import cv2
import copy

## slice a piece of emg data from a given time period
def sliceTimePeriod(emg_data, start, end, toeoff, sliding_results=True):
    '''
    :param start: the start index relative to the start of the original extracted data
    :param end: the end index relative to the start of the original extracted data
    :param toeoff: the time point of toe-off relative to the start of the original extracted data (not relative to the start parameter here)
    '''
    sliced_emg_data = {}
    # Loop through each key-value pair in the original dictionary
    for key, array_list in emg_data.items():
        # Loop through each array in the list and select only the first 65 columns
        sliced_emg_data[key] = [np.copy(arr[start:end, :]) for arr in array_list]  # the slices are views into the original data,
        # not copies.
    if sliding_results:  # calculate prediction results at different delay points
        window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=toeoff - start, endtime_after_toeoff_ms=end - toeoff,
            feature_window_ms=toeoff - start, predict_window_ms=toeoff - start)
    else:
        window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=toeoff - start,
            endtime_after_toeoff_ms=end - toeoff, feature_window_ms=end - start, predict_window_ms=end - start)
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


## separate the emg_data into individual grids first, and then spatial filter each emg grid separately. finally combine them into one again
def spatialFilterModelInput(emg_data, kernel):
    """
    Spatially filters the given EMG data using Gaussian filtering.
    Parameters: - emg_data: Dictionary containing EMG data.
    - kernel: Kernel size for Gaussian filtering.
    Returns:- Filtered EMG data in the same structure as the input.
    """
    def split_into_grids(emg_data):
        """
        Splits the given EMG data into smaller grids.
        Parameters:- emg_data: Dictionary containing EMG data.
        Returns: Dictionary with split EMG data grids.
        """
        grid_data = {}
        for key, array_list in emg_data.items():
            grid_data[key] = {}
            for arr in array_list:
                num_columns = arr.shape[1]
                num_grids = num_columns // 65  # Assumes that each grid has 65 columns
                for i in range(num_grids):  # automatically adapt to different num of grids
                    start_col = i * 65
                    end_col = (i + 1) * 65
                    sub_arr = arr[:, start_col:end_col]
                    # Check if this grid already exists for the current key
                    if f"grid{i}" not in grid_data[key]:
                        grid_data[key][f"grid{i}"] = []
                    grid_data[key][f"grid{i}"].append(sub_arr)
        return grid_data

    def spatial_filter_grids(grid_data, kernel=(1, 1), axes=(1, 2), radius=None):
        """
        Applies Gaussian spatial filtering on each sub-array in the grid lists.
        Parameters:
        - grid_data: Dictionary with split EMG data grids.
        - kernel: Kernel size for Gaussian filtering.
        - axes: Axes on which the filtering will be applied.
        - radius: Optional radius for Gaussian filtering.
        Returns: - Dictionary with spatially filtered grid data.
        """
        filtered_grid_data = copy.deepcopy(grid_data)
        for key, grids in grid_data.items():
            for grid_key, sub_arr_list in grids.items():
                # Stack together all sub-arrays in the current grid list
                combined_sub_arrays = np.stack(sub_arr_list, axis=0)
                # Apply gaussian filter
                filtered_combined_sub_arrays = ndimage.gaussian_filter(combined_sub_arrays, sigma=kernel, mode='reflect', axes=axes,
                    radius=radius)
                # filtered_combined_sub_arrays = cv2.blur(combined_sub_arrays.transpose(1,2,0), kernel).transpose(2,0,1)
                # Splitting the combined array back into a list of sub-arrays
                split_sub_arrays = [np.squeeze(sub_arr, axis=0) for sub_arr in
                    np.array_split(filtered_combined_sub_arrays, len(sub_arr_list), axis=0)]
                # Update the filtered_data dictionary
                filtered_grid_data[key][grid_key] = split_sub_arrays
        return filtered_grid_data

    def combine_filtered_grids(filtered_grid_data):
        """
        Combines the spatially filtered grids to obtain the original EMG data structure.
        Parameters:- filtered_grid_data: Dictionary with spatially filtered grid data.
        Returns: - Dictionary with combined filtered data.
        """
        combined_filtered_data = {}
        for key, grids in filtered_grid_data.items():
            array_list = []
            # Assuming each key value in the 'grids' dict has the same length
            for i in range(len(next(iter(grids.values())))):
                combined_array_parts = []
                for grid_key, sub_arr_list in grids.items():
                    combined_array_parts.append(sub_arr_list[i])
                # Concatenate along columns to form the final combined array
                combined_array = np.concatenate(combined_array_parts, axis=1)
                array_list.append(combined_array)
            combined_filtered_data[key] = array_list
        return combined_filtered_data

    # Split the emg_data into grids
    grid_data = split_into_grids(emg_data)
    # filter each sub-array in the grid lists
    filtered_grid_data = spatial_filter_grids(grid_data, kernel=kernel)
    # Combine the filtered grids to get back the original structure
    recombined_filter_data = combine_filtered_grids(filtered_grid_data)
    return recombined_filter_data


## spatial filtering each emg grid separately and then combine them into one again
def spatialFilterModelInputOldVersion(emg_data, kernel):
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
                # filtered_sub_arr = ndimage.median_filter(sub_arr, mode='nearest', size=size)
                filtered_sub_arr = ndimage.gaussian_filter(sub_arr, mode='reflect', sigma=kernel, axes=[0, 1])
                # filtered_sub_arr = cv2.blur(sub_arr, kernel)
                filtered_arr_parts.append(filtered_sub_arr)
            # Concatenate along columns to form the final filtered array
            filtered_arr = np.concatenate(filtered_arr_parts, axis=1)
            filtered_array_list.append(filtered_arr)

        # Add the new list of filtered arrays to the new dictionary
        filtered_emg[key] = filtered_array_list
    return filtered_emg

