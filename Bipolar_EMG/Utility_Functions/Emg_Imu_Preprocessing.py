##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import os
import datetime
import csv
import json


## align the begin of imu and emg data using row index
def alignEmgImuBeginIndex(emg_start_index, imu_start_index, emg_data, imu_data):
    # only reserve data after the start index
    emg_cropped_begin = emg_data.iloc[emg_start_index:, :].reset_index(drop=True)
    imu_cropped_begin = imu_data.iloc[imu_start_index:, :].reset_index(drop=True)

    # add the sample number as a column for easier comparison
    emg_cropped_begin.insert(loc=0, column='number', value=range(len(emg_cropped_begin)))
    imu_cropped_begin.insert(loc=0, column='number', value=range(len(imu_cropped_begin)))
    return emg_cropped_begin, imu_cropped_begin


## upsamling imu data to match emg
def upsampleImuEqualToEmg(imu_data, emg_data):
    x = np.arange(len(imu_data))
    y = imu_data.iloc[:, 2:-1].to_numpy()  # only extract measurement value columns
    f = PchipInterpolator(x, y)
    x_upsampled = np.linspace(min(x), max(x), len(emg_data))
    y_upsampled = f(x_upsampled)
    imu_upsampled = pd.DataFrame(y_upsampled)
    return imu_upsampled


## plot the pulses of imu and emg for alignment
def plotEmgImu(emg_data, imu_data, start_index, end_index):
    emg_sync = emg_data.loc[:, 7]  # extract total force column from aligned emg data
    imu_sync = imu_data.loc[:, 2]

    # plot
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(range(len(emg_sync.iloc[start_index:end_index])), emg_sync.iloc[start_index:end_index], label="emg_sync")
    axes[1].plot(range(len(imu_sync.iloc[start_index:end_index])), imu_sync.iloc[start_index:end_index], label="imu_sync")

    axes[0].set(title="emg_sync", ylabel="pulse")
    axes[1].set(title="imu_sync", ylabel="pulse")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")


## plot the pulses of imu and emg After alignment
def plotEmgImuAligned(emg_data, imu_data, start_index, end_index):
    emg_sync = emg_data.loc[:, 7]  # extract total force column from aligned emg data
    imu_sync = imu_data.loc[:, 0]

    # plot
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].plot(range(len(emg_sync.iloc[start_index:end_index])), emg_sync.iloc[start_index:end_index], label="emg_sync")
    axes[1].plot(range(len(imu_sync.iloc[start_index:end_index])), imu_sync.iloc[start_index:end_index], label="imu_sync")

    axes[0].set(title="emg_sync", ylabel="pulse")
    axes[1].set(title="imu_sync", ylabel="pulse")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")


## save the alignment parameters into a csv file
def saveAlignParameters(subject, mode, emg_start_index, imu_start_index, emg_end_index, imu_end_index, project='Bipolar_Data'):
    # save file path
    data_dir = f'D:\Data\{project}\subject_{subject}'
    alignment_file = f'subject_{subject}_align_parameters.csv'
    alignment_file_path = os.path.join(data_dir, alignment_file)

    data_file_name = f'{subject}_{mode}'
    alignment_save_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # alignment parameters to save
    columns = ['data_file_name', 'alignment_save_date', 'emg_start_index', 'imu_start_index', 'emg_end_index', 'imu_end_index']
    save_parameters = [data_file_name, alignment_save_date, emg_start_index, imu_start_index, emg_end_index, imu_end_index]

    with open(alignment_file_path, 'a+') as file:
        if os.stat(alignment_file_path).st_size == 0:  # if the file is new created
            print("Creating File.")
            write = csv.writer(file)
            write.writerow(columns)  # write the column fields
            write.writerow(save_parameters)
        else:
            write = csv.writer(file)
            write.writerow(save_parameters)


## read the alignment parameters from a csv file
def readAlignParameters(subject, mode, project='Bipolar_Data'):
    data_dir = f'D:\Data\{project}\subject_{subject}'
    alignment_file = f'subject_{subject}_align_parameters.csv'
    alignment_file_path = os.path.join(data_dir, alignment_file)

    data_file_name = f'{subject}_{mode}'
    alignment_data = pd.read_csv(alignment_file_path, sep=',')  # header exists
    file_parameter = alignment_data.query('data_file_name == @data_file_name')  # use @ to cite variable values
    if file_parameter.empty:  # if no alignment parameter found
        raise Exception(f"No alignment parameter found for data file: {data_file_name}")
    else:
        align_parameter = file_parameter.iloc[[-1]]  # extract the last row (apply the newest parameters)

        emg_start_index = align_parameter['emg_start_index'].iloc[0]
        imu_start_index = align_parameter['imu_start_index'].iloc[0]
        emg_end_index = align_parameter['emg_end_index'].iloc[0]
        imu_end_index = align_parameter['imu_end_index'].iloc[0]

        return emg_start_index, imu_start_index, emg_end_index, imu_end_index


## save all sensor data after alignment into a csc file
def saveAlignedData(subject, mode, imu_aligned, emg_aligned, project='Bipolar_Data'):
    data_dir = f'D:\Data\\{project}\subject_{subject}\\aligned_data'
    data_file_name = f'subject_{subject}_{mode}'

    imu_file = f'IMU\\imu_{data_file_name}_aligned.csv'
    emg_file = f'bipolarEMG\\emg_{data_file_name}_aligned.csv'

    imu_path = os.path.join(data_dir, imu_file)
    emg_path = os.path.join(data_dir, emg_file)

    imu_aligned.to_csv(imu_path, index=False)
    emg_aligned.to_csv(emg_path, index=False)


## read all sensor data after alignment from a csc file
def readAlignedData(subject, mode, project='Bipolar_Data'):
    data_dir = f'D:\Data\\{project}\subject_{subject}\\aligned_data'
    data_file_name = f'subject_{subject}_{mode}'

    imu_file = f'IMU\\imu_{data_file_name}_aligned.csv'
    emg_file = f'bipolarEMG\\emg_{data_file_name}_aligned.csv'

    imu_path = os.path.join(data_dir, imu_file)
    emg_path = os.path.join(data_dir, emg_file)

    imu_aligned = pd.read_csv(imu_path)
    emg_aligned = pd.read_csv(emg_path)
    emg_aligned.columns = emg_aligned.columns.astype(int)

    return imu_aligned, emg_aligned


# calculate and plot MAV values of emg and imu data
def plotMeanAbsValue(emg_filtered, imu_filtered):
    # calculate MAV values of emg and imu data
    emg_mav = emg_filtered.abs().mean(axis=1)
    imu_mav = imu_filtered.abs().mean(axis=1)
    # Creating subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(emg_mav)  # Plotting the emg series on the upper subplot
    ax1.set_title('EMG')
    ax2.plot(imu_mav)  # Plotting the imu series on the bottom subplot
    ax2.set_title('IMU')


## save calculated emg and imu features
def saveFeatures(subject, emg_features, imu_features, feature_set):
    data_dir = f'D:\Data\Bipolar_Data\subject_{subject}\\features'

    # save emg features
    emg_feature_file = f'subject_{subject}_emg_feature_set_{feature_set}.json'
    emg_feature_path = os.path.join(data_dir, emg_feature_file)

    with open(emg_feature_path, 'w') as json_file:
        json.dump(emg_features, json_file, indent=8)

    # save imu features
    imu_feature_file = f'subject_{subject}_imu_feature_set_{feature_set}.json'
    imu_feature_path = os.path.join(data_dir, imu_feature_file)

    with open(imu_feature_path, 'w') as json_file:
        json.dump(imu_features, json_file, indent=8)


## read calculated emg and imu features
def readFeatures(subject, feature_set):
    data_dir = f'D:\Data\Bipolar_Data\subject_{subject}\\features'

    # load emg features
    emg_feature_file = f'subject_{subject}_emg_feature_set_{feature_set}.json'
    emg_feature_path = os.path.join(data_dir, emg_feature_file)

    with open(emg_feature_path) as json_file:
        emg_features = json.load(json_file)

    # load imu features
    imu_feature_file = f'subject_{subject}_imu_feature_set_{feature_set}.json'
    imu_feature_path = os.path.join(data_dir, imu_feature_file)

    # read json file
    with open(imu_feature_path) as json_file:
        imu_features = json.load(json_file)

    return emg_features, imu_features


# ## read calculated emg and imu features
# def readFeatures(subject, feature_set):
#     data_dir = f'D:\Data\Bipolar_Data\subject_{subject}\\features'
#
#     # load emg features
#     emg_feature_file = f'subject_{subject}_emg_feature_set_{feature_set}.json'
#     emg_feature_path = os.path.join(data_dir, emg_feature_file)
#     with open(emg_feature_path) as json_file:
#         emg_feature_list = json.load(json_file)
#     emg_features = {key: np.array(value) for key, value in emg_feature_list.items()}
#
#     # load imu features
#     imu_feature_file = f'subject_{subject}_imu_feature_set_{feature_set}.json'
#     imu_feature_path = os.path.join(data_dir, imu_feature_file)
#     with open(imu_feature_path) as json_file:
#         imu_feature_list = json.load(json_file)
#     imu_features = {key: np.array(value) for key, value in imu_feature_list.items()}
#
#     return emg_features, imu_features