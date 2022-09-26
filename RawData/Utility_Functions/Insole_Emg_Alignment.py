##
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import datetime

## plot insole and sync force data for alignment (only two insole force images share axes)
def plotInsoleSyncForce(recovered_emg_data, recovered_left_data, recovered_right_data, start_index, end_index):
    left_total_force = recovered_left_data.loc[:, 195]  # extract total force column
    right_total_force = recovered_right_data.loc[:, 195]
    sync_force = recovered_emg_data.iloc[:, -3]  # extract load cell column

    # plot
    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)  # only ax1 and ax2 share axes
    ax3 = fig.add_subplot(3, 1, 3)  # ax3 has independent axes

    ax1.plot(range(len(right_total_force.iloc[start_index:end_index])), right_total_force.iloc[start_index:end_index],
        label="Right Insole Force")
    ax2.plot(range(len(left_total_force.iloc[start_index:end_index])), left_total_force.iloc[start_index:end_index],
        label="Left Insole Force")
    ax3.plot(range(len(sync_force.iloc[start_index:end_index])), sync_force.iloc[start_index:end_index], label="Sync Station Force")

    ax1.set(title="Right Insole Force", ylabel="force(kg)")
    ax2.set(title="Left Insole Force", ylabel="force(kg)")
    ax3.set(title="Sync Force Signal", xlabel="Sample Number", ylabel="force value")

    ax1.tick_params(labelbottom=True)  # show x-axis ticklabels
    ax2.tick_params(labelbottom=True)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper right")


## plot insole and emg data for displaying results (all thress images share axes)
def plotInsoleAlignedEmg(emg_aligned, left_insole_upsampled, right_insole_upsampled, start_index, end_index, sync_force=False,
        emg_columms=range(0, 65)):
    left_total_force = left_insole_upsampled.loc[:, 192]  # extract total force column
    right_total_force = right_insole_upsampled.loc[:, 192]
    if emg_aligned.shape[1] == 64:  # the input is filtered emg data, only contain 64 channel data
        emg_data = emg_aligned.iloc[:, emg_columms].sum(axis=1)
    else:  # the input is aligned emg data before filtering, contain other information more than 64 channel data
        if sync_force:  # plot syncstation load cell data
            emg_data = emg_aligned.iloc[:, -3]  # extract load cell column
        else:  # plot selected emg channel data
            emg_data = emg_aligned.iloc[:, emg_columms].sum(axis=1)  # calculate sum of emg signals from selected emg channels

    # plot
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].plot(range(len(right_total_force.iloc[start_index:end_index])),
                 right_total_force.iloc[start_index:end_index], label="Right Insole Force")
    axes[1].plot(range(len(left_total_force.iloc[start_index:end_index])), left_total_force.iloc[start_index:end_index],
                 label="Left Insole Force")
    axes[2].plot(range(len(emg_data.iloc[start_index:end_index])), emg_data.iloc[start_index:end_index],
                 label="Emg Signal")

    axes[0].set(title="Right Insole Force", ylabel="force(kg)")
    axes[1].set(title="Left Insole Force", ylabel="force(kg)")
    axes[2].set(title="Sync Force Signal", xlabel="Sample Number", ylabel="force Value")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels
    axes[1].tick_params(labelbottom=True)

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")


## align EMG and insoles based on timestamp
def alignInsoleEmgTimestamp(raw_emg_data, left_insole_aligned, right_insole_aligned):
    # get the beginning and ending timestamp for both insoles.
    left_insole_aligned[0] = pd.to_datetime(left_insole_aligned[0], format='%Y-%m-%d_%H:%M:%S.%f')  # the column named 0 is the timestamp
    right_insole_aligned[0] = pd.to_datetime(right_insole_aligned[0], format='%Y-%m-%d_%H:%M:%S.%f')
    # always select the earliest one as reference for emg alignment
    insole_begin_timestamp = min(left_insole_aligned.iloc[0, 0], right_insole_aligned.iloc[0, 0]) # select earliest one
    insole_end_timestamp = min((left_insole_aligned[0].iloc[-1], right_insole_aligned[0].iloc[-1]))

    # only keep data between the beginning and ending index
    raw_emg_data[0] = pd.to_datetime(raw_emg_data[0], format='%Y-%m-%d_%H:%M:%S.%f')
    emg_start_index = (pd.to_datetime(raw_emg_data[0]) - insole_begin_timestamp).abs().idxmin() # obtain the closet beginning timestamp
    emg_end_index = (pd.to_datetime(raw_emg_data[0]) - insole_end_timestamp).abs().idxmin() # obtain the closet ending timestamp
    emg_aligned = raw_emg_data.iloc[emg_start_index:emg_end_index+1, :].reset_index(drop=True)
    return emg_aligned


## save the alignment parameters into a csv file
def saveAlignParameters(subject, data_file_name, left_start_index, right_start_index, left_end_index, right_end_index,
        emg_start_index="None", emg_end_index="None"):
    # save file path
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}'
    alignment_file = f'subject_{subject}_align_parameters.csv'
    alignment_file_path = os.path.join(data_dir, alignment_file)

    # alignment parameters to save
    columns = ['data_file_name', 'alignment_save_date', 'left_start_index', 'right_start_index', 'left_end_index', 'right_end_index',
        'emg_start_index', 'emg_end_index']
    save_parameters = [data_file_name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), left_start_index, right_start_index,
        left_end_index, right_end_index, emg_start_index, emg_end_index]

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
def readAlignParameters(subject, session, mode, version):
    data_dir = 'D:\Data\Insole_Emg'
    alignment_file = f'subject_{subject}\subject_{subject}_align_parameters.csv'
    alignment_file_path = os.path.join(data_dir, alignment_file)
    data_file_name = f'subject_{subject}_Experiment_{version}_session_{session}_{mode}'

    alignment_data = pd.read_csv(alignment_file_path, sep=',')  # header exists
    file_parameter = alignment_data.query('data_file_name == @data_file_name') # use @ to cite variable values
    if file_parameter.empty:  # if no alignment parameter found
        raise Exception(f"No alignment parameter found for data file: {data_file_name}")
    else:
        align_parameter = file_parameter.iloc[[-1]]  # extract the last row (apply the newest parameters)

        left_start_index = align_parameter['left_start_index'].iloc[0]
        left_end_index = align_parameter['left_end_index'].iloc[0]
        right_start_index = align_parameter['right_start_index'].iloc[0]
        right_end_index = align_parameter['right_end_index'].iloc[0]
        emg_start_index = align_parameter['emg_start_index'].iloc[0]
        emg_end_index = align_parameter['emg_end_index'].iloc[0]

        return left_start_index, right_start_index, left_end_index, right_end_index, emg_start_index, emg_end_index


## save all sensor data after alignment into a csc file
def saveAlignedData(subject, session, mode, version, left_insole_aligned, right_insole_aligned, emg_aligned):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}'
    data_file_name = f'subject_{subject}_Experiment_{version}_session_{session}_{mode}'

    left_insole_file = f'aligned_{mode}\left_insole\left_{data_file_name}_aligned.csv'
    right_insole_file = f'aligned_{mode}\\right_insole\\right_{data_file_name}_aligned.csv'
    emg_file = f'aligned_{mode}\emg\emg_{data_file_name}_aligned.csv'

    left_insole_path = os.path.join(data_dir, left_insole_file)
    right_insole_path = os.path.join(data_dir, right_insole_file)
    emg_path = os.path.join(data_dir, emg_file)

    left_insole_aligned.to_csv(left_insole_path, index=False)
    right_insole_aligned.to_csv(right_insole_path, index=False)
    emg_aligned.to_csv(emg_path, index=False)


## read all sensor data after alignment from a csc file
def readAlignedData(subject, session, mode, version):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}'
    data_file_name = f'subject_{subject}_Experiment_{version}_session_{session}_{mode}'

    left_insole_file = f'aligned_{mode}\left_insole\left_{data_file_name}_aligned.csv'
    right_insole_file = f'aligned_{mode}\\right_insole\\right_{data_file_name}_aligned.csv'
    emg_file = f'aligned_{mode}\emg\emg_{data_file_name}_aligned.csv'

    left_insole_path = os.path.join(data_dir, left_insole_file)
    right_insole_path = os.path.join(data_dir, right_insole_file)
    emg_path = os.path.join(data_dir, emg_file)

    left_insole_aligned = pd.read_csv(left_insole_path)
    right_insole_aligned = pd.read_csv(right_insole_path)
    emg_aligned = pd.read_csv(emg_path)

    return left_insole_aligned, right_insole_aligned, emg_aligned
