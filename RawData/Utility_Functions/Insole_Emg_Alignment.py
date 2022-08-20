##
import pandas as pd
import matplotlib.pyplot as plt
import os

## align EMG and insoles
def alignInsoleEmg(raw_emg_data, left_insole_aligned, right_insole_aligned):
    # get the average beginning and ending insole timestamp
    left_insole_aligned[0] = pd.to_datetime(left_insole_aligned[0], format='%Y-%m-%d_%H:%M:%S.%f')
    right_insole_aligned[0] = pd.to_datetime(right_insole_aligned[0], format='%Y-%m-%d_%H:%M:%S.%f')
    insole_begin_timestamp = min(left_insole_aligned.iloc[0, 0], right_insole_aligned.iloc[0, 0]) # select earliest one
    insole_end_timestamp = min((left_insole_aligned[0].iloc[-1], right_insole_aligned[0].iloc[-1]))

    # only keep data between the beginning and ending index
    raw_emg_data[0] = pd.to_datetime(raw_emg_data[0], format='%Y-%m-%d_%H:%M:%S.%f')
    emg_start_index = (pd.to_datetime(raw_emg_data[0]) - insole_begin_timestamp).abs().idxmin() # obtain the closet beginning timestamp
    emg_end_index = (pd.to_datetime(raw_emg_data[0]) - insole_end_timestamp).abs().idxmin() # obtain the closet ending timestamp
    emg_aligned = raw_emg_data.iloc[emg_start_index:emg_end_index+1, :].reset_index(drop=True)
    return emg_aligned


## plot insole and emg data
def plotInsoleEmg(emg_dataframe, left_insole_dataframe, right_insole_dataframe, start_index, end_index):
    left_total_force = left_insole_dataframe.iloc[:, 192] # extract total force column
    right_total_force = right_insole_dataframe.iloc[:, 192]
    emg_data = emg_dataframe.sum(1)

    # plot
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].plot(range(len(left_total_force.iloc[start_index:end_index])), left_total_force.iloc[start_index:end_index],
                 label="Left Insole Force")
    axes[1].plot(range(len(right_total_force.iloc[start_index:end_index])),
                 right_total_force.iloc[start_index:end_index], label="Right Insole Force")
    axes[2].plot(range(len(emg_data.iloc[start_index:end_index])), emg_data.iloc[start_index:end_index],
                 label="Emg Signal")

    axes[0].set(title="Left Insole Force", ylabel="force(kg)")
    axes[1].set(title="Right Insole Force", ylabel="force(kg)")
    axes[2].set(title="Emg Signal", xlabel="Sample Number", ylabel="Emg Value")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels
    axes[1].tick_params(labelbottom=True)

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")


def saveAlignedData(subject, session, mode, left_insole_aligned, right_insole_aligned, emg_aligned):
    data_dir = 'D:\Data\Insole_Emg'
    data_file_name = f'subject_{subject}_session_{session}_{mode}'

    left_insole_file = f'subject_{subject}\\aligned_{mode}\left_insole\left_{data_file_name}_aligned.csv'
    right_insole_file = f'subject_{subject}\\aligned_{mode}\\right_insole\\right_{data_file_name}_aligned.csv'
    emg_file = f'subject_{subject}\\aligned_{mode}\emg\emg_{data_file_name}_aligned.csv'

    left_insole_path = os.path.join(data_dir, left_insole_file)
    right_insole_path = os.path.join(data_dir, right_insole_file)
    emg_path = os.path.join(data_dir, emg_file)

    left_insole_aligned.to_csv(left_insole_path, index=False)
    right_insole_aligned.to_csv(right_insole_path, index=False)
    emg_aligned.to_csv(emg_path, index=False)


def readAlignedData(subject, session, mode):
    data_dir = 'D:\Data\Insole_Emg'
    data_file_name = f'subject_{subject}_session_{session}_{mode}'

    left_insole_file = f'subject_{subject}\\aligned_{mode}\left_insole\left_{data_file_name}_aligned.csv'
    right_insole_file = f'subject_{subject}\\aligned_{mode}\\right_insole\\right_{data_file_name}_aligned.csv'
    emg_file = f'subject_{subject}\\aligned_{mode}\emg\emg_{data_file_name}_aligned.csv'

    left_insole_path = os.path.join(data_dir, left_insole_file)
    right_insole_path = os.path.join(data_dir, right_insole_file)
    emg_path = os.path.join(data_dir, emg_file)

    left_insole_aligned = pd.read_csv(left_insole_path)
    right_insole_aligned = pd.read_csv(right_insole_path)
    emg_aligned = pd.read_csv(emg_path)

    return left_insole_aligned, right_insole_aligned, emg_aligned
