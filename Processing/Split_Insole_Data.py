##
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##
data_file_name = 'subject0_20220805_175751'

##
split_data_1 = pd.DataFrame({
    'turnSSSS': [82500, 128758],
    'SSLW': [90565, 149219],
    'LW_1': [91210, 149893],
    'LWLW': [93010, 151800],
    'LW_2': [93841, 152590],
    'LWSA': [95627, 154156],
    'SA_1': [96404, 155000],
    'SASA': [98254, 156700],
    'SA_2': [99124, 157645],
    'SASS': [101031, 159567],
    'turnSASD': [101760, 160320],
    'SSSD': [117344, 170611],
    'SD_1': [117963, 171286],
    'SDSD': [119800, 173305],
    'SD_2': [120628, 174077],
    'SDLW': [122727, 176016],
    'LW_3': [123501, 176716],
    'LWLW': [125153, 178618],
    'LW_4': [126039, 179573],
    'LWSS': [128450, 181869]
})

##
split_data_2 = pd.DataFrame({
    'turnSSSS': [, ],
    'SSLW': [, ],
    'LW_1': [],
    'LWLW': [, ],
    'LW_2': [],
    'LWSD': [, ],
    'SD_1': [],
    'SDSD': [, ],
    'SD_2': [],
    'SDSS': [, ],
    'turnSDSA': [, ],
    'SSSA': [, ],
    'SA_1': [],
    'SASA': [],
    'SA_2': [],
    'SALW': [],
    'LW_3': [],
    'LWLW': [],
    'LW_4': [],
    'LWSS': []
})

##
def plotSplitLine(emg_dataframe, left_insole_dataframe, right_insole_dataframe, start_index, end_index,
                  left_force_baseline, right_force_baseline):
    left_total_force = left_insole_dataframe[195]
    right_total_force = right_insole_dataframe[195]
    emg_data = emg_dataframe.sum(1)
    left_length = len(left_total_force.iloc[start_index:end_index])
    right_length = len(right_total_force.iloc[start_index:end_index])
    emg_length = len(emg_data.iloc[start_index:end_index])

    # plot emg and insole force
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].plot(range(left_length), left_total_force.iloc[start_index:end_index], label="Left Insole Force")
    axes[1].plot(range(right_length), right_total_force.iloc[start_index:end_index], label="Right Insole Force")
    axes[2].plot(range(emg_length), emg_data.iloc[start_index:end_index], label="Emg Signal")

    # add force baseline
    axes[0].plot(range(left_length), left_length * [left_force_baseline], label="Left Force Baseline")
    axes[1].plot(range(right_length), right_length * [right_force_baseline], label="Right Force Baseline")

    # find intersection point's x value
    left_x = np.array(range(left_length))
    right_x = np.array(range(right_length))
    left_force = left_total_force.iloc[start_index:end_index].to_numpy()
    right_force = right_total_force.iloc[start_index:end_index].to_numpy()

    left_baseline = np.full(left_length, left_force_baseline)
    right_baseline = np.full(right_length, right_force_baseline)
    left_cross_idx = np.argwhere(np.diff(np.sign(left_force - left_baseline))).flatten()
    right_cross_idx = np.argwhere(np.diff(np.sign(right_force - right_baseline))).flatten()

    # plot intersection point's x value
    axes[0].plot(left_x[left_cross_idx], left_baseline[left_cross_idx], 'r')  # plot intersection points
    for i, x_value in enumerate(left_cross_idx):  # annotate intersection points
        axes[0].annotate(x_value, (left_x[left_cross_idx[i]], left_baseline[left_cross_idx[i]]), fontsize=9)
    axes[1].plot(right_x[right_cross_idx], right_baseline[right_cross_idx], 'r')  # plot intersection points
    for i, x_value in enumerate(right_cross_idx):  # annotate intersection points
        axes[1].annotate(x_value, (right_x[right_cross_idx[i]], right_baseline[right_cross_idx[i]]), fontsize=9)

    # plot parameters
    axes[0].set(title="Left Insole Force", ylabel="force(kg)")
    axes[1].set(title="Right Insole Force", ylabel="force(kg)")
    axes[2].set(title="Emg Signal", xlabel="Sample Number", ylabel="Emg Value")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels
    axes[1].tick_params(labelbottom=True)

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")

