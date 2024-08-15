'''
    Extract emg data of each locomotion mode from .mat files and save the extracted data in the aligned data folder.
'''

##
import scipy.io
import pandas as pd
import os

## load emg data from .mat file
# Load the .mat file
project = 'HDsEMG_Recognition'
subject = 'Number9'
mat_file = scipy.io.loadmat(f'D:\Data\\{project}\subject_{subject}\\raw_data\\Experiment_Data_20211203_Shibo.mat')

# extract data from the .mat file
emg_data = mat_file['orig_raw_EMG']
emg_np = [emg_data[0, i] for i in range(emg_data.shape[1])]
emg_pd = [pd.DataFrame(matrix) for matrix in emg_np]

##
modes = {'standing': 0, 'level': 1, 'upstairs': 2, 'downstairs': 2, 'upslope': 3, 'downslope': 3}

for mode, index in modes.items():
    data_dir = f'D:\Data\\{project}\subject_{subject}\\aligned_data'
    data_file = f'emg_subject_{subject}_{mode}_aligned.csv'
    emg_path = os.path.join(data_dir, data_file)

    emg_pd[index].to_csv(emg_path, index=False)