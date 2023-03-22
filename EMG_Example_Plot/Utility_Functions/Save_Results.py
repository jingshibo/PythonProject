import os
import numpy as np


##  save hdemg samples with the same bipolar value
def saveHdemgWithSameBipolar(subject, version, feature_type, hdemg_data, bipolar_data):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\comparison_data'
    data_file = f'subject_{subject}_Experiment_{version}_hdemg_set_{feature_type}.npy'
    data_path = os.path.join(data_dir, data_file)
    np.save(data_path, hdemg_data)

    data_file = f'subject_{subject}_Experiment_{version}_bipolar_set_{feature_type}.npy'
    data_path = os.path.join(data_dir, data_file)
    np.save(data_path, bipolar_data)


##  load hdemg samples with the same bipolar value
def loadFeatureExamples(subject, version, feature_type):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\comparison_data'
    data_file = f'subject_{subject}_Experiment_{version}_hdemg_set_{feature_type}.npy'
    data_path = os.path.join(data_dir, data_file)
    hdemg_data = np.load(data_path)
    hdemg_list = [hdemg_data[i] for i in range(hdemg_data.shape[0])]

    data_file = f'subject_{subject}_Experiment_{version}_bipolar_set_{feature_type}.npy'
    data_path = os.path.join(data_dir, data_file)
    bipolar_data = np.load(data_path)
    bipolar_list = [bipolar_data[i] for i in range(bipolar_data.shape[0])]

    return hdemg_list, bipolar_list


## save channel lost hdemg samples
def saveChannelLostHdemg(subject, version, feature_type, emg_data):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\comparison_data'
    data_file = f'subject_{subject}_Experiment_{version}_channel_lost_{feature_type}.npy'
    data_path = os.path.join(data_dir, data_file)
    np.save(data_path, emg_data)


##  load channel lost hdemg samples
def loadChannelLostHdemg(subject, version, feature_type):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\comparison_data'
    data_file = f'subject_{subject}_Experiment_{version}_channel_lost_{feature_type}.npy'
    data_path = os.path.join(data_dir, data_file)
    emg_data = np.load(data_path)
    emg_list = [emg_data[i] for i in range(emg_data.shape[0])]

    return emg_list

