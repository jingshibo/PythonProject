## import modules
import copy
import random
import numpy as np
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Plot_Emg_Data


class Dtw_Distance:
    def __init__(self, modes_generation, num_sample, num_reference):
        self.modes_generation = modes_generation
        self.num_sample = num_sample
        self.num_reference = num_reference


    ## using dtw package to calculate DTW distance between reference and synthetic data
    def calcuDtwDistance(self, fake_data, real_data):
        dtw_results = {}
        # loop over the modes to generate
        for transition_type in self.modes_generation.keys():
            fake = fake_data[transition_type].T
            reference = real_data[transition_type].T
            fake_sample_number = fake.shape[0]
            reference_sample_number = reference.shape[0]

            time_series_matrix = np.vstack([fake, reference]).astype(np.double)
            # compute the distance between each fake data and each reference data
            ds = dtw.distance_matrix_fast(time_series_matrix, block=((0, fake_sample_number), (fake_sample_number, len(time_series_matrix))))
            # compute the distance within reference data
            ds_reference = dtw.distance_matrix_fast(reference)
            # only retain valid distance values in the sparse matrix
            dtw_results[transition_type] = {'fake_reference_ds': ds[fake_sample_number:, :fake_sample_number],
                'within_reference_ds': ds_reference}

        return dtw_results


    ## select reference curves and return the most matching fake dataset
    def selectFakeData(self, dtw_results, synthetic_data, random_reference=False):
        # when selecting fake data based on references, there may be same data from different reference. Then we would get the next one instead.
        def avoidReplicatedData(): # nested functions have access to variables from the enclosing function's scope, so no need for any arguments
            select_fake_index = []
            added_fake_indices = set()  # To keep track of already added fake indices
            for i in selected_reference_index:
                sorted_fake_index = np.argsort(dtw_results[transition_type]['fake_reference_ds'][i, :])

                count = 0  # To keep track of how many unique fake indices have been added for the current i
                idx = 0  # To iterate through sorted_fake_index
                current_selected_fake = []  # To temporarily store the selected fake indices for the current i
                while count < self.num_sample // self.num_reference:
                    if sorted_fake_index[idx] not in added_fake_indices:
                        current_selected_fake.append(sorted_fake_index[idx])
                        added_fake_indices.add(sorted_fake_index[idx])
                        count += 1
                    idx += 1
                select_fake_index.extend(current_selected_fake)

            select_fake_index = np.array(select_fake_index)
            return select_fake_index

        selected_fake_data = copy.deepcopy(synthetic_data)
        selected_fake_indices = {}
        selected_reference_indices = {}
        # calculate the average distance value between the reference data to select a representative reference curve
        for transition_type in self.modes_generation.keys():
            # select reference data
            if random_reference:  # select references randomly
                selected_reference_index = random.sample(range(len(dtw_results[transition_type]['within_reference_ds']), self.num_sample))
            else:  # select representative references
                dtw_reference_mean = np.mean(dtw_results[transition_type]['within_reference_ds'], axis=0)
                sorted_reference_index = np.argsort(dtw_reference_mean)
                selected_reference_index = sorted_reference_index[0::len(sorted_reference_index) // self.num_reference][0: self.num_reference]

            # return the fake dataset closest to the selected references
            selected_fake_index = avoidReplicatedData()
            # selected_fake_index = []
            # for i in selected_reference_index:  # i is the row index of references in the dtw_results 2d array, the column refers to fake data
            #     sorted_fake_index = np.argsort(dtw_results[transition_type]['fake_reference_ds'][i, :])
            #     selected_fake_index.append(sorted_fake_index[: self.num_sample // self.num_reference])  # get the closest fake data index
            # selected_fake_index = np.concatenate(selected_fake_index)

            selected_fake_data[transition_type] = [synthetic_data[transition_type][index] for index in selected_fake_index]
            selected_fake_indices[transition_type] = selected_fake_index
            selected_reference_indices[transition_type] = selected_reference_index
        return selected_fake_data, selected_fake_indices, selected_reference_indices


    # select representative fake data based on all reference curves available
    def bestFakeData(self, dtw_results, synthetic_data):
        fake_data = copy.deepcopy(synthetic_data)
        selected_fake_indices = {}
        for transition_type in self.modes_generation.keys():
            # calculate the average distance value for each fake data with respect to all references
            dtw_fake_mean = np.mean(dtw_results[transition_type]['fake_reference_ds'], axis=0)
            sorted_fake_index = np.argsort(dtw_fake_mean)
            # select those data with smallest average distance to all references
            selected_fake_index = sorted_fake_index[:self.num_sample]
            fake_data[transition_type] = [synthetic_data[transition_type][index] for index in selected_fake_index]
            selected_fake_indices[transition_type] = selected_fake_index
        selected_reference_index = []  # for consistence with the pickFakeData() function.
        return fake_data, selected_fake_indices, selected_reference_index


def extractFakeData(synthetic_data, real_data, modes_generation, cutoff_frequency, num_sample, num_reference, method='select'):
    # low pass filtering
    synthetic_envelope = {key: Process_Fake_Data.clipSmoothEmgData(value, cutoff_frequency) for key, value in synthetic_data.items()}
    old_emg_envelope = {key: Process_Fake_Data.clipSmoothEmgData(value, cutoff_frequency) for key, value in real_data.items()}
    # calculate mean value across all channels for each repetition
    fake = Plot_Emg_Data.averageEmgValues(synthetic_envelope)
    real = Plot_Emg_Data.averageEmgValues(old_emg_envelope)
    # compute dtw distance between fake data and references
    dtw_distance = Dtw_Distance(modes_generation, num_sample=num_sample, num_reference=num_reference)
    dtw_results_1 = dtw_distance.calcuDtwDistance(fake['emg_1_repetition_list'], real['emg_1_repetition_list'])
    dtw_results_2 = dtw_distance.calcuDtwDistance(fake['emg_2_repetition_list'], real['emg_2_repetition_list'])
    if method == 'select':
        selected_fake_data_1, selected_fake_index_1, selected_reference_index_1 = dtw_distance.selectFakeData(dtw_results_1, synthetic_data)
        selected_fake_data_2, selected_fake_index_2, selected_reference_index_2 = dtw_distance.selectFakeData(dtw_results_2, synthetic_data)
    elif method == 'best':
        selected_fake_data_1, selected_fake_index_1, selected_reference_index_1 = dtw_distance.bestFakeData(dtw_results_1, synthetic_data)
        selected_fake_data_2, selected_fake_index_2, selected_reference_index_2 = dtw_distance.bestFakeData(dtw_results_2, synthetic_data)
    extracted_data = {'selected_fake_data_1': selected_fake_data_1, 'selected_fake_index_1': selected_fake_index_1,
        'selected_reference_index_1': selected_reference_index_1, 'selected_fake_data_2': selected_fake_data_2,
        'selected_fake_index_2': selected_fake_index_2, 'selected_reference_index_2': selected_reference_index_2, 'fake_averaged': fake,
        'real_averaged': real}
    return extracted_data


def plotPath(fake_curve, real_curve, source, mode, fake_index, reference_index):
    s1 = fake_curve[source][mode][:, fake_index]
    s2 = real_curve[source][mode][:, reference_index]
    d, paths = dtw.warping_paths(s1, s2)  # distance and DTW matrix
    best_path = dtw.best_path(paths)  # optimal path
    dtwvis.plot_warping(s1, s2, best_path)  # plot optimal path connecting two curves in two subplots
    dtwvis.plot_warpingpaths(s1, s2, paths, best_path)  # plot DTW matrix
    # dtwvis.plot_warping_single_ax(s1, s2, best_path)  # plot optimal path and two curves in a single figure

