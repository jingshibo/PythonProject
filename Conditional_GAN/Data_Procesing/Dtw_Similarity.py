## import modules
import copy
import random
import numpy as np
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw


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
        fake_data = copy.deepcopy(synthetic_data)
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
            selected_fake_index = []
            for i in selected_reference_index:  # i is the row index of references in the dtw_results 2d array, the column refers to fake data
                sorted_fake_index = np.argsort(dtw_results[transition_type]['fake_reference_ds'][i, :])
                selected_fake_index.append(sorted_fake_index[: self.num_sample // self.num_reference])  # get the closest fake data index
            selected_fake_index = np.concatenate(selected_fake_index)
            fake_data[transition_type] = [synthetic_data[transition_type][index] for index in selected_fake_index]
        return fake_data, selected_fake_index, selected_reference_index


    # select representative fake data based on all reference curves available
    def bestFakeData(self, dtw_results, synthetic_data):
        fake_data = copy.deepcopy(synthetic_data)
        for transition_type in self.modes_generation.keys():
            # calculate the average distance value for each fake data with respect to all references
            dtw_fake_mean = np.mean(dtw_results[transition_type]['fake_reference_ds'], axis=0)
            sorted_fake_index = np.argsort(dtw_fake_mean)
            # select those data with smallest average distance to all references
            selected_fake_index = sorted_fake_index[:self.num_sample]
            fake_data[transition_type] = [synthetic_data[transition_type][index] for index in selected_fake_index]
        selected_references = []  # for consistence with the pickFakeData() function.
        return fake_data, selected_fake_index, selected_references


    def plotPath(self, fake_curve, real_curve, source, mode, fake_index, reference_index):
        s1 = fake_curve[source][mode][:, fake_index]
        s2 = real_curve[source][mode][:, reference_index]
        d, paths = dtw.warping_paths(s1, s2)  # distance and DTW matrix
        best_path = dtw.best_path(paths)  # optimal path
        dtwvis.plot_warping(s1, s2, best_path)  # plot optimal path connecting two curves in two subplots
        # dtwvis.plot_warpingpaths(s1, s2, paths, best_path)  # plot DTW matrix
        # dtwvis.plot_warping_single_ax(s1, s2, best_path)  # plot optimal path and two curves in a single figure