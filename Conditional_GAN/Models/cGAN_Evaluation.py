'''
   Using the blending factors estimated from the trained cGAN model, we generate synthetic data to train the classifier. Subsequently,
   we test the classifier using real data. If the classification results are satisfactory, it indicates that the blending factors
   produced by the trained GAN model are effective.
'''


##
import copy
import numpy as np
from Transition_Prediction.Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Transition_Prediction.Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Conditional_GAN.Models import Classify_Testing, Classify_TL_Training
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model
from Cycle_GAN.Functions import Data_Processing


##
class cGAN_Evaluation:
    def __init__(self, gen_results, window_parameters):
        self.gen_results = gen_results
        self.sample_rate = window_parameters['sample_rate']
        self.feature_window_size = window_parameters['feature_window_size']
        self.feature_window_increment_ms = window_parameters['feature_window_increment_ms']
        self.start_before_toeoff_ms = window_parameters['start_before_toeoff_ms']
        self.endtime_after_toeoff_ms = window_parameters['endtime_after_toeoff_ms']
        self.predict_window_shift_unit = window_parameters['predict_window_shift_unit']
        self.predict_using_window_number = window_parameters['predict_using_window_number']
        self.feature_window_per_repetition = None


    # generate fake emg data and substitute original emg dataset using this data
    def generateFakeData(self, extracted_emg, data_source, modes_generation, real_emg_normalized, repetition=1, random_pairing=True,
            spatial_filtering=True, sigma=1, axes=(2, 3), radius=4):
        '''
            :param data_source: selected from 'old' and 'new'
            :param modes_generation: such as  {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}.
            The order in the list is important, corresponding to gen_data_1 and gen_data_2.
            :param repetition and random_pairing are two unnecessary parameter, used for compatibility with the
            Process_Fake_Data.generateFakeDataRandomMatch() function.

        '''
        length = self.start_before_toeoff_ms + self.endtime_after_toeoff_ms
        synthetic_data = copy.deepcopy(real_emg_normalized)
        data_for_generation = {'gen_data_1': None, 'gen_data_2': None, 'blending_factors': None}

        fake_emg_images = {}
        fake_emg_reorganized = {}
        for transition_type, modes in modes_generation.items():
            # create data_for_generation dict
            data_for_generation['gen_data_1'] = Process_Fake_Data.separateByInterval(
                extracted_emg[transition_type][data_source][modes[0]], timepoint_interval=1, length=length)
            data_for_generation['gen_data_2'] = Process_Fake_Data.separateByInterval(
                extracted_emg[transition_type][data_source][modes[1]], timepoint_interval=1, length=length)
            data_for_generation['blending_factors'] = self.gen_results[transition_type]['model_results']
            # generate fake data
            fake_data = Process_Fake_Data.generateFakeDataByCurve(data_for_generation,  # only the first parameter is useful
                self.gen_results[transition_type]['training_parameters']['interval'], repetition=repetition, random_pairing=random_pairing)
            reorganized_fake_data = Process_Fake_Data.reorganizeFakeData(fake_data)
            emg_reshaped = [np.transpose(np.reshape(arr, newshape=(arr.shape[0], 13, -1, 1), order='F'), (0, 3, 1, 2)) for arr in
                reorganized_fake_data]  # convert emg to images

            # spatial filter fake data
            if spatial_filtering:
                filtered_fake_emg_images = Process_Raw_Data.spatialFilterEmgData({transition_type: emg_reshaped},
                    sigma=sigma, axes=axes, radius=radius)
                fake_emg_reorganized[transition_type] = [np.reshape(np.transpose(arr, (0, 2, 3, 1)), newshape=(arr.shape[0], -1), order='F')
                    for arr in filtered_fake_emg_images[transition_type]]
                fake_emg_images[transition_type] = filtered_fake_emg_images[transition_type]
            else:
                fake_emg_reorganized[transition_type] = reorganized_fake_data
                fake_emg_images[transition_type] = emg_reshaped


        synthetic_data = Process_Fake_Data.replaceUsingFakeEmg(fake_emg_reorganized, synthetic_data)
        return synthetic_data, fake_emg_images


    # generate fake data by adding noise to reference real emg data and substitute original emg dataset using this data
    def generateNoiseData(self, real_emg_normalized, reference_data, num_sample=60, snr=25):
        '''
        :param real_emg_normalized: real emg data dict
        :param reference_data: selected real data used for noisy sample generation
        :param num_sample:  the number of noisy samples to generate for each locomotion mode
        :param snr_db:  the amplitude of noise to add based on signal-to-noise ratio
        :return: real emg data with certain modes replaced by generated noisy data
        '''

        # add noise to reference data
        def generate_noisy_sample(real_data, snr_db):  # add Gaussian Noise with signal-to-noise ratio (SNR) of 25
            # Calculate the variance of the signal
            signal_var = np.var(real_data)
            # Convert SNR from dB scale to linear scale
            snr_linear = 10 ** (snr_db / 10)
            # Calculate the required noise variance
            noise_variance = signal_var / snr_linear
            # Generate the noise with the calculated standard deviation
            noise = np.random.normal(0, np.sqrt(noise_variance), real_data.shape)
            return real_data + noise

        synthetic_data = copy.deepcopy(real_emg_normalized)
        # Generate noisy samples for each numpy array in reference_data
        reference_data_with_noise = {locomotion_mode: [generate_noisy_sample(array, snr) for array in array_list for _ in
            range(int(num_sample / len(array_list)))] for locomotion_mode, array_list in reference_data.items()}
        synthetic_data = Process_Fake_Data.replaceUsingFakeEmg(reference_data_with_noise, synthetic_data)

        return synthetic_data


    # extract reference real emg data from the real dataset and remove them from the real dataset
    def addressReferenceData(self, reference_indices, filtered_real_data):
        # selecting the data based on reference_indices
        reference_new_real_data = {}
        adjusted_new_real_data = copy.deepcopy(filtered_real_data)
        for key, indices in reference_indices.items():
            if len(indices) > 5:  # maximum reference data number is 5
                raise Exception('too many reference data!')
            reference_new_real_data[key] = [filtered_real_data[key][i] for i in indices]
            # Update filtered_new_real_data by excluding the selected data
            adjusted_new_real_data[key] = [filtered_real_data[key][i] for i in range(len(filtered_real_data[key])) if
                i not in indices]
        return reference_new_real_data, adjusted_new_real_data


    # divide classifier training set for transfer learning
    def classifierTlTrainSet(self, generated_data, reference_data, dataset='leave_one_set', fold=5, training_percent=0.8, minimum_train_number=10):
        if dataset == 'leave_one_set':
            train_dataset = Data_Preparation.leaveOutDataSet(training_percent, generated_data, shuffle=True)
        elif dataset == 'cross_validation_set':
            train_dataset = Data_Preparation.crossValidationSet(fold, generated_data, shuffle=True)
        else:
            raise Exception("wrong keywords")
        # adjust the train dataset to meet the requirement of transfer learning
        for group_number, group_value in train_dataset.items():
            # exchange the training set and test set
            group_value['train_set'], group_value['test_set'] = group_value['test_set'], group_value['train_set']
            # add reference data into the train_set as they are available new transition data for model training
            for key, data_list in reference_data.items():
                if key in group_value['train_set']:
                    group_value['train_set'][key].extend(data_list)
            # move data from test set to train set to keep a minimum sample number (default:10) for each mode
            for key in group_value['train_set']:
                while len(group_value['train_set'][key]) < minimum_train_number and group_value['test_set'][key]:
                    group_value['train_set'][key].append(group_value['test_set'][key].pop(0))
        # shuffle training set
        sliding_window_dataset, self.feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(train_dataset,
            self.feature_window_size, increment=self.feature_window_increment_ms * self.sample_rate)
        normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
        shuffled_train_set = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
        return train_dataset, shuffled_train_set


    # divide classifier training set
    def classifierTrainSet(self, generated_data, dataset='leave_one_set', fold=5, training_percent=0.8):
        if dataset == 'leave_one_set':
            train_dataset = Data_Preparation.leaveOutDataSet(training_percent, generated_data, shuffle=True)
        elif dataset == 'cross_validation_set':
            train_dataset = Data_Preparation.crossValidationSet(fold, generated_data, shuffle=True)
        else:
            raise Exception("wrong keywords")
        sliding_window_dataset, self.feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(train_dataset,
            self.feature_window_size, increment=self.feature_window_increment_ms * self.sample_rate)
        normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
        shuffled_train_set = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
        return train_dataset, shuffled_train_set


    # divide classifier test set
    def classifierTestSet(self, modes_generation, real_emg_normalized, train_dataset, test_ratio=0.5):
        test_dataset = copy.deepcopy(train_dataset)
        for transition_type, modes in modes_generation.items():
            test_dataset = Process_Fake_Data.replaceUsingRealEmg(transition_type, real_emg_normalized, test_dataset, test_ratio)
        sliding_window_dataset, self.feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(test_dataset,
            self.feature_window_size, increment=self.feature_window_increment_ms * self.sample_rate)
        normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
        shuffled_test_set = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
        return test_dataset, shuffled_test_set


    # train classify models
    def trainClassifier(self, shuffled_groups, num_epochs=50, batch_size=1800, decay_epochs=20, report_period=10):
        train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=report_period)
        models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)
        return models, model_results


    # train classify models based on transfer learning
    def trainTlClassifier(self, pretrained_models, shuffled_groups, num_epochs=50, batch_size=1800, decay_epochs=20, report_period=10):
        train_model = Classify_TL_Training.ModelTraining(pretrained_models, num_epochs, batch_size, report_period=report_period)
        models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)
        return models, model_results


    # test classify models
    def testClassifier(self, models, shuffled_test_set, batch_size=2048):
        test_model = Classify_Testing.ModelTesting(models, batch_size)
        test_result = test_model.testModel(shuffled_test_set)
        return test_result


    # calculate classification accuracy with prior knowledge
    def evaluateClassifyResultsByGroup(self, model_results):
        reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
        sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results,
            self.feature_window_per_repetition, self.predict_window_shift_unit, self.predict_using_window_number, initial_start=0)
        accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
        average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(
            accuracy_bygroup, cm_bygroup)
        accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay,
            self.feature_window_increment_ms, self.predict_window_shift_unit)
        return accuracy, cm_recall


    # calculate classification accuracy without prior knowledge
    def evaluateClassifyResults(self, model_results):
        sliding_majority_vote = Sliding_Ann_Results.majorityVoteResults(model_results, self.feature_window_per_repetition,
            self.predict_window_shift_unit, self.predict_using_window_number, initial_start=0)
        accuracy_allgroup, cm_allgroup = Data_Processing.slidingMvResults(sliding_majority_vote)
        average_accuracy_with_delay, average_cm_recall_with_delay = Data_Processing.averageAccuracyCm(accuracy_allgroup, cm_allgroup,
            self.feature_window_increment_ms, self.predict_window_shift_unit)
        return average_accuracy_with_delay, average_cm_recall_with_delay



