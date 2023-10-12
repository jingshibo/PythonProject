'''
   Using the blending factors estimated from the trained cGAN model, we generate synthetic data to train the classifier. Subsequently,
   we test the classifier using real data. If the classification results are satisfactory, it indicates that the blending factors
   produced by the trained GAN model are effective.
'''


##
import copy
from Transition_Prediction.Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Transition_Prediction.Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Conditional_GAN.Models import Classify_Testing
from Conditional_GAN.Data_Procesing import Process_Fake_Data
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model


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
    def generateFakeData(self, extracted_emg, data_source, modes_generation, real_emg_normalized, repetition=1, random_pairing=True):
        '''
            :param data_source: selected from 'old' and 'new'
            :param modes_generation: such as  {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}.
            The order in the list is important, corresponding to gen_data_1 and gen_data_2.
            :param repetition and random_pairing are two unnecessary parameter, used for compatibility with the generateFakeDataRandomMatch() function.

        '''
        length = self.start_before_toeoff_ms + self.endtime_after_toeoff_ms
        synthetic_data = copy.deepcopy(real_emg_normalized)
        data_for_generation = {'gen_data_1': None, 'gen_data_2': None, 'blending_factors': None}

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
            # create synthetic training data
            fake_emg_data = {transition_type: reorganized_fake_data}
            synthetic_data = Process_Fake_Data.replaceUsingFakeEmg(fake_emg_data, synthetic_data)

        return synthetic_data


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
        return shuffled_test_set


    # train classify models
    def trainClassifier(self, shuffled_groups, num_epochs=50, batch_size=2048, decay_epochs=20, report_period=10):
        train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=report_period)
        models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)
        return models, model_results


    # test classify models
    def testClassifier(self, models, shuffled_test_set, batch_size=2048):
        test_model = Classify_Testing.ModelTesting(models, batch_size)
        test_result = test_model.testModel(shuffled_test_set)
        return test_result


    # calculate classification accuracy
    def evaluateClassifyResults(self, model_results):
        reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
        sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results,
            self.feature_window_per_repetition, self.predict_window_shift_unit, self.predict_using_window_number, initial_start=0)
        accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
        average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(
            accuracy_bygroup, cm_bygroup)
        accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay,
            self.feature_window_increment_ms, self.predict_window_shift_unit)
        return accuracy, cm_recall




