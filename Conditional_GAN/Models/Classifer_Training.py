from Transition_Prediction.Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Transition_Prediction.Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Cycle_GAN.Functions import Classify_Testing
from Conditional_GAN.Data_Procesing import cGAN_Processing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model


class DataProcessing:
    def __init__(self, real_emg, estimated_blending_factors, old_emg_normalized, feature_window_size, feature_window_increment_ms,
            sample_rate, num_epochs, batch_size, decay_epochs, report_period=10):
        self.real_emg = real_emg
        self.estimated_blending_factors = estimated_blending_factors
        self.old_emg_normalized = old_emg_normalized
        self.feature_window_size = feature_window_size
        self.feature_window_increment_ms = feature_window_increment_ms
        self.sample_rate = sample_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.decay_epochs = decay_epochs
        self.report_period = report_period

    def separate_data(self, period):
        reorganized_old_LWLW = cGAN_Processing.separateByTimeInterval(self.real_emg['old']['emg_LWLW'], timepoint_interval=1, length=period)
        reorganized_old_SASA = cGAN_Processing.separateByTimeInterval(self.real_emg['old']['emg_SASA'], timepoint_interval=1, length=period)
        return {'gen_data_1': reorganized_old_LWLW, 'gen_data_2': reorganized_old_SASA, 'blending_factors': self.estimated_blending_factors}

    def generate_fake_data(self, reorganized_data, interval, repetition=1, random_pairing=False):
        fake_data = cGAN_Processing.generateFakeData(reorganized_data, interval, repetition=repetition, random_pairing=random_pairing)
        reorganized_fake_data = cGAN_Processing.reorganizeFakeData(fake_data)
        return {'emg_LWSA': reorganized_fake_data}

    def build_training_data(self, generated_data, leave_percentage):
        leave_one_groups = Data_Preparation.leaveOneSet(leave_percentage, generated_data, shuffle=True)
        sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(leave_one_groups,
            self.feature_window_size,
            increment=self.feature_window_increment_ms * self.sample_rate)
        normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
        shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
        return shuffled_groups, feature_window_per_repetition

    def train(self, shuffled_groups):
        train_model = Raw_Cnn2d_Model.ModelTraining(self.num_epochs, self.batch_size, report_period=self.report_period)
        models, model_results = train_model.trainModel(shuffled_groups, self.decay_epochs)
        return models, model_results

    def evaluate_fake_data(self, model_results, feature_window_per_repetition, predict_window_shift_unit, predict_using_window_number):
        reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
        sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
            predict_window_shift_unit, predict_using_window_number, initial_start=0)
        accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
        average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(
            accuracy_bygroup, cm_bygroup)
        accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
            predict_window_shift_unit)
        return accuracy, cm_recall

    def get_real_data_test_set(self, keys, old_emg_normalized, leave_one_groups, test_ratio, feature_window_size, increment, sample_rate):
        real_test_data = cGAN_Processing.getRealDataSet(keys, old_emg_normalized, leave_one_groups, test_ratio)
        sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(real_test_data, feature_window_size,
            increment=increment * sample_rate)
        normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
        shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
        return shuffled_groups, feature_window_per_repetition

    def test_real_data(self, models, shuffled_groups, batch_size):
        test_model = Classify_Testing.ModelTesting(models[0], batch_size)
        test_result = test_model.testModel(shuffled_groups)
        return test_result