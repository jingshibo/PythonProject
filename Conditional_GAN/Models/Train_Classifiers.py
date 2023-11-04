##
# import copy
import random
import datetime
from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, cGAN_Evaluation, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data, Dtw_Similarity, Post_Process_Data
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Transition_Prediction.Models.Utility_Functions import Data_Preparation


class ClassifierTraining():
    def __init__(self, classifier_filter_kernel, gan_filter_kernel, window_parameters, modes_generation):
        self.gan_filter_kernel = gan_filter_kernel
        self.classifier_filter_kernel = classifier_filter_kernel
        self.window_parameters = window_parameters
        self.modes_generation = modes_generation


    ## load training data and blending factors
    def loadTrainingData(self, old_emg_data, new_emg_data, subject, version, gan_result_set, checkpoint_result_path,
            start_index, end_index, start_before_toeoff_ms, range_limit, time_interval, length, epoch_number):  # slice blending factors
        load_gen_results = Model_Storage.loadBlendingFactors(subject, version, gan_result_set, 'cGAN', self.modes_generation,
            checkpoint_result_path, epoch_number=epoch_number)
        # gen_results = Process_Raw_Data.spatialFilterBlendingFactors(gen_result, kernel=gan_filter_kernel) # filter blending factor
        gen_results = Post_Process_Data.sliceBlendingFactors(load_gen_results, start_index, end_index)
        # slice emg data
        old_emg_data_classify, self.window_parameters = Post_Process_Data.sliceEmgData(old_emg_data, start=start_index, end=end_index,
            toeoff=start_before_toeoff_ms, sliding_results=False)
        new_emg_data_classify, _ = Post_Process_Data.sliceEmgData(new_emg_data, start=start_index, end=end_index, toeoff=start_before_toeoff_ms,
            sliding_results=False)

        # normalize and extract emg data for classification model training
        old_emg_classify_normalized, new_emg_classify_normalized, old_emg_classify_reshaped, new_emg_classify_reshaped = \
            Process_Raw_Data.normalizeFilterEmgData(
            old_emg_data_classify, new_emg_data_classify, range_limit, normalize='(0,1)', spatial_filter=True, kernel=self.gan_filter_kernel)
        extracted_emg_classify, _ = Process_Raw_Data.extractSeparateEmgData(self.modes_generation, old_emg_classify_reshaped,
            new_emg_classify_reshaped, time_interval, length, output_list=False)
        del old_emg_classify_reshaped, new_emg_classify_reshaped

        return gen_results, old_emg_classify_normalized, new_emg_classify_normalized, extracted_emg_classify


    ## train classifier (basic scenarios), training and testing using data from the same and different time
    def trainClassifierBasicScenarios(self, old_emg_classify_normalized, new_emg_classify_normalized):
        # original dataset
        old_real_emg_grids = Post_Process_Data.separateEmgGrids(old_emg_classify_normalized, separate=True)
        new_real_emg_grids = Post_Process_Data.separateEmgGrids(new_emg_classify_normalized, separate=True)
        processed_old_real_data = Process_Fake_Data.reorderSmoothDataSet(old_real_emg_grids['grid_1'], filtering=False, modes=None)
        processed_new_real_data = Process_Fake_Data.reorderSmoothDataSet(new_real_emg_grids['grid_1'], filtering=False, modes=None)
        filtered_old_real_data = Post_Process_Data.spatialFilterModelInput(processed_old_real_data, kernel=self.classifier_filter_kernel)
        filtered_new_real_data = Post_Process_Data.spatialFilterModelInput(processed_new_real_data, kernel=self.classifier_filter_kernel)
        del old_real_emg_grids, new_real_emg_grids, processed_old_real_data, processed_new_real_data

        # classification
        gen_results = None  # no gan model blending factor results needed here
        # use old data to train old model
        basis_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, self.window_parameters)
        old_train_set, shuffled_train_set = basis_evaluation.classifierTrainSet(filtered_old_real_data, dataset='cross_validation_set')
        models_basis, model_result_basis = basis_evaluation.trainClassifier(shuffled_train_set, num_epochs=50, batch_size=64, decay_epochs=20)
        accuracy_basis, cm_recall_basis = basis_evaluation.evaluateClassifyResults(model_result_basis)
        # use new data to train new model
        best_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, self.window_parameters)
        new_train_set, shuffled_train_set = best_evaluation.classifierTrainSet(filtered_new_real_data, dataset='cross_validation_set')
        models_best, model_result_best = best_evaluation.trainClassifier(shuffled_train_set, num_epochs=50, batch_size=64, decay_epochs=20)
        accuracy_best, cm_recall_best = best_evaluation.evaluateClassifyResults(model_result_best)  # training and testing data from the same time
        # using old model to classify new data
        test_results = basis_evaluation.testClassifier(models_basis, shuffled_train_set)
        accuracy_worst, cm_recall_worst = basis_evaluation.evaluateClassifyResults(test_results)  # training and testing data from different time
        del filtered_old_real_data, filtered_new_real_data, old_train_set, new_train_set, shuffled_train_set

        return models_basis, accuracy_basis, cm_recall_basis, accuracy_best, cm_recall_best, accuracy_worst, cm_recall_worst


    ##  train classifier (on old data), for testing gan generation performance
    def trainClassifierOldData(self, old_emg_classify_normalized, extracted_emg_classify, gen_results):
        old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, self.window_parameters)
        synthetic_old_data, fake_old_images = old_evaluation.generateFakeData(extracted_emg_classify, 'old', self.modes_generation,
            old_emg_classify_normalized, spatial_filtering=False, kernel=self.gan_filter_kernel)
        # separate and store grids in a list if only use one grid later
        old_fake_emg_grids = Post_Process_Data.separateEmgGrids(synthetic_old_data, separate=True)
        old_real_emg_grids = Post_Process_Data.separateEmgGrids(old_emg_classify_normalized, separate=True)
        del synthetic_old_data
        # only preprocess selected grid
        processed_old_fake_data = Process_Fake_Data.reorderSmoothDataSet(old_fake_emg_grids['grid_1'], filtering=False, modes=self.modes_generation)
        processed_old_real_data = Process_Fake_Data.reorderSmoothDataSet(old_real_emg_grids['grid_1'], filtering=False, modes=None)
        del old_fake_emg_grids, old_real_emg_grids

        # select representative fake data for classification model training
        filtered_old_fake_data = Post_Process_Data.spatialFilterModelInput(processed_old_fake_data, kernel=self.classifier_filter_kernel)
        filtered_old_real_data = Post_Process_Data.spatialFilterModelInput(processed_old_real_data, kernel=self.classifier_filter_kernel)
        # select representative fake data for classification model training
        selected_old_fake_data = Dtw_Similarity.extractFakeData(filtered_old_fake_data, filtered_old_real_data, self.modes_generation,
            envelope_frequency=None, num_sample=100, num_reference=1, method='select', random_reference=False, split_grids=True)
        del processed_old_fake_data, processed_old_real_data, filtered_old_fake_data

        # classification
        train_set, shuffled_train_set = old_evaluation.classifierTrainSet(selected_old_fake_data['fake_data_based_on_grid_1'],
            dataset='cross_validation_set')
        models_old, model_result_old = old_evaluation.trainClassifier(shuffled_train_set, num_epochs=50, batch_size=64, decay_epochs=20)
        acc_old, cm_old = old_evaluation.evaluateClassifyResults(model_result_old)
        # test classifier
        test_set, shuffled_test_set = old_evaluation.classifierTestSet(self.modes_generation, filtered_old_real_data, train_set, test_ratio=1)
        test_results = old_evaluation.testClassifier(models_old, shuffled_test_set)
        accuracy_old, cm_recall_old = old_evaluation.evaluateClassifyResults(test_results)
        del train_set, shuffled_train_set, test_set, shuffled_test_set

        return models_old, accuracy_old, cm_recall_old, selected_old_fake_data, filtered_old_real_data


    ##  train classifier (on new data), for evaluating the proposed method performance
    def trainClassifierNewData(self, new_emg_classify_normalized, extracted_emg_classify, gen_results, models_basis):
        new_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, self.window_parameters)
        synthetic_new_data, fake_new_images = new_evaluation.generateFakeData(extracted_emg_classify, 'new', self.modes_generation,
            new_emg_classify_normalized, spatial_filtering=False, kernel=self.gan_filter_kernel)
        # separate and store grids in a list if only use one grid later
        new_fake_emg_grids = Post_Process_Data.separateEmgGrids(synthetic_new_data, separate=True)
        new_real_emg_grids = Post_Process_Data.separateEmgGrids(new_emg_classify_normalized, separate=True)
        del synthetic_new_data
        # only preprocess selected grid
        processed_new_fake_data = Process_Fake_Data.reorderSmoothDataSet(new_fake_emg_grids['grid_1'], filtering=False, modes=self.modes_generation)
        processed_new_real_data = Process_Fake_Data.reorderSmoothDataSet(new_real_emg_grids['grid_1'], filtering=False, modes=None)
        del new_fake_emg_grids, new_real_emg_grids

        # post-process dataset for model input
        filtered_new_fake_data = Post_Process_Data.spatialFilterModelInput(processed_new_fake_data, kernel=self.classifier_filter_kernel)
        filtered_new_real_data = Post_Process_Data.spatialFilterModelInput(processed_new_real_data, kernel=self.classifier_filter_kernel)
        # select representative fake data for classification model training
        selected_new_fake_data = Dtw_Similarity.extractFakeData(filtered_new_fake_data, filtered_new_real_data, self.modes_generation,
            envelope_frequency=None, num_sample=100, num_reference=1, method='select', random_reference=False, split_grids=True)

        # classification
        reference_indices = selected_new_fake_data['reference_index_based_on_grid_1']
        reference_new_real_data, adjusted_new_real_data = new_evaluation.addressReferenceData(reference_indices, filtered_new_real_data)
        train_set, shuffled_train_set = new_evaluation.classifierTlTrainSet(selected_new_fake_data['fake_data_based_on_grid_1'],
            reference_new_real_data, dataset='cross_validation_set', minimum_train_number=10)
        models_new, model_results_new = new_evaluation.trainTlClassifier(models_basis, shuffled_train_set, num_epochs=30, batch_size=32,
            decay_epochs=10)
        acc_new, cm_new = new_evaluation.evaluateClassifyResults(model_results_new)
        # test classifier
        test_set, shuffled_test_set = new_evaluation.classifierTestSet(self.modes_generation, adjusted_new_real_data, train_set, test_ratio=1)
        test_results = new_evaluation.testClassifier(models_new, shuffled_test_set)
        accuracy_new, cm_recall_new = new_evaluation.evaluateClassifyResults(test_results)
        del train_set, shuffled_train_set, test_set, shuffled_test_set

        return models_new, accuracy_new, cm_recall_new, selected_new_fake_data, adjusted_new_real_data, reference_new_real_data, processed_new_real_data


    ##  train classifier (on old and new data), for comparison purpose
    def trainClassifierMixData(self, old_emg_classify_normalized, new_emg_classify_normalized, reference_new_real_data, adjusted_new_real_data, models_basis):
        # build training data
        old_emg_for_replacement = {modes[2]: old_emg_classify_normalized[modes[2]] for transition_type, modes in self.modes_generation.items()}
        mix_old_new_data = Process_Fake_Data.replaceUsingFakeEmg(old_emg_for_replacement, new_emg_classify_normalized)
        # separate and store grids in a list if only use one grid later
        mix_old_new_grids = Post_Process_Data.separateEmgGrids(mix_old_new_data, separate=True)

        # only preprocess selected grid
        processed_mix_data = Process_Fake_Data.reorderSmoothDataSet(mix_old_new_grids['grid_1'], filtering=False, modes=self.modes_generation)
        # spatial filtering
        filtered_mix_data = Post_Process_Data.spatialFilterModelInput(processed_mix_data, kernel=self.classifier_filter_kernel)
        del mix_old_new_data, mix_old_new_grids

        # classification
        gen_results = None
        mix_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, self.window_parameters)
        # use the same reference new data above as the available new data for the mix dataset, to adjust both the train_set and test_set
        train_set, shuffled_train_set = mix_evaluation.classifierTlTrainSet(filtered_mix_data, reference_new_real_data,
            dataset='cross_validation_set')
        models_compare, model_results_compare = mix_evaluation.trainTlClassifier(models_basis, shuffled_train_set, num_epochs=30,
            batch_size=32, decay_epochs=10)
        acc_compare, cm_compare = mix_evaluation.evaluateClassifyResults(model_results_compare)
        # test classifier
        test_set, shuffled_test_set = mix_evaluation.classifierTestSet(self.modes_generation, adjusted_new_real_data, train_set, test_ratio=1)
        test_results = mix_evaluation.testClassifier(models_compare, shuffled_test_set)
        accuracy_compare, cm_recall_compare = mix_evaluation.evaluateClassifyResults(test_results)
        del train_set, shuffled_train_set, test_set, shuffled_test_set

        return accuracy_compare, cm_recall_compare, models_compare, filtered_mix_data


    ## train classifier (on old and fake new data), for improvement purpose
    def trainClassifierCombineData(self, selected_new_fake_data, filtered_mix_data, reference_new_real_data, adjusted_new_real_data, models_basis):
        ## build training data
        gen_results = None
        combine_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, self.window_parameters)
        new_fake_train_set, _ = combine_evaluation.classifierTlTrainSet(selected_new_fake_data['fake_data_based_on_grid_1'],
            reference_new_real_data, dataset='cross_validation_set')
        old_real_train_set, _ = combine_evaluation.classifierTlTrainSet(filtered_mix_data, reference_new_real_data,
            dataset='cross_validation_set')
        # select 40% data from new_fake_data and combine these selected data with old_real_data for plotting purpose
        new_fake_data = selected_new_fake_data['fake_data_based_on_grid_1']
        selected_fake_data = {key: random.sample(new_fake_data[key], int(len(new_fake_data[key]) * 0.4)) for key in new_fake_data}
        filtered_combined_data = {key: selected_fake_data[key] + filtered_mix_data[key] for key in filtered_mix_data}

        # classification
        train_set, shuffled_train_set = combine_evaluation.combineTrainingSet(new_fake_train_set, old_real_train_set)
        models_combine, model_results_combine = combine_evaluation.trainTlClassifier(models_basis, shuffled_train_set, num_epochs=30,
            batch_size=32, decay_epochs=10)
        acc_combine, cm_combine = combine_evaluation.evaluateClassifyResults(model_results_combine)
        # test classifier
        test_set, shuffled_test_set = combine_evaluation.classifierTestSet(self.modes_generation, adjusted_new_real_data, train_set,
            test_ratio=1)
        test_results = combine_evaluation.testClassifier(models_combine, shuffled_test_set)
        accuracy_combine, cm_recall_combine = combine_evaluation.evaluateClassifyResults(test_results)
        del new_fake_train_set, old_real_train_set, train_set, shuffled_train_set, test_set, shuffled_test_set

        return accuracy_combine, cm_recall_combine, models_combine, filtered_combined_data


    ## train classifier (on noisy new data), select some reference new data and augment them with noise for training comparison
    def trainClassifierNoiseData(self, processed_new_real_data, reference_new_real_data, adjusted_new_real_data, models_basis):
        # generate noise data
        gen_results = None
        noise_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, self.window_parameters)  # window_parameters are in line with the above
        noise_new_data = noise_evaluation.generateNoiseData(processed_new_real_data, reference_new_real_data, num_sample=100, snr=0.05)
        # median filtering
        filtered_noise_data = Post_Process_Data.spatialFilterModelInput(noise_new_data, kernel=self.classifier_filter_kernel)

        # classification
        # use the same reference new data above as the available new data for the mix dataset, to adjust both the train_set and test_set
        train_set, shuffled_train_set = noise_evaluation.classifierTlTrainSet(filtered_noise_data, reference_new_real_data,
            dataset='cross_validation_set')
        models_noise, model_results_noise = noise_evaluation.trainTlClassifier(models_basis, shuffled_train_set, num_epochs=30,
            batch_size=32, decay_epochs=10)
        acc_noise, cm_noise = noise_evaluation.evaluateClassifyResults(model_results_noise)
        # test classifier
        test_set, shuffled_test_set = noise_evaluation.classifierTestSet(self.modes_generation, adjusted_new_real_data, train_set, test_ratio=1)
        test_results = noise_evaluation.testClassifier(models_noise, shuffled_test_set)
        accuracy_noise, cm_recall_noise = noise_evaluation.evaluateClassifyResults(test_results)
        del train_set, shuffled_train_set, test_set, shuffled_test_set

        return accuracy_noise, cm_recall_noise, models_noise, filtered_noise_data

    ## plotting fake and real emg data for comparison
    def plotEmgData(self,fake_emg_data, real_emg_data, plot_ylim, title):
        # calculate average values
        fake_data = Plot_Emg_Data.calcuAverageEmgValues(fake_emg_data)
        real_data = Plot_Emg_Data.calcuAverageEmgValues(real_emg_data)
        # plot values of certain transition type
        transition_type = 'emg_LWSA'
        modes = self.modes_generation[transition_type]
        Plot_Emg_Data.plotMultipleEventMeanValues(fake_data, real_data, modes, title=title, ylim=(0, plot_ylim))
        transition_type = 'emg_SALW'
        modes = self.modes_generation[transition_type]
        Plot_Emg_Data.plotMultipleEventMeanValues(fake_data, real_data, modes, title=title, ylim=(0, plot_ylim))
        transition_type = 'emg_LWSD'
        modes = self.modes_generation[transition_type]
        Plot_Emg_Data.plotMultipleEventMeanValues(fake_data, real_data, modes, title=title, ylim=(0, plot_ylim))
        transition_type = 'emg_SDLW'
        modes = self.modes_generation[transition_type]
        Plot_Emg_Data.plotMultipleEventMeanValues(fake_data, real_data, modes, title=title, ylim=(0, plot_ylim))

