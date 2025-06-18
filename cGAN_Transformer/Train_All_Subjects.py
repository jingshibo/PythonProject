##
from Conditional_GAN.Data_Procesing import Process_Raw_Data, Train_Classifiers, Train_cGan
from cGAN_Transformer.Functions import Preprocessing, Results, Storage, Plot_Raw_Data
from cGAN_Transformer.Models import Classification_Model, Classification_TL_Model, Classification_Test, Transformer_GAN_Training, Transformer_GAN_Testing
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
from Conditional_GAN.Data_Procesing import Dtw_Similarity
import numpy as np
import datetime
import copy


'''train generative model'''
## subject information
subjects = {
    'Number0': {'grid': 'grid_2', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [0, 1, 2, 3, 4], 'down_up_session_t0': [0, 1, 2, 5, 6], 'up_down_session_t1': [5, 6, 7, 8, 9],
        'down_up_session_t1': [4, 5, 6, 8, 9], },
    'Number1': {'grid': 'grid_1', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [0, 1, 2, 3, 4], 'down_up_session_t0': [1, 2, 3, 4, 5], 'up_down_session_t1': [0, 1, 2, 3, 4],
        'down_up_session_t1': [1, 2, 3, 4], },
    'Number2': {'grid': 'grid_1', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [5, 6, 7, 8, 9], 'down_up_session_t0': [6, 8, 9, 10], 'up_down_session_t1': [5, 6, 7, 8, 9],
        'down_up_session_t1': [6, 7, 8, 9, 10], },
    'Number3': {'grid': 'grid_1', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [0, 1, 2, 3, 4], 'down_up_session_t0': [0, 1, 2, 3, 4], 'up_down_session_t1': [0, 1, 2, 3, 4, 5],
        'down_up_session_t1': [1, 2, 3, 4, 5], },
    'Number4': {'grid': 'grid_1', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [0, 1, 2, 4, 5], 'down_up_session_t0': [0, 1, 2, 3, 4], 'up_down_session_t1': [0, 1, 2, 3, 4],
        'down_up_session_t1': [0, 1, 3, 4], },
    'Number5': {'grid': 'grid_1', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [0, 1, 2, 3, 4], 'down_up_session_t0': [0, 1, 2, 3, 4], 'up_down_session_t1': [0, 1, 2, 3, 4],
        'down_up_session_t1': [0, 1, 2, 3, 4], },
    'Number6': {'grid': 'grid_1', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [0, 1, 2, 3, 4], 'down_up_session_t0': [2, 3, 4, 5], 'up_down_session_t1': [0, 1, 2, 3, 4],
        'down_up_session_t1': [0, 1, 2, 3, 4], },
    'Number7': {'grid': 'grid_1', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [0, 1, 2, 3, 4], 'down_up_session_t0': [0, 1, 2, 3, 4], 'up_down_session_t1': [0, 1, 2, 3],
        'down_up_session_t1': [0, 1, 2, 3], },
    'Number8': {'grid': 'grid_2', 'version': 0,  # the data from which experiment version to process
        'up_down_session_t0': [5, 6, 7, 8, 9], 'down_up_session_t0': [7, 8, 9, 10], 'up_down_session_t1': [0, 1, 2, 3, 4],
        'down_up_session_t1': [0, 1, 2, 3], }, }
result_set = 0
class_labels = ['LW', 'LWSA', 'LWSD', 'SALW', 'SA', 'SDLW', 'SD']
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'],
    'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}  # The order in each list is important, corresponding to gen_data_1 and gen_data_2.
transition_encoding = {"emg_LWSA": 0, "emg_LWSD": 1, "emg_SALW": 2, "emg_SDLW": 3}  # encode the conditions into integer


## loop over subjects
for subject, data_selection in subjects.items():
    version = data_selection['version']
    old_emg_data, new_emg_data, window_parameters, start_before_toeoff_ms = Train_cGan.realEmgData(subject, version,
        data_selection['up_down_session_t0'], data_selection['down_up_session_t0'], data_selection['up_down_session_t1'],
        data_selection['down_up_session_t1'], grid=data_selection['grid'], envelope=True, envelope_cutoff=400, reordering=False)

    ## parameters for normalize emg data to train models
    amplitude_limit = 1500
    length = window_parameters['start_before_toeoff_ms'] + window_parameters[
        'endtime_after_toeoff_ms']  # the length of data in each repetition
    old_emg_normalized, new_emg_normalized, _, _ = Process_Raw_Data.normalizeFilterEmgData(old_emg_data, new_emg_data, amplitude_limit,
        normalize='(0,1)', spatial_filter=False, kernel=(2, 1))
    old_emg_aligned, old_lags_butter = Preprocessing.align_by_cross_correlation(old_emg_normalized, max_lag=100, num_iterations=3,
        initial_reference_method='average', verbose=False, butter_cutoff_freq=20, butter_filter_order=4)
    new_emg_aligned, new_lags_butter = Preprocessing.align_by_cross_correlation(new_emg_normalized, max_lag=100, num_iterations=3,
        initial_reference_method='average', verbose=False, butter_cutoff_freq=20, butter_filter_order=4)

    ## parameters for extracting emg data to train models
    window_length = 1200
    window_increment = 100
    num_window_per_transition = 1
    window_shift = 0
    channel_shift = 15
    start = - window_length - window_shift
    end = window_increment * num_window_per_transition + window_shift
    time_range = [-600, 600]  # select data centered around toe-off
    old_emg_central, new_emg_central, train_gan_data, new_gan_data = Preprocessing.extractGanTrainingData(modes_generation, old_emg_aligned,
        new_emg_aligned, time_range)

    ## gan model train parameters
    NUM_EPOCH_TO_TRAIN = 100
    num_batch_per_epoch = 50
    num_sample_per_condition = 5  # in each batch
    num_transitions = len(transition_encoding)
    BATCH_SIZE = num_sample_per_condition * num_transitions * num_window_per_transition
    model_type = 'Transformer_Gan'
    model_name = ['gen', 'disc']
    training_parameters = {'modes_generation': modes_generation, 'transition_encoding': transition_encoding,
        'num_epochs': NUM_EPOCH_TO_TRAIN, 'num_batch_per_epoch': num_batch_per_epoch, 'num_sample_per_condition': num_sample_per_condition,
        'window_length': window_length, 'window_increment': window_increment, 'num_window_per_transition': num_window_per_transition,
        'window_shift': window_shift, 'channel_shift': channel_shift}
    storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name, 'gan_result_set': 0}
    # trainer = Transformer_GAN_Training.GanTraining(NUM_EPOCH_TO_TRAIN, num_sample_per_condition, num_batch_per_epoch)
    # gan_model = trainer.trainModel(train_gan_data, transition_encoding, training_parameters, storage_parameters)

    ## generate and sample transition data
    epoch_number_to_generate = 70
    gan_model = Storage.loadCheckPointModels(storage_parameters, epoch_number_to_generate)
    all_generated_data = Transformer_GAN_Testing.generateTransitionData(gan_model['gen'], train_gan_data, transition_encoding,
        num_window_per_transition, window_length, window_increment, window_shift, number_to_generate=3600, batch_size=30)
    time_0_fake_data, ordered_sampled_data = Transformer_GAN_Testing.returnDataForPlotting(all_generated_data, ordered_sample_number=50)
    # only generate old synthetic data for old model training, since new synthetic data will be generated on-the-fly when required
    extracted_data = Dtw_Similarity.extractFakeData(time_0_fake_data, old_emg_central, modes_generation, envelope_frequency=30,
        num_sample=50, num_reference=1, method='random', random_reference=False, split_grids=True)
    selected_fake_data, organized_fake_data = Transformer_GAN_Testing.fakeDataForTraining(extracted_data)


    '''  
        train old classifier (old model scenarios), training and testing using old data   
    '''
    ##  train the old model using real old transition data
    _, original_dataset = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data, modes_generation,
        n_splits=5, n_real_steady_state=50, n_synthetic_transition=0, n_real_transition=50, random_sampling=False)
    train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
    models_old_real, results_old_real = train_model.trainModel(original_dataset, decay_epochs=20)
    accuracy_old_real, cm_recall_old_real = Results.getAccuracyCm(results_old_real)
    # Confusion_Matrix.plotConfusionMatrix(cm_recall_old_real, class_labels, normalize=False)
    Storage.saveClassifyResult(subject, accuracy_old_real, cm_recall_old_real, version, result_set, 'classify_old_real', num_reference=None)

    ##  train the old model using synthetic transition data
    synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data, modes_generation,
        n_splits=5, n_real_steady_state=50, n_synthetic_transition=50, n_real_transition=0, random_sampling=False)
    train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
    models_old_synthetic, results_old_synthetic = train_model.trainModel(synthetic_dataset, decay_epochs=20)
    accuracy_old_synthetic, cm_recall_old_synthetic = Results.getAccuracyCm(results_old_synthetic)
    # Confusion_Matrix.plotConfusionMatrix(cm_recall_old_synthetic, class_labels, normalize=False)
    Storage.saveClassifyResult(subject, accuracy_old_synthetic, cm_recall_old_synthetic, version, result_set, 'classify_old_synthetic',
        num_reference=None)

    ## train the old model with old data and noisy data
    synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data, modes_generation,
        n_splits=5, n_real_steady_state=50, n_synthetic_transition=0, n_real_transition=5, random_sampling=False)
    train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
    models_old_imbalance, results_old_imbalance = train_model.trainModel(synthetic_dataset, decay_epochs=20)
    accuracy_old_imbalance, cm_recall_old_imbalance = Results.getAccuracyCm(results_old_imbalance)
    # Confusion_Matrix.plotConfusionMatrix(cm_recall_old_imbalance, class_labels, normalize=False)
    Storage.saveClassifyResult(subject, accuracy_old_imbalance, cm_recall_old_imbalance, version, result_set, 'classify_old_imbalance',
        num_reference=5)

    ## train the old model with old data and noisy data
    synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data, modes_generation,
        n_splits=5, n_real_steady_state=50, n_synthetic_transition=50, n_real_transition=5, random_sampling=False)
    train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
    models_old_rebalanced, results_old_rebalanced = train_model.trainModel(synthetic_dataset, decay_epochs=20)
    accuracy_old_rebalanced, cm_recall_old_rebalanced = Results.getAccuracyCm(results_old_rebalanced)
    # Confusion_Matrix.plotConfusionMatrix(cm_recall_old_rebalanced, class_labels, normalize=False)
    Storage.saveClassifyResult(subject, accuracy_old_rebalanced, cm_recall_old_rebalanced, version, result_set, 'classify_old_rebalanced',
        num_reference=5)


    '''  
        train new classifier (basic updating scenarios), training and testing using data from the same and different time  
    '''
    ##  train old model using old data
    _, original_old_dataset = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data, modes_generation,
        n_splits=5)
    train_old_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
    models_basis, results_basis = train_old_model.trainModel(original_old_dataset, decay_epochs=20)
    accuracy_basis, cm_recall_basis = Results.getAccuracyCm(results_basis)
    # Confusion_Matrix.plotConfusionMatrix(cm_recall_basis, class_labels, normalize=False)
    Storage.saveClassifyResult(subject, accuracy_basis, cm_recall_basis, version, result_set, 'classify_basis', num_reference=None)

    ##  train new model using new data
    _, original_new_dataset = Preprocessing.build_cv_dataset_with_augmented_data(new_emg_central, organized_fake_data, modes_generation,
        n_splits=5)
    train_new_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
    models_best, results_best = train_new_model.trainModel(original_new_dataset, decay_epochs=20)
    accuracy_best, cm_recall_best = Results.getAccuracyCm(results_best)
    # Confusion_Matrix.plotConfusionMatrix(cm_recall_best, class_labels, normalize=False)
    Storage.saveClassifyResult(subject, accuracy_best, cm_recall_best, version, result_set, 'classify_best', num_reference=None)

    ## old model to classify new data
    _, original_new_dataset = Preprocessing.build_cv_dataset_with_augmented_data(new_emg_central, organized_fake_data, modes_generation,
        n_splits=5)
    test_models = Classification_Test.ModelTesting(models_basis, batch_size=32)
    test_results = test_models.testModel(original_new_dataset)
    accuracy_worst, cm_recall_worst = Results.getAccuracyCm(test_results)
    # Confusion_Matrix.plotConfusionMatrix(cm_recall_worst, class_labels, normalize=False)
    Storage.saveClassifyResult(subject, accuracy_worst, cm_recall_worst, version, result_set, 'classify_worst', num_reference=None)

    ## update old model with new data (10 per mode) to classify new data
    synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_noisy_data(new_emg_central, modes_generation, snr=None, n_splits=5,
        n_real_steady_state=10, n_synthetic_transition=0, n_real_transition=10, random_sampling=False)
    train_model = Classification_TL_Model.ModelTraining(models_basis, num_epochs=30, batch_size=8, report_period=10)
    models_tf, results_tf = train_model.trainModel(synthetic_dataset, decay_epochs=10)
    accuracy_tf, cm_recall_tf = Results.getAccuracyCm(results_tf)
    Storage.saveClassifyResult(subject, accuracy_tf, cm_recall_tf, version, result_set, 'classify_tf', num_reference=10)


    '''  
        train new classifier (with data augmentation), training and testing using data augmented by different methods  
    '''
    for num_reference in [0, 1, 3, 5]:
        ## update the old model with new data and old data (the n_synthetic_transition here is actually old transition data)
        synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_augmented_data(new_emg_central, old_emg_central, modes_generation,
            n_splits=5, n_real_steady_state=10, n_synthetic_transition=10, n_real_transition=num_reference, random_sampling=False)
        train_model = Classification_TL_Model.ModelTraining(models_basis, num_epochs=30, batch_size=8, report_period=10)
        models_old, results_old = train_model.trainModel(synthetic_dataset, decay_epochs=10)
        accuracy_old, cm_recall_old = Results.getAccuracyCm(results_old)
        # Confusion_Matrix.plotConfusionMatrix(cm_recall_old, class_labels, normalize=False)
        Storage.saveClassifyResult(subject, accuracy_old, cm_recall_old, version, result_set, 'classify_with_old',
            num_reference=num_reference)

        ## update the old model with new data and synthetic data
        synthetic_dataset, _ = Preprocessing.build_cv_dataset_for_model_updating(new_emg_central, gan_model, modes_generation,
            training_parameters, n_splits=5, n_real_steady_state=10, n_synthetic_transition=10, n_real_transition=num_reference,
            random_sampling=False)
        train_model = Classification_TL_Model.ModelTraining(models_basis, num_epochs=30, batch_size=8, report_period=10)
        models_synthetic, results_synthetic = train_model.trainModel(synthetic_dataset, decay_epochs=10)
        accuracy_synthetic, cm_recall_synthetic = Results.getAccuracyCm(results_synthetic)
        # Confusion_Matrix.plotConfusionMatrix(cm_recall_synthetic, class_labels, normalize=False)
        Storage.saveClassifyResult(subject, accuracy_synthetic, cm_recall_synthetic, version, result_set, 'classify_with_synthetic',
            num_reference=num_reference)

        ## update the old model with new data and noisy data
        if num_reference != 0:  # only applies when num_reference is greater than 0
            for snr in [None, 25]:
                synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_noisy_data(new_emg_central, modes_generation, snr=snr, n_splits=5,
                    n_real_steady_state=10, n_synthetic_transition=10, n_real_transition=num_reference, random_sampling=False)
                train_model = Classification_TL_Model.ModelTraining(models_basis, num_epochs=30, batch_size=8, report_period=10)
                models_noise, results_noise = train_model.trainModel(synthetic_dataset, decay_epochs=10)
                accuracy_noise, cm_recall_noise = Results.getAccuracyCm(results_noise)
                # Confusion_Matrix.plotConfusionMatrix(cm_recall_noise, class_labels, normalize=False)
                if snr:
                    Storage.saveClassifyResult(subject, accuracy_noise, cm_recall_noise, version, result_set, 'classify_with_noisy',
                        num_reference=num_reference)
                else:
                    Storage.saveClassifyResult(subject, accuracy_noise, cm_recall_noise, version, result_set, 'classify_with_copy',
                        num_reference=num_reference)


