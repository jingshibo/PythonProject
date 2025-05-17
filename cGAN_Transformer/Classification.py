

##
from Conditional_GAN.Data_Procesing import Process_Raw_Data, Train_Classifiers, Train_cGan
from cGAN_Transformer.Functions import Preprocessing, Results
from cGAN_Transformer.Models import Classification_Model
import datetime


## load and filter data
subject = 'Number1'
grid = 'grid_1'
version = 0  # the data from which experiment version to process
up_down_session_t0 = [0, 1, 2, 3, 4]
down_up_session_t0 = [1, 2, 3, 4, 5]
up_down_session_t1 = [0, 1, 2, 3, 4]
down_up_session_t1 = [1, 2, 3, 4, 5]
old_emg_data, new_emg_data, window_parameters, start_before_toeoff_ms = Train_cGan.realEmgData(subject, version, up_down_session_t0,
    down_up_session_t0, up_down_session_t1, down_up_session_t1, grid=grid, envelope=True, envelope_cutoff=400, reordering=False)


## parameters for extracting emg data to train models
range_limit = 1500
spatial_filter_kernel = (2, 1)
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'], 'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']} # The order in each list is important, corresponding to gen_data_1 and gen_data_2.
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']  # the length of data in each repetition
old_emg_normalized, new_emg_normalized, _, _ = Process_Raw_Data.normalizeFilterEmgData(old_emg_data, new_emg_data, range_limit,
    normalize='(0,1)', spatial_filter=False, kernel=spatial_filter_kernel)
time_length = 500  # select only 1200ms data around toe-off
classify_old_emg, classify_new_emg, train_gan_data = Preprocessing.extractGanTrainingData(modes_generation, old_emg_normalized, new_emg_normalized, time_length)
fold_number = 5
cross_validation_indices, cross_validation_dataset = Preprocessing.crossValidationSet(fold_number, classify_old_emg)


##  classify using a single cnn 2d model
num_epochs = 50
batch_size = 64
decay_epochs = 20
now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size)
train_model = Classification_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(classify_old_emg, cross_validation_indices, decay_epochs)
print(datetime.datetime.now() - now)


##
accuracy, cm_recall = Results.getAccuracyCm(model_results)

# class_labels = ['LW', 'LWSA', 'LWSD', 'SALW', 'SA', 'SDLW', 'SD']
# Confusion_Matrix.plotConfusionMatrix(cm_recall, class_labels, normalize=False)