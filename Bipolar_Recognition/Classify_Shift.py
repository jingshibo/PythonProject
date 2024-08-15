##
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing, Manipulate_Channels
from Bipolar_EMG.Models import Dataset_Model
from Bipolar_Recognition.Models import Result_Processing
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Transition_Prediction.Models.ANN.Functions import Ann_Dataset
import copy


## read emg features
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
original_set = ['hdsemg_original', 'bipolar_original']
shift_set = ['hdsemg_original', 'bipolar_original', 'hdsemg_h_shift', 'bipolar_h_shift_1', 'bipolar_h_shift_3', 'bipolar_h_shift_4',
    'bipolar_h_shift_6', 'hdsemg_v_shift', 'bipolar_v_shift_1', 'bipolar_v_shift_3', 'bipolar_v_shift_4', 'bipolar_v_shift_6']


## construct dataset
subject = 'Number1'
original_features = 'bipolar_original'
shift_features = 'bipolar_h_shift_6'
emg_cross_validation = Manipulate_Channels.constructTrainingDataset(subject, original_features, shift_features)
# shuffle normalized dataset
emg_normalized = Dataset_Model.combineNormalizedDataset(emg_cross_validation)
emg_shuffled_groups = Ann_Dataset.shuffleTrainingSet(emg_normalized)


## classify using a single ann model
models, model_results = Dataset_Model.classifyUsingAnnModel(emg_shuffled_groups)


