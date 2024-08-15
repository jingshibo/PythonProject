##
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing

##
subject = 'Number1'
feature_set = 'HDsEMG'
interp_emg_features = Emg_Preprocessing.readInterpFeatures(subject, feature_set)