## import modules
import tensorflow as tf

##
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use

## load trained model
fold = 5
model = []
for number in range(fold):
    model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models\cross_validation_set_{number}'
    model.append(tf.keras.models.load_model(model_dir))  # display model weight parameters

