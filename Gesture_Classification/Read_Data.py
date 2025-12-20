##
import numpy as np
import pandas as pd
from Gesture_Classification import Functions, Model
from scipy.signal import savgol_filter
from Transition_Prediction.Pre_Processing.Utility_Functions import Feature_Calculation
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Bipolar_EMG.Models import Dataset_Model
from Transition_Prediction.Models.ANN.Functions import Ann_Dataset
import tensorflow as tf
from tinymlgen import port
import datetime


## Read data
subject = 'Number1'
mode = 'SD1_dataFile_003'


# Read IMU data
with open(f'D:\Data\Gesture_Classification\subject_{subject}\\raw_data\\IMU\{mode}.dat', 'rb') as file:
    # Read the entire file into a numpy array of type 'float32'
    arr = np.fromfile(file, dtype=np.float32)
    imu_data = pd.DataFrame(np.reshape(arr[:], newshape=(-1, 22)))

# ## check data loss
diff = np.diff(imu_data, axis=0)
rows_with_zero = np.where(diff[:, 0] != 1)[0]

# plot the imu value
Functions.plotImuData(imu_data, column=2, start_index=0, end_index=-1)
# Plot.plotImuData(imu_data, column=3, start_index=0, end_index=-1)
# Plot.plotImuData(imu_data, column=9, start_index=0, end_index=-1)
# Plot.plotImuData(imu_data, column=15, start_index=0, end_index=-1)


##
# extract data for each class (in order: rest, flat, fist, flexion, extension)
imu_classes = {}
modes = {'rest': [200, 2800], 'flat': [3200, 5800], 'fist': [6200, 8800], 'flexion': [9200, 11800], 'extension': [12200, 14800]}
# only accelerator
cols = list(range(3, 6)) + list(range(9, 12)) + list(range(15, 18))
for mode, time_slice in modes.items():
    imu_classes[mode] = imu_data.iloc[time_slice[0]: time_slice[1], cols]


## reorganize by sliding windows
window_size = 10  # 200ms
window_increment = 5  # 100ms

# separate data into windows
def createWindows(data, window_size, increment):
    windows_data = []
    for start in range(0, len(data) - window_size + 1, increment):
        end = start + window_size
        window = data.iloc[start:end]
        if len(window) == window_size:  # drop the last one with the size smaller than the window size
            windows_data.append(window.to_numpy())
    return windows_data

# filter each window as what is done in real-time processing
imu_filtered = {}
for mode, imu_value in imu_classes.items():
    imu_windowed = createWindows(imu_value, window_size, window_increment)
    imu_filtered[mode] = [savgol_filter(w, window_length=7, polyorder=2, axis=0, mode="mirror") for w in imu_windowed] # mirror match C code


## calculate features for each window data
imu_features = {}
for mode, imu_value in imu_filtered.items():
    imu_feature_list = []
    for imu_window_data in imu_value:
        imu_feature = Feature_Calculation.calcuImuFeatures(imu_window_data)
        imu_feature_list.append(imu_feature)
    imu_features[mode] = np.vstack(imu_feature_list).tolist()  # convert numpy to list for dict storage


## build training dataset
imu_cross_validation = Data_Preparation.crossValidationSet(5, imu_features, shuffle=False)
input_normalized = Functions.combineNormalizedDataset(imu_cross_validation)
input_shuffled_groups = Ann_Dataset.shuffleTrainingSet(input_normalized)


## classification
models, model_results = Model.classifyUsingAnnModel(input_shuffled_groups)


## predict MV results
predict_results, true_labels = Dataset_Model.reorganizePredictionResults(model_results)
predict_mv_results = [{mode: Dataset_Model.majority_vote(value, n=5) for mode, value in group_result.items()} for group_result in
    predict_results]
average_accuracy, average_cm_number, average_cm_recall = Dataset_Model.calculateAccuracy(predict_mv_results, true_labels)


## save model for deployment
deploy_model = models[0] # choose a model to deploy, e.g. group 0
deploy_model.export("D:\Data\Gesture_Classification\subject_Number1\\ann_mkr_zero")


##  Convert to TFLite Micro format
'''
    TensorFlow Lite Micro Method
'''
# converter = tf.lite.TFLiteConverter.from_keras_model(deploy_model)  # if you do not want to save the model
converter = tf.lite.TFLiteConverter.from_saved_model("D:\\Data\\Gesture_Classification\\subject_Number1\\ann_mkr_zero")

tflite_model_path="D:\Data\Gesture_Classification\subject_Number1\\ann_mkr_zero_int8.tflite"
train_x = input_shuffled_groups['group_0']['train_feature_x']
def representative_dataset():
    for i in range(min(500, train_x.shape[0])):
        x = train_x[i:i+1].astype(np.float32)  # shape (1, feature_dim)
        yield [x]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

## Test the TFLite model on PC (Just to make sure quantization didn’t destroy performance)
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example test
x = train_x[1000:1001]  # float32
# Quantize input
scale, zero_point = input_details[0]['quantization']
x_q = (x / scale + zero_point).astype(np.int8)

interpreter.set_tensor(input_details[0]['index'], x_q)
interpreter.invoke()
output_q = interpreter.get_tensor(output_details[0]['index'])

# Dequantize if needed:
out_scale, out_zero = output_details[0]['quantization']
output = (output_q.astype(np.float32) - out_zero) * out_scale
print(output)

## Check if the model is fully int8
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensor_details = interpreter.get_tensor_details()

print("Input dtype:", input_details[0]["dtype"])
print("Output dtype:", output_details[0]["dtype"])

# Check all tensors
dtypes = set()
for t in tensor_details:
    dtypes.add((t["name"], t["dtype"], t["quantization"]))

for name, dtype, quant in dtypes:
    print(name, dtype, "quant:", quant)

# Check which ops your model uses (this directly affects your TFLM resolver)
# If you see ops like RESHAPE, QUANTIZE, DEQUANTIZE, MUL, ADD, etc., you must register them in TFLM (or just use AllOpsResolver).
tf.lite.experimental.Analyzer.analyze(model_path=tflite_model_path)

## convert the .tflite model to C/C++ file
with open(tflite_model_path, "rb") as f:
    data = f.read()

array = np.frombuffer(data, dtype=np.uint8)

with open("D:\Data\Gesture_Classification\subject_Number1\\ann_mkr_zero_model_data.cc", "w") as f:
    f.write('#include <cstdint>\n\n')
    f.write('alignas(8) const unsigned char g_model[] = {\n  ')

    for i, val in enumerate(array):
        f.write(str(val))
        if i != len(array) - 1:
            f.write(', ')
        if (i + 1) % 12 == 0:
            f.write('\n  ')

    f.write('\n};\n\n')
    f.write(f'const int g_model_len = {len(array)};\n')


## from tflite to C code for EloquentTinyML
'''
    EloquentTinyML method
'''
c_code = port(deploy_model)
with open("D:\Data\Gesture_Classification\subject_Number1\gesture_ann_model.h", "w") as f:
    f.write(c_code)