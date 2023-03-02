## import
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt
all_subjects = {}  # save all subject results


## model set
subject = 'Number5'
version = 0
model_type = 'Reduced_Cnn'
all_subjects[subject] = {}


##
reduce_area_dataset = ['channel_area_35', 'channel_area_25', 'channel_area_15', 'channel_area_6']
reduce_density_dataset = ['channel_density_33', 'channel_density_21', 'channel_density_11', 'channel_density_8']
reduce_muscle_dataset = ['channel_muscle_hdemg1', 'channel_muscle_hdemg2', 'channel_muscle_bipolar', 'channel_muscle_bipolar1', 'channel_muscle_bipolar2']
data_set = {'reduce_area_dataset': reduce_area_dataset, 'reduce_density_dataset': reduce_density_dataset,
    'reduce_muscle_dataset': reduce_muscle_dataset}
for key, value in data_set.items():
    all_subjects[subject][key] = {}
    for dataset in value:
        model_results = Sliding_Ann_Results.getPredictResults(subject, version, dataset, model_type)
        all_subjects[subject][key][dataset] = model_results


##  print all accuracy results
# subject = 'Shibo'
# accuracy = all_subjects[f'{subject}']['accuracy']
accuracy = all_subjects['Number5']['reduce_muscle_dataset']['channel_muscle_bipolar2']['accuracy']
predict_window_increment_ms = 20
x_label = [i * predict_window_increment_ms for i in range(len(accuracy))]
y_value = [round(i, 1) for i in list(accuracy.values())]

fig, ax = plt.subplots()
bars = ax.bar(range(len(accuracy)), y_value)
ax.set_xticks(range(len(accuracy)), x_label)
ax.bar_label(bars)
ax.set_ylim([60, 100])
ax.set_xlabel('prediction time delay(ms)')
ax.set_ylabel('prediction accuracy(%)')
plt.title('Prediction accuracy for the prediction at different delay points')
