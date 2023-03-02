## import
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt
all_subjects = {}  # save all subject results


## model set
subject = 'Number5'
version = 0
model_type = 'Losing_Cnn'
all_subjects[subject] = {}


##
# channel_random_lost = ['channel_random_lost_5', 'channel_random_lost_6', 'channel_random_lost_10', 'channel_random_lost_15']
# channel_random_lost_recovered = ['channel_random_lost_5_recovered', 'channel_random_lost_6_recovered', 'channel_random_lost_10_recovered', 'channel_random_lost_15_recovered']
channel_corner_lost = ['channel_corner_lost_5', 'channel_corner_lost_6', 'channel_corner_lost_10', 'channel_corner_lost_15']
channel_corner_lost_recovered = ['channel_corner_lost_5_recovered', 'channel_corner_lost_6_recovered', 'channel_corner_lost_10_recovered', 'channel_corner_lost_13_recovered']
# data_set = {'channel_random_lost': channel_random_lost, 'channel_corner_lost': channel_corner_lost,
#     'channel_random_lost_recovered': channel_random_lost_recovered, 'channel_corner_lost_recovered': channel_corner_lost_recovered}
data_set = {'channel_corner_lost': channel_corner_lost, 'channel_corner_lost_recovered': channel_corner_lost_recovered}
for key, value in data_set.items():
    all_subjects[subject][key] = {}
    for dataset in value:
        model_results = Sliding_Ann_Results.getPredictResults(subject, version, dataset, model_type)
        all_subjects[subject][key][dataset] = model_results


##  print all accuracy results
# subject = 'Shibo'
# accuracy = all_subjects[f'{subject}']['accuracy']
accuracy = all_subjects['Number5']['channel_corner_lost_recovered']['channel_corner_lost_13_recovered']['accuracy']
predict_window_increment_ms = 20
x_label = [i * predict_window_increment_ms for i in range(len(accuracy))]
y_value = [round(i, 1) for i in list(accuracy.values())]

fig, ax = plt.subplots()
bars = ax.bar(range(len(accuracy)), y_value)
ax.set_xticks(range(len(accuracy)), x_label)
ax.bar_label(bars)
ax.set_ylim([40, 100])
ax.set_xlabel('prediction time delay(ms)')
ax.set_ylabel('prediction accuracy(%)')
plt.title('Prediction accuracy for the prediction at different delay points')
