## import modules
from Models.Utility_Functions import Data_Preparation
from Models.ANN.Functions import Ann_Dataset



## read emg data
# basic information
subject = "Number1"
version = 0  # which experiment data to process
feature_set = 0  # which feature set to use
fold = 5  # 5-fold cross validation


##  define windows
down_sampling = False
start_before_toeoff_ms = 450
endtime_after_toeoff_ms = 400
predict_window_ms = start_before_toeoff_ms
feature_window_ms = 350
sample_rate = 1 if down_sampling is True else 2
predict_window_size = predict_window_ms * sample_rate
feature_window_size = feature_window_ms * sample_rate
predict_window_increment_ms = 20
feature_window_increment_ms = 20
predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
predict_using_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate)) + 1
predict_window_per_repetition = int((endtime_after_toeoff_ms + start_before_toeoff_ms - predict_window_ms) / predict_window_increment_ms) + 1


## read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)  # only remove some modes here
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)

# get shuffled cross validation data set
feature_window_per_repetition = cross_validation_groups['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition
normalized_groups = Ann_Dataset.combineNormalizedDataset(cross_validation_groups, feature_window_per_repetition)
shuffled_groups = Ann_Dataset.shuffleTrainingSet(normalized_groups)


##
input_features = shuffled_groups['group_0']['test_feature_x']
input_labels = shuffled_groups['group_0']['test_int_y']


## process flatten cnn features
import numpy as np
indices = np.where(np.isin(input_labels, [4, 5, 6]))[0]
input_features = input_features[indices.tolist(), :]
category_labels = input_labels[indices.tolist()]


##
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_standardized = scaler.fit_transform(input_features)


## Choose the number of components you want to keep, e.g., 100
pca = PCA(3)
data_pca = pca.fit_transform(data_standardized)

## Check the explained variance ratio to see how much variance is explained by each component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance_ratio)

## calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print("Cumulative explained variance:", cumulative_explained_variance)


##
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
cmap = ListedColormap(['r', 'g', 'b'])  # define colors for 3 categories
x = data_pca[:, 0]
y = data_pca[:, 1]
z = data_pca[:, 2]
cat = category_labels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=cat, cmap=cmap, edgecolors="black")

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

colors = ['r', 'g', 'b']
labels = ['SALW', 'SASA', 'SASS']
handles = [plt.Line2D([], [], color=colors[i], marker='o', linestyle='', label=labels[i]) for i in range(len(labels))]
plt.legend(handles=handles)

plt.show()

