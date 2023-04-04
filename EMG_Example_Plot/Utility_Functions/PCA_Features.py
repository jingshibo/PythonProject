
## import
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np
import torch


##  save example features
def saveFeatureExamples(subject, version, feature_example_set, features, labels, feature_type):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\comparison_data'
    data_file = f'subject_{subject}_Experiment_{version}_{feature_type}_features_set_{feature_example_set}.npy'
    data_path = os.path.join(data_dir, data_file)
    np.save(data_path, features)
    data_file = f'subject_{subject}_Experiment_{version}_{feature_type}_label_set_{feature_example_set}.npy'
    data_path = os.path.join(data_dir, data_file)
    np.save(data_path, labels)


##  load example features
def loadFeatureExamples(subject, version, feature_example_set, feature_type):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\comparison_data'
    data_file = f'subject_{subject}_Experiment_{version}_{feature_type}_features_set_{feature_example_set}.npy'
    data_path = os.path.join(data_dir, data_file)
    features = np.load(data_path)
    data_file = f'subject_{subject}_Experiment_{version}_{feature_type}_label_set_{feature_example_set}.npy'
    data_path = os.path.join(data_dir, data_file)
    labels = np.load(data_path)

    return features, labels


##  save the trained cnn models from 5 groups
def savePytorchModels(subject, version, models, model_type):
    # save only the parameters
    for group_number in range(5):
        data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_group_{group_number}.pth'
        model_path = os.path.join(data_dir, model_file)
        torch.save(models[group_number].state_dict(), model_path)

    # save the structure as well as parameters
    for group_number in range(5):
        data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_group_{group_number}.pickle'
        model_path = os.path.join(data_dir, model_file)
        torch.save(models[group_number], model_path)


##  load the trained cnn models from 5 groups
def loadPytorchModels(subject, version, model_type):
    # load only the parameters
    load_model_parameters = []
    for group_number in range(5):
        data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_group_{group_number}.pth'
        model_path = os.path.join(data_dir, model_file)
        load_model_parameters.append(torch.load(model_path))

    # load the structure as well as parameters
    load_models = []
    for group_number in range(5):
        data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_group_{group_number}.pickle'
        model_path = os.path.join(data_dir, model_file)
        load_models.append(torch.load(model_path))

    return load_model_parameters, load_models


##  pca calculation
def calculatePca(input_features, dimension=3):
    data_standardized = StandardScaler().fit_transform(input_features)  # standardize the features
    pca = PCA(dimension)  ## Choose the number of components you want to keep, e.g., 100. it can also be left empty
    data_pca = pca.fit_transform(data_standardized)   # get reduced-dimensional array calculated by pca
    explained_variance_ratio = pca.explained_variance_ratio_ # Check the explained variance ratio to see how much variance is explained
    # by each component
    print("Explained variance ratio:", explained_variance_ratio)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio) ## calculate the cumulative explained variance
    print("Cumulative explained variance:", cumulative_explained_variance)

    return data_pca

    # ## if you want to choose the number of components based on the explanation ability instead of selecting the number in advance
    # pca = PCA()  # leave the parameter empty
    # explained_variance_ratio = pca.explained_variance_ratio_  # Check the explained variance ratio to see how much variance is
    # explained by each component
    # print("Explained variance ratio:", explained_variance_ratio)
    # cumulative_explained_variance = np.cumsum(explained_variance_ratio)  # calculate the cumulative explained variance
    # print("Cumulative explained variance:", cumulative_explained_variance)
    # n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1  # select the number of components that explain 95% of the
    # variance
    # print("Number of components explaining 95% variance:", n_components_95)
    # pca_95 = PCA(n_components=n_components_95)  # apply PCA with the calculted number of components
    # data_pca_95 = pca_95.fit_transform(data_standardized)


## plot scatter pca features
def plotPcaFeatures(cnn_feature_after_pca, cnn_labels_to_pca, manual_feature_after_pca, manual_labels_to_pca):
    labels = ['SA-SA', 'SA-SS', 'SA-LW']  # order of number 5,6,4
    colors = ['r', 'g', 'b']
    font_size = 16
    label_pad = 20
    colormap = ListedColormap(colors)  # define colors for 3 categories
    fig = plt.figure(figsize=(10, 5))

    x_cnn = cnn_feature_after_pca[:, 0]
    y_cnn = cnn_feature_after_pca[:, 1]
    z_cnn = cnn_feature_after_pca[:, 2]
    cat_cnn = cnn_labels_to_pca

    ax_1 = fig.add_subplot(121, projection='3d')
    ax_1.scatter(x_cnn, y_cnn, z_cnn, c=cat_cnn, cmap=colormap, edgecolors="black")
    ax_1.set_xlabel('PCA 1', fontsize=font_size, labelpad=label_pad)
    ax_1.set_ylabel('PCA 2', fontsize=font_size, labelpad=label_pad)
    ax_1.set_zlabel('PCA 3', fontsize=font_size, labelpad=label_pad)
    # Set the font size of the axis ticks
    ax_1.tick_params(axis='x', labelsize=font_size)
    ax_1.tick_params(axis='y', labelsize=font_size)
    ax_1.tick_params(axis='z', labelsize=font_size)
    handles = [plt.Line2D([], [], color=colors[i], marker='o', linestyle='', label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=handles, fontsize=font_size)
    ax_1.set_title('(a) CNN-Extracted Features with PCA Dimension Reduction')

    x_manual = manual_feature_after_pca[:, 0]
    y_manual = manual_feature_after_pca[:, 1]
    z_manual = manual_feature_after_pca[:, 2]
    cat_manual = manual_labels_to_pca

    ax_2 = fig.add_subplot(122, projection='3d')
    ax_2.scatter(x_manual, y_manual, z_manual, c=cat_manual, cmap=colormap, edgecolors="black")
    ax_2.set_xlabel('PCA 1', fontsize=font_size, labelpad=label_pad)
    ax_2.set_ylabel('PCA 2', fontsize=font_size, labelpad=label_pad)
    ax_2.set_zlabel('PCA 3', fontsize=font_size, labelpad=label_pad)
    # Set the font size of the axis ticks
    ax_2.tick_params(axis='x', labelsize=font_size)
    ax_2.tick_params(axis='y', labelsize=font_size)
    ax_2.tick_params(axis='z', labelsize=font_size)
    handles = [plt.Line2D([], [], color=colors[i], marker='o', linestyle='', label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=handles, fontsize=font_size)
    ax_2.set_title('(b) Manually-Extracted Features with PCA Dimension Reduction')
    # fig.suptitle('Comparing CNN-extracted and Manually-Extracted Features Using PCA Dimension Reduction Visualization', fontsize=12)