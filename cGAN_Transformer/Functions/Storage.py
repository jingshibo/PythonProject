import os
import torch
import copy
import json
import numpy as np


##
def saveGanModels(models, gen_model, storage_parameters, project='cGAN_Model'):
    subject = storage_parameters['subject']
    version = storage_parameters['version']
    model_type = storage_parameters['model_type']
    for name in storage_parameters['model_name']:
        # model path
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\\transformer_model\\{gen_model}'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # save model
        models[name].to("cpu")
        torch.save(models[name].to("cpu"), model_path)


##
def loadGanModels(storage_parameters, gen_model, project='cGAN_Model'):
    models = {}
    # model path
    subject = storage_parameters['subject']
    version = storage_parameters['version']
    model_type = storage_parameters['model_type']
    for name in storage_parameters['model_name']:
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\\transformer_model\\{gen_model}'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # load model
        model = torch.load(model_path, weights_only=False)
        models[name] = model
    return models


## save models during training at certain check points
def saveCheckPointModels(models, storage_parameters, epoch_number, gen_model, project='cGAN_Model'):
    subject = storage_parameters['subject']
    version = storage_parameters['version']
    model_type = storage_parameters['model_type']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name in storage_parameters['model_name']:
        # model path
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\\transformer_model\\{gen_model}'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{name}_{epoch_number}.pth'
        model_path = os.path.join(data_dir, model_file)
        # save model
        models[name].to("cpu")
        torch.save(models[name].state_dict(), model_path)
        models[name].to(device)


## load models from certain check points
def loadCheckPointModels(storage_parameters, epoch_number, model_class_map, gen_model, project='cGAN_Model'):
    """
    Loads models saved via `state_dict`.

    Parameters:
    - storage_parameters: dict containing keys 'subject', 'version', 'model_type', and 'model_name' (list of model keys)
    - epoch_number: int, used to identify checkpoint file
    - model_class_map: dict {model_name: model_class_instance} with instantiated models
    - project: project folder name

    Returns:
    - dict {model_name: model instance with loaded weights}
    """
    models = {}
    subject = storage_parameters['subject']
    version = storage_parameters['version']
    model_type = storage_parameters['model_type']

    for name in storage_parameters['model_name']:
        # Build path
        data_dir = f'D:\\Data\\{project}\\subject_{subject}\\Experiment_{version}\\transformer_model\\{gen_model}'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{name}_{epoch_number}.pth'
        model_path = os.path.join(data_dir, model_file)

        # Load weights into the provided model structure
        model = model_class_map[name]
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        models[name] = model

    return models


## save classification accuracy and cm recall values
def saveClassifyResult(subject, accuracy, cm_recall, version, result_set, model_type, gen_model, project='cGAN_Model', num_reference=None):
    data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\\transformer_model_results\\{gen_model}'
    result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_reference_{num_reference}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # Combine the two dictionaries into one
    combined_data = {'accuracy': accuracy, 'cm_recall': cm_recall.tolist()}

    # Save to JSON file
    with open(result_path, 'w') as f:
        json.dump(combined_data, f, indent=8)


## read classification accuracy and cm recall values
def loadClassifyResult(subject, version, result_set, model_type, gen_model, project='cGAN_Model', num_reference=None):
    data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\\transformer_model_results\\{gen_model}'
    result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_reference_{num_reference}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    with open(result_path, 'r') as f:
        loaded_data = json.load(f)

    accuracy = loaded_data['accuracy']
    cm_recall = np.array(loaded_data['cm_recall'])

    return accuracy, cm_recall
