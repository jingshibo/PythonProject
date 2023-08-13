import os
import torch
import copy
import json
import numpy as np

def saveModels(models, subject, version, model_type, model_name, project='CycleGAN_Model'):
    for name in model_name:
        # model path
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # save model
        torch.save(models[name].to("cpu"), model_path)


def loadModels(subject, version, model_type, model_name, project='CycleGAN_Model'):
    models = {}
    # model path
    for name in model_name:
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # load model
        model = torch.load(model_path)
        models[name] = model
    return models


## save models during training at certain check points
def saveCheckPointModels(checkpoint_model_path, epoch_number, models):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_models_folder = os.path.join(checkpoint_model_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 0
    # Create the directory if it doesn't exist
    if not os.path.exists(checkpoint_models_folder):
        os.makedirs(checkpoint_models_folder)

    for model_name in models.keys():
        checkpoint_model_file = f'{model_name}_checkpoint_epoch_{epoch_number}.pt'
        model_path = os.path.join(checkpoint_models_folder, checkpoint_model_file)
        torch.save(models[model_name].to('cpu'), model_path)
        models[model_name].to(device)


## load models from certain check points
def loadCheckPointModels(checkpoint_model_path, epoch_number, model_name):
    models = {}
    # model path
    checkpoint_models_folder = os.path.join(checkpoint_model_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 1
    for name in model_name:
        checkpoint_model_file = f'{name}_checkpoint_epoch_{epoch_number}.pt'
        model_path = os.path.join(checkpoint_models_folder, checkpoint_model_file)
        # load model
        model = torch.load(model_path)
        models[name] = model
    return models


## save GAN model results
def saveCGanResults(subject, model_results, version, result_set, training_parameters, model_type, project='cGAN_Model'):
    data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\model_results'
    result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    result = {'model_results': {key: value.tolist() for key, value in model_results.items()}, 'training_parameters': training_parameters}
    with open(result_path, 'w') as json_file:
        json.dump(result, json_file, indent=8)


## load GAN model results
def loadCGanResults(subject, version, result_set, model_type, project='cGAN_Model'):
    data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\model_results'
    result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # read json file
    with open(result_path) as json_file:
        result_dict = json.load(json_file)

    return result_dict


## save model results during training at certain check points
def saveCheckPointCGanResults(checkpoint_result_path, epoch_number, blending_factors):
    checkpoint_models_folder = os.path.join(checkpoint_result_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 0
    # Create the directory if it doesn't exist
    if not os.path.exists(checkpoint_models_folder):
        os.makedirs(checkpoint_models_folder)

    checkpoint_result_file = f'result_checkpoint_epoch_{epoch_number}.pt'
    result_path = os.path.join(checkpoint_models_folder, checkpoint_result_file)

    model_results = {key: value.tolist() for key, value in blending_factors.items()}
    with open(result_path, 'w') as json_file:
        json.dump(model_results, json_file, indent=8)


## load model results from certain check points
def loadCheckPointCGanResults(checkpoint_result_path, epoch_number):
    checkpoint_models_folder = os.path.join(checkpoint_result_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 0
    checkpoint_result_file = f'result_checkpoint_epoch_{epoch_number}.pt'
    result_path = os.path.join(checkpoint_models_folder, checkpoint_result_file)

    # read json file
    with open(result_path) as json_file:
        result_dict = json.load(json_file)

    checkpoint_result = {key: np.array(value) for key, value in result_dict.items()}
    return checkpoint_result
