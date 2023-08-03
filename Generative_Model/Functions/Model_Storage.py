import os
import torch


def saveModels(models, subject, version, model_type, model_name, project='Generative_Model'):
    for name in model_name:
        # model path
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # save model
        torch.save(models[name], model_path)


def loadModels(subject, version, model_type, model_name, project='Generative_Model'):
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


def saveCheckPointModels(checkpoint_folder_path, epoch_number, models):
    checkpoint_models_folder = os.path.join(checkpoint_folder_path, f'checkpoint_epoch_{epoch_number + 1}')  # epoch number starts from 0
    # Create the directory if it doesn't exist
    if not os.path.exists(checkpoint_models_folder):
        os.makedirs(checkpoint_models_folder)

    for model_name in models.keys():
        checkpoint_model_file = f'{model_name}_checkpoint_epoch_{epoch_number + 1}.pt'
        model_path = os.path.join(checkpoint_models_folder, checkpoint_model_file)
        torch.save(models[model_name], model_path)


def loadCheckPointModels(checkpoint_folder_path, epoch_number, model_name):
    models = {}
    # model path
    checkpoint_models_folder = os.path.join(checkpoint_folder_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 1
    for name in model_name:
        checkpoint_model_file = f'{model_name}_checkpoint_epoch_{epoch_number}.pt'
        model_path = os.path.join(checkpoint_models_folder, checkpoint_model_file)
        # load model
        model = torch.load(model_path)
        models[name] = model
    return models

