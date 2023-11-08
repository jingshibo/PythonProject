import os
import torch
import copy
import json
import numpy as np

##
def saveGanModels(models, subject, version, model_type, model_name, transition_type=None, project='cGAN_Model'):
    for name in model_name:
        # model path
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{transition_type}_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # save model
        torch.save(models[name].to("cpu"), model_path)

##
def loadGanModels(subject, version, model_type, model_name, transition_type=None, project='cGAN_Model'):
    models = {}
    # model path
    for name in model_name:
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{transition_type}_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # load model
        model = torch.load(model_path)
        models[name] = model
    return models


## save GAN model results
def saveCGanResults(subject, model_results, version, result_set, training_parameters, model_type, transition_type=None, project='cGAN_Model'):
    data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\model_results'
    result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{transition_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    result = {'model_results': {key: value.tolist() for key, value in model_results.items()}, 'training_parameters': training_parameters}
    with open(result_path, 'w') as json_file:
        json.dump(result, json_file, indent=8)


## load GAN model results
def loadCGanResults(subject, version, result_set, model_type, transition_type=None, project='cGAN_Model'):
    data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\model_results'
    result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_{transition_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # read json file
    with open(result_path) as json_file:
        result_dict = json.load(json_file)
    result_dict['model_results'] = {key: np.array(value) for key, value in result_dict['model_results'].items()}

    return result_dict


## save models during training at certain check points
def saveCheckPointModels(checkpoint_model_path, epoch_number, models, transition_type=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_models_folder = os.path.join(checkpoint_model_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 0
    # Create the directory if it doesn't exist
    if not os.path.exists(checkpoint_models_folder):
        os.makedirs(checkpoint_models_folder)

    for model_name in models.keys():
        checkpoint_model_file = f'{transition_type}_{model_name}_checkpoint_epoch_{epoch_number}.pt'
        model_path = os.path.join(checkpoint_models_folder, checkpoint_model_file)
        torch.save(models[model_name].to('cpu'), model_path)
        models[model_name].to(device)


## load models from certain check points
def loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=200, transition_type=None):
    models = {}
    # model path
    checkpoint_models_folder = os.path.join(checkpoint_model_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 1
    for name in model_name:
        checkpoint_model_file = f'{transition_type}_{name}_checkpoint_epoch_{epoch_number}.pt'
        model_path = os.path.join(checkpoint_models_folder, checkpoint_model_file)
        # load model
        model = torch.load(model_path)
        models[name] = model
    return models


## save model results during training at certain check points
def saveCheckPointCGanResults(checkpoint_result_path, epoch_number, blending_factors, training_parameters, transition_type=None):
    checkpoint_models_folder = os.path.join(checkpoint_result_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 0
    # Create the directory if it doesn't exist
    if not os.path.exists(checkpoint_models_folder):
        os.makedirs(checkpoint_models_folder)

    checkpoint_result_file = f'{transition_type}_result_checkpoint_epoch_{epoch_number}.pt'
    result_path = os.path.join(checkpoint_models_folder, checkpoint_result_file)

    result = {'model_results': {key: value.tolist() for key, value in blending_factors.items()}, 'training_parameters': training_parameters}
    with open(result_path, 'w') as json_file:
        json.dump(result, json_file, indent=8)


## load model results from certain check points
def loadCheckPointCGanResults(checkpoint_result_path, transition_type=None, epoch_number=200):
    checkpoint_models_folder = os.path.join(checkpoint_result_path, f'checkpoint_epoch_{epoch_number}')  # epoch number starts from 0
    checkpoint_result_file = f'{transition_type}_result_checkpoint_epoch_{epoch_number}.pt'
    result_path = os.path.join(checkpoint_models_folder, checkpoint_result_file)

    # read json file
    with open(result_path) as json_file:
        result_dict = json.load(json_file)
    result_dict['model_results'] = {key: np.array(value) for key, value in result_dict['model_results'].items()}
    return result_dict


## save classification accuracy and cm recall values
def saveClassifyResult(subject, accuracy, cm_recall, version, result_set, model_type, project='cGAN_Model', num_reference=None):
    if num_reference is None:
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\model_results'
        result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_results_{result_set}.json'
    else:  # save data to the num_reference folder
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\model_results\\num_reference\\filter_set_{result_set}'
        result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_reference_{num_reference}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # Convert numpy arrays in the cm_recall dictionary to lists
    cm_recall_to_list = {key: value.tolist() for key, value in cm_recall.items()}
    # Combine the two dictionaries into one
    combined_data = {'accuracy': accuracy, 'cm_recall': cm_recall_to_list}

    # Save to JSON file
    with open(result_path, 'w') as f:
        json.dump(combined_data, f, indent=8)


## read classification accuracy and cm recall values
def loadClassifyResult(subject, version, result_set, model_type, project='cGAN_Model', num_reference=None):
    if num_reference is None:
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\model_results'
        result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_results_{result_set}.json'
    else:  # save data to the num_reference folder
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\model_results\\num_reference\\filter_set_{result_set}'
        result_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_reference_{num_reference}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    with open(result_path, 'r') as f:
        loaded_data = json.load(f)

    accuracy = loaded_data['accuracy']
    cm_recall = {key: np.array(value) for key, value in loaded_data['cm_recall'].items()}

    return accuracy, cm_recall


##
def saveClassifyModels(models, subject, version, model_type, model_number=None, project='cGAN_Model', num_reference=None):
    # model path
    for number in model_number:
        # model path
        if num_reference is None:
            data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models\classifier'
            model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_group_{number}.json'
        else:
            data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models\classifier'
            model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_reference_{num_reference}_group_{number}.json'
        model_path = os.path.join(data_dir, model_file)
        # save model
        torch.save(models[number].to("cpu"), model_path)


##
def loadClassifyModels(subject, version, model_type, model_number=None, project='cGAN_Model', num_reference=None):
    models = []
    # model path
    for number in model_number:
        if num_reference is None:
            data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models\classifier'
            model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_group_{number}.json'
        else:
            data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models\classifier'
            model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_reference_{num_reference}_group_{number}.json'
        model_path = os.path.join(data_dir, model_file)
        # load model
        model = torch.load(model_path)
        models.append(model)
    return models


## read blending factors for fake data generation
def loadBlendingFactors(subject, version, result_set, model_type, modes_generation, checkpoint_result_path, epoch_number=None):
    gen_results = {}
    for transition_type in modes_generation.keys():
        if epoch_number is None:
            gen_result = loadCGanResults(subject, version, result_set, model_type, transition_type, project='cGAN_Model')
        else:  # if no epoch_number input, output the final epoch results
            gen_result = loadCheckPointCGanResults(checkpoint_result_path, transition_type, epoch_number=epoch_number)
        gen_results[transition_type] = gen_result  # there could be more than one transition to generate
    return gen_results