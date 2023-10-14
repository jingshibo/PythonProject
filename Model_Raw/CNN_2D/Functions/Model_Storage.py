import os
import torch

##
def saveModels(models, subject, version, model_type, model_name, project='Insole_Emg'):
    for name in model_name:
        # model path
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_group_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # save model
        torch.save(models[name].to("cpu"), model_path)

##
def loadModels(subject, version, model_type, model_name, project='Insole_Emg'):
    models = []
    # model path
    for name in model_name:
        data_dir = f'D:\Data\{project}\subject_{subject}\Experiment_{version}\models'
        model_file = f'subject_{subject}_Experiment_{version}_model_{model_type}_group_{name}.json'
        model_path = os.path.join(data_dir, model_file)
        # load model
        model = torch.load(model_path)
        models.append(model)
    return models