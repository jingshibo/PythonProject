## Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchinfo import summary # Uncomment if you use it
import numpy as np
import datetime
import os
import itertools


## Design Model
class Raw_Cnn_2d(nn.Module):
    def __init__(self, input_size, class_number, conv1_out_channels=32, conv1_kernel_size=3, pool1_kernel_size=2, pool1_stride=1,
            conv2_out_channels=32, conv2_kernel_size=3, pool2_kernel_size=2, pool2_stride=1, conv3_out_channels=32, conv3_kernel_size=3,
            pool3_kernel_size=2, pool3_stride=1, linear1_units=512, linear2_units=128, dropout_rate=0.5):
        super(Raw_Cnn_2d, self).__init__()

        self.conv1_parameter = [conv1_out_channels, conv1_kernel_size]
        self.pool1_ks = pool1_kernel_size
        self.pool1_s = pool1_stride

        self.conv2_parameter = [conv2_out_channels, conv2_kernel_size]
        self.pool2_ks = pool2_kernel_size
        self.pool2_s = pool2_stride

        self.conv3_parameter = [conv3_out_channels, conv3_kernel_size]
        self.pool3_ks = pool3_kernel_size
        self.pool3_s = pool3_stride

        self.linear1_parameter = linear1_units
        self.linear2_parameter = linear2_units
        self.dropout_rate = dropout_rate

        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=self.conv1_parameter[0], kernel_size=self.conv1_parameter[1], dilation=2,
                stride=2), nn.BatchNorm2d(self.conv1_parameter[0]), nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=self.pool1_ks, stride=self.pool1_s, padding=0),

            nn.Conv2d(in_channels=self.conv1_parameter[0], out_channels=self.conv2_parameter[0], kernel_size=self.conv2_parameter[1],
                dilation=2, stride=2), nn.BatchNorm2d(self.conv2_parameter[0]), nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=self.pool2_ks, stride=self.pool2_s, padding=0),

            nn.Conv2d(in_channels=self.conv2_parameter[0], out_channels=self.conv3_parameter[0], kernel_size=self.conv3_parameter[1],
                dilation=2, stride=2), nn.BatchNorm2d(self.conv3_parameter[0]), nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=self.pool3_ks, stride=self.pool3_s, padding=0), )

        self.linear_layer = nn.Sequential(nn.LazyLinear(self.linear1_parameter), nn.BatchNorm1d(self.linear1_parameter), nn.LeakyReLU(0.01),
            nn.Dropout(self.dropout_rate),

            nn.LazyLinear(self.linear2_parameter), nn.BatchNorm1d(self.linear2_parameter), nn.LeakyReLU(0.01), nn.Dropout(self.dropout_rate),

            nn.LazyLinear(class_number))  # self.initialize_weights() # Call if needed, perhaps after model creation

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):  # Tuple for multiple types
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, intermediate_features=False):
        x = self.convolutional_layer(x)
        cnn_features = x.detach().clone() if intermediate_features else None
        x = torch.flatten(x, 1)
        flatten_features = x.detach().clone() if intermediate_features else None
        x = self.linear_layer(x)

        if intermediate_features:
            return x, cnn_features, flatten_features
        else:
            return x


## Training Class
class ModelTraining():
    def __init__(self, num_epochs, batch_size, report_period=10, model_params=None, optimizer_params=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.report_period = report_period
        self.model_params = model_params if model_params is not None else {}
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.optimizer_params.setdefault('lr', 0.01)  # Default LR if not specified
        self.optimizer_params.setdefault('weight_decay', 0.0001)  # Default WD if not specified
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None  # Will be instantiated per fold
        self.base_result_dir = os.path.join('D:', os.sep, 'Project', 'pythonProject', 'Model_Raw', 'CNN_2D',
            'Results')  # Safer path construction

    def trainModel(self, classify_emg_dict, cross_validation_indices, decay_epochs_param, gamma_param, select_channels='emg_all',
            return_score_only=False, current_hparams_str=""):

        fold_models_state_dicts = []
        fold_results_detailed = []
        fold_last_epoch_val_accuracies = []

        fold_number_count = len(cross_validation_indices)
        for fold_id in range(fold_number_count):
            group_number = f"group_{fold_id}"

            # --- TensorBoard Writer Initialization ---
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            if current_hparams_str and current_hparams_str != "all_defaults_run":  # Be more specific for GS trials
                experiment_name = f'gs_trial_{current_hparams_str}_fold_{fold_id}_{timestamp}'
            elif current_hparams_str == "all_defaults_run":  # Specific name for all_defaults run
                experiment_name = f'all_defaults_run_fold_{fold_id}_{timestamp}'
            else:  # For final training or single run without GS context
                experiment_name = f'final_run_fold_{fold_id}_{timestamp}'

            # Ensure result_dir is unique if running multiple final_trainings without restarting script
            # by incorporating a unique part of best_hyperparams if they exist
            if not return_score_only and current_hparams_str:  # e.g. final training with best hparams
                experiment_name = f'final_model_best_hps_{current_hparams_str}_fold_{fold_id}_{timestamp}'

            self.result_dir_fold = os.path.join(self.base_result_dir, experiment_name)  # Specific dir for this fold/trial
            os.makedirs(self.result_dir_fold, exist_ok=True)
            # self.writer = SummaryWriter(log_dir=self.result_dir_fold)
            # --- End TensorBoard Writer Initialization ---

            input_size = classify_emg_dict['data_x'].shape[1]
            # Ensure int_y is 1D for set()
            y_labels_for_class_count = classify_emg_dict['int_y']
            if isinstance(y_labels_for_class_count, np.ndarray) and y_labels_for_class_count.ndim > 1:
                y_labels_for_class_count = y_labels_for_class_count.flatten()
            class_number = len(set(y_labels_for_class_count))

            data_set = self.selectSamples(classify_emg_dict, select_channels)

            self.train_loader, self.val_loader = foldDataloader(data_set, cross_validation_indices, fold_id, self.batch_size,
                onehot_label=False, shuffle_train=True, shuffle_test=False, drop_last=True)

            self.model = Raw_Cnn_2d(input_size, class_number, **self.model_params).to(self.device)
            # self.model.initialize_weights() # Optional: call weight initialization

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['lr'],
                weight_decay=self.optimizer_params['weight_decay'])
            self.loss_fn = torch.nn.CrossEntropyLoss()

            self.lr_scheduler = None  # Initialize to None
            if self.train_loader and len(self.train_loader) > 0:
                decay_steps = int(decay_epochs_param * len(self.train_loader))  # Ensure int
                if decay_steps > 0:
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=gamma_param)
                else:
                    print(f"Warning: decay_steps is {decay_steps} for fold {fold_id}. LR scheduler not created with StepLR.")
            else:
                print(f"Warning: train_loader is None or empty for fold {fold_id}. LR scheduler not created.")

            last_epoch_val_accuracy_this_fold = 0.0

            print(f"\n--- Fold {fold_id + 1}/{fold_number_count} (Epochs: {self.num_epochs}, Batch: {self.batch_size}) ---")
            if current_hparams_str: print(f"Varied HParams for this trial: {current_hparams_str.replace('_', ', ')}")

            for epoch_number in range(self.num_epochs):
                train_acc_epoch, train_loss_epoch = self.trainOneEpoch(group_number, epoch_number)

                if self.writer:
                    self.writer.add_scalar(f'{group_number}/Loss/Train', train_loss_epoch, epoch_number)
                    self.writer.add_scalar(f'{group_number}/Accuracy/Train', train_acc_epoch, epoch_number)
                    self.writer.add_scalar(f'{group_number}/LearningRate', self.optimizer.param_groups[0]['lr'], epoch_number)

                is_last_epoch = (epoch_number == self.num_epochs - 1)
                should_validate_for_monitoring = (not return_score_only) and ((epoch_number + 1) % self.report_period == 0 or is_last_epoch)

                if (return_score_only and is_last_epoch) or should_validate_for_monitoring:
                    _, _, _, val_accuracy_epoch_fold, val_loss_epoch_fold = self.predictValResults(group_number, epoch_number)

                    if self.writer and should_validate_for_monitoring:
                        self.writer.add_scalar(f'{group_number}/Loss/Validation', val_loss_epoch_fold, epoch_number)
                        self.writer.add_scalar(f'{group_number}/Accuracy/Validation', val_accuracy_epoch_fold, epoch_number)

                    if is_last_epoch:
                        last_epoch_val_accuracy_this_fold = val_accuracy_epoch_fold

            fold_last_epoch_val_accuracies.append(last_epoch_val_accuracy_this_fold)

            if not return_score_only:
                # For final training, detailed results come from the last epoch's validation
                # predictValResults was already called for the last epoch if should_validate_for_monitoring was true
                # We need its full output, not just accuracy.
                val_true_final, val_softmax_final, val_pred_final, _, _ = self.predictValResults(group_number, self.num_epochs - 1)
                fold_results_detailed.append(
                    {"true_value": val_true_final, "predict_softmax": val_softmax_final, "predict_value": val_pred_final})
                fold_models_state_dicts.append(self.model.to("cpu").state_dict())

            if self.writer:
                # Prepare hparams for logging to TensorBoard
                # These are the *effective* parameters used for this fold/trial
                hparams_for_tb = {'lr_eff': self.optimizer_params['lr'],  # Use a prefix to denote effective
                    'wd_eff': self.optimizer_params['weight_decay'], 'epochs_eff': self.num_epochs, 'batch_eff': self.batch_size,
                    'decay_ep_eff': decay_epochs_param, 'gamma_eff': gamma_param, 'fold': fold_id, **self.model_params
                    # Add all model params used
                }
                # The metric is the last epoch's validation accuracy for this fold
                metric_to_log_for_hparams = {'hparam/last_epoch_val_accuracy_fold': last_epoch_val_accuracy_this_fold}

                self.writer.add_hparams(hparams_for_tb, metric_to_log_for_hparams)
                self.writer.close()  # Close writer for this fold

        if return_score_only:
            avg_last_epoch_val_accuracy = np.mean(fold_last_epoch_val_accuracies) if fold_last_epoch_val_accuracies else 0.0
            return avg_last_epoch_val_accuracy
        else:
            return fold_models_state_dicts, fold_results_detailed

    def trainOneEpoch(self, group_number, epoch_number):
        self.model.train(True)
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        if not self.train_loader:
            print(f"Warning: train_loader is empty or None for group {group_number}, epoch {epoch_number}. Skipping training.")
            return 0.0, 0.0

        for batch_number, data in enumerate(self.train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(device=self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            train_loss = self.loss_fn(outputs, labels)
            train_loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()

            running_loss += train_loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_preds if total_preds > 0 else 0.0
        epoch_acc = correct_preds / total_preds if total_preds > 0 else 0.0

        if (epoch_number + 1) % self.report_period == 0 or epoch_number == self.num_epochs - 1:
            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Group: {group_number}, Epoch: {epoch_number + 1}/{self.num_epochs}, Train Acc: {epoch_acc:.4f}, Train Loss: "
                f"{epoch_loss:.4f}, LR: {current_lr:.2e}")
        # else: # Optional: print LR more frequently if needed for debugging
        #     current_lr = self.optimizer.param_groups[0]['lr']
        #     print(f"Group: {group_number}, Epoch: {epoch_number + 1}, Current LR: {current_lr:.2e}")

        return epoch_acc, epoch_loss

    def predictValResults(self, group_number, epoch_number):
        self.model.eval()  # Use eval mode
        val_true_labels = []
        val_predict_softmax = []
        val_predict_labels = []
        running_loss = 0.0
        num_correct = 0
        num_sample = 0

        if not self.val_loader:
            print(f"Warning: val_loader is empty or None for group {group_number}, epoch {epoch_number}. Skipping validation.")
            return np.array([]), np.array([]), np.array([]), 0.0, 0.0

        with torch.no_grad():
            for i, val_data in enumerate(self.val_loader):
                val_inputs, val_labels = val_data[0].to(self.device), val_data[1].to(device=self.device, dtype=torch.long)
                val_outputs = self.model(val_inputs)
                loss = self.loss_fn(val_outputs, val_labels)
                val_softmax_batch = F.softmax(val_outputs, dim=1)

                running_loss += loss.item() * val_inputs.size(0)
                _, predicted = torch.max(val_outputs.data, 1)
                num_correct += (predicted == val_labels).sum().item()
                num_sample += val_labels.size(0)

                val_predict_labels.extend(predicted.cpu().numpy())
                val_predict_softmax.extend(val_softmax_batch.cpu().numpy())
                val_true_labels.extend(val_labels.cpu().numpy())

        avg_loss = running_loss / num_sample if num_sample > 0 else 0.0
        accuracy = num_correct / num_sample if num_sample > 0 else 0.0

        if (epoch_number + 1) % self.report_period == 0 or epoch_number == self.num_epochs - 1:
            print(
                f"Group: {group_number}, Epoch: {epoch_number + 1}/{self.num_epochs} (Val), Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")

        return np.array(val_true_labels), np.array(val_predict_softmax), np.array(val_predict_labels), accuracy, avg_loss

    def selectSamples(self, group_value, select_channels='emg_all', bipolar_position=(0, 0)):
        source_data_x_key = 'data_x' if 'data_x' in group_value else 'train_feature_x'
        if source_data_x_key not in group_value:
            raise ValueError("Key 'data_x' or 'train_feature_x' not found in group_value for channel selection.")
        source_data_x = group_value[source_data_x_key]

        if select_channels == 'emg_all':
            data_x_selected = source_data_x
        elif select_channels == 'emg_1':
            data_x_selected = source_data_x[:, :, :, 0:65]
        elif select_channels == 'emg_2':
            data_x_selected = source_data_x[:, :, :, 65:130]
        elif select_channels == 'bipolar':
            raise NotImplementedError("Bipolar channel selection not implemented.")
        else:
            raise Exception(f"No Such Channels or selection logic not implemented for: {select_channels}")

        data_set = {'data_x': data_x_selected, 'int_y': group_value['int_y'], 'onehot_y': group_value.get('onehot_y'),
            'label_map': group_value.get('label_map')}
        return data_set


## Grid Search Function
def perform_grid_search(classify_emg_dict, cross_validation_indices, select_channels='emg_all'):
    # These are the single source of truth for defaults for training/optimizer HPs
    # if they are not specified in the param_grid.
    DEFAULT_LR = 0.001
    DEFAULT_WEIGHT_DECAY = 0.01
    DEFAULT_NUM_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_DECAY_EPOCHS = 20
    DEFAULT_GAMMA = 0.1
    # CNN Model architecture defaults are defined in Raw_Cnn_2d.__init__

    # --- CUSTOMIZE YOUR HYPERPARAMETER GRID HERE ---
    # For any hyperparameter NOT listed here, its DEFAULT value (from above or from
    # class __init__ methods for model architecture) will be used.
    param_grid = {
        # Optimizer parameters (Adam)
        'lr': [0.005, 0.001, 0.0005],
        'weight_decay': [0.1, 0.05, 0.01],

        # Training process parameters
        'num_epochs_hp': [50, 60],  # Renamed to avoid clash with ModelTraining.num_epochs
        'batch_size_hp': [32, 64],  # Renamed
        'decay_epochs_hp': [10, 20, 30],  # Renamed, for StepLR step_size calculation
        'gamma_hp': [0.1, 0.3, 0.5],  # Renamed, for StepLR gamma
        # 'conv1_out_channels': [16, 32], # To tune CNN params, add them here
    }
    # To run with ALL defaults, make param_grid empty:
    # param_grid = {}
    # --- END OF CUSTOMIZATION ---

    best_score = -1.0
    best_params_from_grid = {}  # Stores ONLY the varied params from the grid for the best trial
    best_effective_params_for_final_run = {}  # Stores the FULL config of the best trial
    all_trial_results = []

    if not param_grid:
        grid_keys = []
        grid_values_for_product = [[]]  # Ensures one iteration for the default run
        print("--- param_grid is empty. Running a single trial with default hyperparameters. ---")
    else:
        grid_keys = list(param_grid.keys())
        grid_values_for_product = list(param_grid.values())

    hyperparameter_combinations = list(itertools.product(*grid_values_for_product))
    total_combinations = len(hyperparameter_combinations)

    print(f"--- Starting Grid Search: {total_combinations} combinations ---")
    if total_combinations > 1 and total_combinations > 200:
        print(f"WARNING: Grid search has {total_combinations} combinations, this may take a very long time!")

    for i, combo_values_from_grid in enumerate(hyperparameter_combinations):
        # This dictionary will ONLY contain parameters specified in the param_grid for this combo
        current_params_from_grid_dict = dict(zip(grid_keys, combo_values_from_grid))

        # Determine the effective parameters for THIS TRIAL by starting with global defaults
        # and then overriding with what's in current_params_from_grid_dict.
        # Also, any model architecture parameters from param_grid will be added.
        current_trial_effective_params = {'lr': DEFAULT_LR, 'weight_decay': DEFAULT_WEIGHT_DECAY, 'num_epochs_hp': DEFAULT_NUM_EPOCHS,
            'batch_size_hp': DEFAULT_BATCH_SIZE, 'decay_epochs_hp': DEFAULT_DECAY_EPOCHS, 'gamma_hp': DEFAULT_GAMMA, }
        # Update with values from the current grid combination.
        # This also adds any model architecture params that were in param_grid.
        current_trial_effective_params.update(current_params_from_grid_dict)

        # Extract parameters for ModelTraining and trainModel calls
        effective_lr = current_trial_effective_params['lr']
        effective_weight_decay = current_trial_effective_params['weight_decay']
        effective_num_epochs = current_trial_effective_params['num_epochs_hp']
        effective_batch_size = current_trial_effective_params['batch_size_hp']
        effective_decay_epochs = current_trial_effective_params['decay_epochs_hp']
        effective_gamma = current_trial_effective_params['gamma_hp']

        optimizer_hparams_dict = {'lr': effective_lr, 'weight_decay': effective_weight_decay}

        model_hparams_dict = {}
        # Populate model_hparams_dict with model-specific keys from current_trial_effective_params
        # (These would be keys that were in param_grid but are not training/optimizer process keys)
        for key, value in current_trial_effective_params.items():
            if key not in ['lr', 'weight_decay', 'num_epochs_hp', 'batch_size_hp', 'decay_epochs_hp', 'gamma_hp']:
                model_hparams_dict[key] = value

        varied_params_for_log_str = current_params_from_grid_dict if grid_keys else {'status': 'all_defaults'}
        hparams_str_for_log = "_".join([f"{k}-{v}" for k, v in varied_params_for_log_str.items()]).replace(".", "p")
        if not hparams_str_for_log: hparams_str_for_log = "all_defaults_run"  # Should not happen if varied_params_for_log_str has status

        print(f"\nGrid Search Trial {i + 1}/{total_combinations}. Varied params for this trial: {varied_params_for_log_str}")
        # print(f"  Full effective params for this trial: {current_trial_effective_params}") # Uncomment for debugging

        trainer = ModelTraining(num_epochs=effective_num_epochs, batch_size=effective_batch_size, report_period=5,
            # Or make this a hyperparameter if needed
            model_params=model_hparams_dict,  # Will be empty if no model HPs in param_grid
            optimizer_params=optimizer_hparams_dict)

        avg_val_accuracy = trainer.trainModel(classify_emg_dict, cross_validation_indices, decay_epochs_param=effective_decay_epochs,
            gamma_param=effective_gamma, select_channels=select_channels, return_score_only=True, current_hparams_str=hparams_str_for_log)

        all_trial_results.append({'params_from_grid': current_params_from_grid_dict.copy(),  # What was varied
            'effective_params_trial': current_trial_effective_params.copy(),  # Full config for this trial
            'avg_val_accuracy': avg_val_accuracy})

        if avg_val_accuracy > best_score:
            best_score = avg_val_accuracy
            best_params_from_grid = current_params_from_grid_dict.copy()
            best_effective_params_for_final_run = current_trial_effective_params.copy()

        print(
            f"Grid Search Trial {i + 1}/{total_combinations} COMPLETE. Avg Last-Epoch Val Acc: {avg_val_accuracy:.4f}. Best so far: "
            f"{best_score:.4f}")
        print("----------------------------------------------------")

    print("\n========== Grid Search Complete ==========")
    if not param_grid and not best_params_from_grid and all_trial_results:  # Empty grid case, one default run
        if all_trial_results:  # Ensure the default run actually produced a result
            best_score = all_trial_results[0]['avg_val_accuracy']
            best_effective_params_for_final_run = all_trial_results[0][
                'effective_params_trial'].copy()  # best_params_from_grid remains empty as nothing was varied from a grid.
        else:
            print("Warning: Default run did not produce results.")

    print(f"Best VARIED Parameters from Grid Search: {best_params_from_grid or 'None (param_grid was empty)'}")
    print(f"Full EFFECTIVE Parameters for the Best Trial: {best_effective_params_for_final_run}")
    print(f"Best Average Last-Epoch Validation Accuracy from Grid Search: {best_score:.4f}")

    all_trial_results_sorted = sorted(all_trial_results, key=lambda x: x.get('avg_val_accuracy', -1), reverse=True)
    print("\n--- All Grid Search Trial Results (Sorted by Avg Last-Epoch Validation Accuracy) ---")
    for res in all_trial_results_sorted:
        print(
            f"Varied Params: {res.get('params_from_grid')} -> Effective Params: {res.get('effective_params_trial')} -> Avg Last-Epoch Val "
            f"Acc: {res.get('avg_val_accuracy', 'N/A'):.4f}")

    return best_params_from_grid, best_effective_params_for_final_run, best_score, all_trial_results_sorted


## Load data one by one from a fold
class EmgDataSet(Dataset):
    def __init__(self, data_x, data_y, indices):
        if isinstance(data_x, np.ndarray):
            self.data_x = torch.from_numpy(data_x.astype(np.float32))
        elif not isinstance(data_x, torch.Tensor):
            self.data_x = torch.tensor(data_x, dtype=torch.float32)
        else:
            self.data_x = data_x

        if isinstance(data_y, np.ndarray):
            self.labels = torch.from_numpy(data_y.astype(np.int64))  # Ensure long
        elif isinstance(data_y, list):
            self.labels = torch.tensor(data_y, dtype=torch.long)
        elif isinstance(data_y, torch.Tensor) and data_y.dtype != torch.long:
            self.labels = data_y.long()
        else:  # Assuming already a torch.LongTensor
            self.labels = data_y

        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.data_x[i], self.labels[i]


## Build a dataloader to load data from each fold
def foldDataloader(classify_emg_dict, cross_validation_indices, fold_id, batch_size, onehot_label=False, shuffle_train=True,
        shuffle_test=False, drop_last=True):
    fold = cross_validation_indices[fold_id]
    train_idx = fold['train']
    val_idx = fold['val']

    # Always use int_y for CrossEntropyLoss
    labels_source = classify_emg_dict['onehot_y'] if onehot_label else classify_emg_dict['int_y']
    data_x_source = classify_emg_dict['data_x']

    train_dataset = EmgDataSet(data_x_source, labels_source, train_idx)
    val_dataset = EmgDataSet(data_x_source, labels_source, val_idx)

    # Consider adding num_workers > 0 for performance if not debugging, but can cause issues on Windows/Jupyter
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=0)

    return train_loader, val_loader