## Imports
from Conditional_GAN.Data_Procesing import Process_Raw_Data, Train_Classifiers, Train_cGan
from cGAN_Transformer.Functions import Preprocessing, Result_Analysis
# Assuming model.py is in the same directory or Python path is configured
from cGAN_Transformer.Grid_Search.Grid_Search_Model import Raw_Cnn_2d, ModelTraining, perform_grid_search, EmgDataSet, foldDataloader
import datetime
import numpy as np
import torch
import json
import os

# --- Data loading and initial preprocessing code ---
subject = 'Number1'
grid = 'grid_1'
version = 0
up_down_session_t0 = [0, 1, 2, 3, 4];
down_up_session_t0 = [1, 2, 3, 4, 5]
up_down_session_t1 = [0, 1, 2, 3, 4];
down_up_session_t1 = [1, 2, 3, 4, 5]
print("Loading and preprocessing data...")
# Placeholder for your actual data loading functions
# This part needs to be functional in your environment
try:
    old_emg_data, new_emg_data, window_parameters, start_before_toeoff_ms = Train_cGan.realEmgData(subject, version, up_down_session_t0,
        down_up_session_t0, up_down_session_t1, down_up_session_t1, grid=grid, envelope=True, envelope_cutoff=400, reordering=False)
    range_limit = 1500
    spatial_filter_kernel = (2, 1)
    modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
        'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'], 'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}
    length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']
    old_emg_normalized, new_emg_normalized, _, _ = Process_Raw_Data.normalizeFilterEmgData(old_emg_data, new_emg_data, range_limit,
        normalize='(0,1)', spatial_filter=False, kernel=spatial_filter_kernel)
    time_length = 500
    classify_old_emg, _, _ = Preprocessing.extractGanTrainingData(modes_generation, old_emg_normalized, new_emg_normalized, time_length)
except Exception as e:
    print(f"Error during actual data loading: {e}")
    print("Using dummy data for classify_old_emg to proceed with script structure testing.")
    num_samples, num_channels, feature_height, feature_width, num_classes = 100, 1, 60, 100, 5
    classify_old_emg = {'data_x': torch.randn(num_samples, num_channels, feature_height, feature_width),
        'int_y': np.random.randint(0, num_classes, num_samples)}
print("Data loading and preprocessing complete.")

num_cv_folds_for_gs = 5  # Folds for Grid Search
num_cv_folds_for_final = 5  # Folds for Final Model evaluation
print(f"Creating {num_cv_folds_for_gs}-fold CV indices for Grid Search...")
try:
    cross_validation_indices_gs, _ = Preprocessing.crossValidationSet(num_cv_folds_for_gs, classify_old_emg)
except Exception as e:
    print(f"Error creating CV splits with Preprocessing.crossValidationSet: {e}")
    print("Using dummy CV splits to proceed.")
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=num_cv_folds_for_gs, shuffle=True, random_state=42)
    indices = np.arange(classify_old_emg['data_x'].shape[0])
    cross_validation_indices_gs = []
    for train_idx, val_idx in skf.split(indices, classify_old_emg['int_y']):
        cross_validation_indices_gs.append({'train': train_idx, 'val': val_idx})

# --- Perform Grid Search ---
select_channels_gs = 'emg_all'  # Or any other specific channel selection
print("\n--- Starting Grid Search for Hyperparameter Optimization ---")
now_gs = datetime.datetime.now()

best_params_from_grid, best_effective_params, best_val_score_gs, all_gs_results_sorted = perform_grid_search(
    classify_emg_dict=classify_old_emg, cross_validation_indices=cross_validation_indices_gs, select_channels=select_channels_gs)
grid_search_duration = datetime.datetime.now() - now_gs
# Grid search summary will be printed at the very end.

# --- Conditional Final Training ---
# Final training runs if:
# 1. best_effective_params is not empty (meaning GS completed at least one trial, even if it was all defaults).
# 2. And you want to run final training even if param_grid was empty (i.e., best_params_from_grid is empty).
#    If you ONLY want to run final training if specific HPs were tuned and found best,
#    then the condition should be `bool(best_params_from_grid) and best_val_score_gs > -1.0`.
# Let's assume you want to run a "final" model configured by the best GS trial,
# even if that trial used all defaults because param_grid was empty.
run_final_training = bool(best_effective_params) and best_val_score_gs > -1.0

final_model_full_hparams_for_summary = None
final_training_duration_for_summary = None
avg_final_accuracy_for_summary = None
final_model_fold_accuracies_for_summary = []

if run_final_training:
    print("\n\n--- Proceeding to Final Model Training based on Grid Search Results ---")
    now_final_train = datetime.datetime.now()

    # best_effective_params contains the full configuration from the best GS trial
    final_num_epochs = best_effective_params['num_epochs_hp']
    final_batch_size = best_effective_params['batch_size_hp']
    final_decay_epochs_param = best_effective_params['decay_epochs_hp']
    final_gamma_param = best_effective_params['gamma_hp']

    final_model_arch_hparams = {}
    final_optimizer_hparams = {'lr': best_effective_params['lr'], 'weight_decay': best_effective_params['weight_decay']}
    for key, value in best_effective_params.items():
        if key not in ['lr', 'weight_decay', 'num_epochs_hp', 'batch_size_hp', 'decay_epochs_hp', 'gamma_hp']:
            final_model_arch_hparams[key] = value

    final_model_full_hparams_for_summary = best_effective_params  # Store for final summary

    print("\nEffective Hyperparameters for Final Model Training (from best grid search trial):")
    print(
        f"  Training Process: Epochs={final_num_epochs}, Batch={final_batch_size}, DecayEpochs={final_decay_epochs_param}, "
        f"Gamma={final_gamma_param}")
    print(f"  Optimizer Parameters: {final_optimizer_hparams}")
    print(f"  Model Architecture Parameters:")
    if final_model_arch_hparams:
        for k, v in final_model_arch_hparams.items(): print(f"    {k}: {v}")
    else:
        print("    (All model architecture parameters are using Raw_Cnn_2d defaults)")

    final_trainer = ModelTraining(num_epochs=final_num_epochs, batch_size=final_batch_size, report_period=10,
        model_params=final_model_arch_hparams, optimizer_params=final_optimizer_hparams)

    print(f"\nCreating {num_cv_folds_for_final}-fold CV indices for Final Model Training...")
    cross_validation_indices_final, _ = Preprocessing.crossValidationSet(num_cv_folds_for_final, classify_old_emg)

    print(f"Starting final training across {num_cv_folds_for_final} folds...")
    _, final_model_results_detailed = final_trainer.trainModel(classify_emg_dict=classify_old_emg,
        cross_validation_indices=cross_validation_indices_final, decay_epochs_param=final_decay_epochs_param, gamma_param=final_gamma_param,
        select_channels=select_channels_gs, return_score_only=False)
    final_training_duration_for_summary = datetime.datetime.now() - now_final_train
    print(f"Final model training completed in: {final_training_duration_for_summary}")

    if final_model_results_detailed:
        avg_final_accuracy_for_summary, _ = Result_Analysis.getAccuracyCm(final_model_results_detailed)
        for i, fold_result in enumerate(final_model_results_detailed):
            true_v = fold_result['true_value'];
            pred_v = fold_result['predict_value']
            fold_acc = np.sum(true_v == pred_v) / len(true_v) if len(true_v) > 0 else 0
            final_model_fold_accuracies_for_summary.append(fold_acc)
else:
    if not bool(best_params_from_grid) and best_val_score_gs > -1.0:
        print("\n--- Grid search ran with all default parameters (param_grid was empty). ---")
        print(
            f"--- The single grid search run (score: {best_val_score_gs:.4f}) with effective params: {best_effective_params} can be "
            f"considered the main model run. ---")
        print("--- Final training with specifically tuned parameters is skipped. ---")
    else:
        print(
            "\n--- Final training skipped. This could be because param_grid was not empty but no improvement was found, or an issue "
            "occurred in grid search. ---")

# --- Combined Summary Section (Printed at the very end) ---
print("\n\n\n========================= OVERALL SUMMARY =========================")
print(f"Script Execution Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n-------------------- GRID SEARCH DETAILS --------------------")
print(f"Grid Search Duration: {grid_search_duration}")
print(f"Total Grid Search Combinations Tried: {len(all_gs_results_sorted) if all_gs_results_sorted else 0}")
print(f"\nBest VARIED Parameters from Grid Search (parameters specified in param_grid):")
if best_params_from_grid and isinstance(best_params_from_grid, dict):
    for k, v in best_params_from_grid.items(): print(f"  {k}: {v}")
else:
    print("  None (param_grid was empty, or no specific varied combination was chosen as best).")
print(f"Full EFFECTIVE Parameters for the Best Grid Search Trial (includes defaults): {best_effective_params}")
print(f"Corresponding Best Average Last-Epoch Validation Accuracy during Grid Search: {best_val_score_gs:.4f}")

print("\nAll Grid Search Trial Results (Sorted by Avg Last-Epoch Validation Accuracy):")
if all_gs_results_sorted:
    for i, res in enumerate(all_gs_results_sorted):
        print(
            f"  Rank {i + 1}: AvgLastEpochValAcc={res.get('avg_val_accuracy', 'N/A'):.4f}, VariedParams="
            f"{res.get('params_from_grid') or 'None'}, EffectiveParams={res.get('effective_params_trial')}")
else:
    print("  No grid search trial results to display.")

print("\n-------------------- FINAL MODEL TRAINING DETAILS --------------------")
if run_final_training and final_model_full_hparams_for_summary:
    print(f"Final Model Training Duration: {final_training_duration_for_summary}")
    print("Full Effective Hyperparameters Used for Final Model (derived from best grid search trial):")
    print(
        f"  Training Process: Epochs={final_model_full_hparams_for_summary['num_epochs_hp']}, "
        f"Batch={final_model_full_hparams_for_summary['batch_size_hp']}, DecayEpochs="
        f"{final_model_full_hparams_for_summary['decay_epochs_hp']}, Gamma={final_model_full_hparams_for_summary['gamma_hp']}")
    print(
        f"  Optimizer: lr={final_model_full_hparams_for_summary['lr']}, weight_decay="
        f"{final_model_full_hparams_for_summary['weight_decay']}")
    final_model_arch_summary_from_effective = {k: v for k, v in final_model_full_hparams_for_summary.items() if
        k not in ['lr', 'weight_decay', 'num_epochs_hp', 'batch_size_hp', 'decay_epochs_hp', 'gamma_hp']}
    print(
        f"  Model Architecture: "
        f"{final_model_arch_summary_from_effective if final_model_arch_summary_from_effective else '(Raw_Cnn_2d Defaults Used)'}")

    if avg_final_accuracy_for_summary is not None:
        print(f"\nPerformance Metrics ({num_cv_folds_for_final}-fold CV):")
        print(f"  Average Validation Accuracy: {avg_final_accuracy_for_summary:.4f}")
        if final_model_fold_accuracies_for_summary:
            print("\n  Validation Accuracy per Fold:")
            for i, fold_acc in enumerate(final_model_fold_accuracies_for_summary):
                print(f"    Fold {i + 1}: {fold_acc:.4f}")
    else:
        print("\nNo detailed results from final model training (or it did not complete successfully).")

elif not bool(best_params_from_grid) and best_val_score_gs > -1.0:  # Case where param_grid was empty
    print("Final model training with specific 'best varied' parameters was skipped because param_grid was empty.")
    print(
        f"The single grid search run (AvgLastEpochValAcc: {best_val_score_gs:.4f}) with effective params: {best_effective_params} can be "
        f"considered the main model run.")
else:
    print("Final model training was skipped (e.g., no varied HPs improved performance or an issue occurred).")
print("==================================================================")

# Optional: Save the best_effective_params (full config for best GS trial)
# if best_effective_params:
#     gs_best_effective_path = os.path.join("results_summary", "grid_search_best_effective_params.json")
#     os.makedirs(os.path.dirname(gs_best_effective_path), exist_ok=True)
#     with open(gs_best_effective_path, 'w') as f:
#         json.dump(best_effective_params, f, indent=4)
#     print(f"Saved best EFFECTIVE HPs from grid search to {gs_best_effective_path}")

