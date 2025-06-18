
##
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from cGAN_Transformer.Functions import Storage


##
def getAccuracyCm(model_results):
    group_accuracies = []
    all_true = []
    all_pred = []
    for group in model_results:
        y_true = group['true_value']
        y_pred = group['predict_value']

        # Accuracy per group
        acc = accuracy_score(y_true, y_pred)
        group_accuracies.append(acc)

        # Collect for overall confusion matrix
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    # Convert to arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Average accuracy across groups
    avg_accuracy = np.mean(group_accuracies)
    # Confusion matrix
    conf_mat = confusion_matrix(all_true, all_pred)
    cm_recall = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], 3)  # calculate cm recall

    return avg_accuracy, cm_recall


## load results from all models of the subject
def getSubjectResults(subject, version, result_set, num_reference=1):
    # model update results
    accuracy_basis, cm_recall_basis = Storage.loadClassifyResult(subject, version, result_set, 'classify_basis',
        project='cGAN_Model')
    accuracy_best, cm_recall_best = Storage.loadClassifyResult(subject, version, result_set, 'classify_best',
        project='cGAN_Model')
    accuracy_tf, cm_recall_tf = Storage.loadClassifyResult(subject, version, result_set, 'classify_tf',
        project='cGAN_Model', num_reference=10)
    accuracy_worst, cm_recall_worst = Storage.loadClassifyResult(subject, version, result_set, 'classify_worst',
        project='cGAN_Model')

    accuracy_old, cm_recall_old = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_old',
        project='cGAN_Model', num_reference=num_reference)
    accuracy_synthetic, cm_recall_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_synthetic',
        project='cGAN_Model', num_reference=num_reference)
    if num_reference != 0:
        accuracy_noise, cm_recall_noise = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_noisy',
            project='cGAN_Model', num_reference=num_reference)
        accuracy_copy, cm_recall_copy = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_copy',
            project='cGAN_Model', num_reference=num_reference)
    else:
        accuracy_noise, cm_recall_noise = (0, np.zeros_like(cm_recall_synthetic))
        accuracy_copy, cm_recall_copy = (0, np.zeros_like(cm_recall_synthetic))

    # old model results
    accuracy_old_real, cm_recall_old_real = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_real',
        project='cGAN_Model')
    accuracy_old_synthetic, cm_recall_old_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_synthetic',
        project='cGAN_Model')
    accuracy_old_imbalance, cm_recall_old_imbalance = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_imbalance',
        project='cGAN_Model', num_reference=5)
    accuracy_old_rebalanced, cm_recall_old_rebalanced = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_rebalanced',
        project='cGAN_Model', num_reference=5)

    accuracy = {'accuracy_old_real': accuracy_old_real, 'accuracy_old_synthetic': accuracy_old_synthetic,
        'accuracy_old_imbalance': accuracy_old_imbalance, 'accuracy_old_rebalanced': accuracy_old_rebalanced,
        'accuracy_worst': accuracy_worst, 'accuracy_best': accuracy_best, 'accuracy_copy': accuracy_copy, 'accuracy_noise': accuracy_noise,
        'accuracy_synthetic': accuracy_synthetic, 'accuracy_tf': accuracy_tf, 'accuracy_basis': accuracy_basis,
        'accuracy_old': accuracy_old}
    cm_recall = {'cm_recall_old_real': cm_recall_old_real, 'cm_recall_old_synthetic': cm_recall_old_synthetic,
        'cm_recall_old_imbalance': cm_recall_old_imbalance, 'cm_recall_old_rebalanced': cm_recall_old_rebalanced,
        'cm_recall_worst': cm_recall_worst, 'cm_recall_best': cm_recall_best, 'cm_recall_copy': cm_recall_copy,
        'cm_recall_noise': cm_recall_noise, 'cm_recall_synthetic': cm_recall_synthetic, 'cm_recall_tf': cm_recall_tf,
        'cm_recall_basis': cm_recall_basis, 'cm_recall_old': cm_recall_old}
    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}

    return classify_results


## load results from benchmark datasets of the subject (worst, best, tf_best, old_best)
def getBenchmarkResults(subject, version, result_set):
    accuracy_basis, cm_recall_basis = Storage.loadClassifyResult(subject, version, result_set, 'classify_basis',
        project='cGAN_Model')
    accuracy_best, cm_recall_best = Storage.loadClassifyResult(subject, version, result_set, 'classify_best',
        project='cGAN_Model')
    accuracy_tf, cm_recall_tf = Storage.loadClassifyResult(subject, version, result_set, 'classify_tf',
        project='cGAN_Model', num_reference=10)
    accuracy_worst, cm_recall_worst = Storage.loadClassifyResult(subject, version, result_set, 'classify_worst',
        project='cGAN_Model')

    accuracy = {'accuracy_worst': accuracy_worst, 'accuracy_best': accuracy_best, 'accuracy_tf': accuracy_tf,
        'accuracy_basis': accuracy_basis}
    cm_recall = {'cm_recall_worst': cm_recall_worst, 'cm_recall_best': cm_recall_best, 'cm_recall_tf': cm_recall_tf,
        'cm_recall_basis': cm_recall_basis}
    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}

    return classify_results


## load results of difference reference number for the subject
def getNumOfReferenceResults(subject, version, result_set, num_references):
    classify_results = {'accuracy': {}, 'cm_recall': {}}
    for reference in num_references:
        accuracy_old, cm_recall_old = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_old', project='cGAN_Model',
            num_reference=reference)
        accuracy_synthetic, cm_recall_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_synthetic',
            project='cGAN_Model', num_reference=reference)
        if reference != 0:
            accuracy_noise, cm_recall_noise = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_noisy',
                project='cGAN_Model', num_reference=reference)
            accuracy_copy, cm_recall_copy = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_copy',
                project='cGAN_Model', num_reference=reference)
        else:
            accuracy_noise, cm_recall_noise = (0, np.zeros_like(cm_recall_synthetic))
            accuracy_copy, cm_recall_copy = (0, np.zeros_like(cm_recall_synthetic))

        accuracy = {'accuracy_synthetic': accuracy_synthetic, 'accuracy_noise': accuracy_noise, 'accuracy_copy': accuracy_copy,
            'accuracy_old': accuracy_old}
        cm_recall = {'cm_recall_synthetic': cm_recall_synthetic, 'cm_recall_noise': cm_recall_noise, 'cm_recall_copy': cm_recall_copy,
            'cm_recall_old': cm_recall_old}

        classify_results['accuracy'][f'reference_{reference}'] = accuracy
        classify_results['cm_recall'][f'reference_{reference}'] = cm_recall

    return classify_results


## load only old results of different reference number for the subject
def getModeAccuracyResults(subject, version, result_set, num_references):
    classify_results = {'accuracy': {}, 'cm_recall': {}}

    accuracy_tf, cm_recall_tf = Storage.loadClassifyResult(subject, version, result_set, 'classify_tf',
        project='cGAN_Model', num_reference=10)
    classify_results['accuracy'][f'accuracy_tf'] = accuracy_tf
    classify_results['cm_recall'][f'cm_recall_tf'] = cm_recall_tf

    for reference in num_references:
        accuracy_synthetic, cm_recall_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_synthetic',
            project='cGAN_Model', num_reference=reference)
        classify_results['accuracy'][f'accuracy_synthetic_{reference}'] = accuracy_synthetic
        classify_results['cm_recall'][f'cm_recall_synthetic_{reference}'] = cm_recall_synthetic

    accuracy_worst, cm_recall_worst = Storage.loadClassifyResult(subject, version, result_set, 'classify_worst',
        project='cGAN_Model')
    classify_results['accuracy'][f'accuracy_worst'] = accuracy_worst
    classify_results['cm_recall'][f'cm_recall_worst'] = cm_recall_worst

    accuracy_best, cm_recall_best = Storage.loadClassifyResult(subject, version, result_set, 'classify_best',
        project='cGAN_Model')
    classify_results['accuracy'][f'accuracy_best'] = accuracy_best
    classify_results['cm_recall'][f'cm_recall_best'] = cm_recall_best

    return classify_results

