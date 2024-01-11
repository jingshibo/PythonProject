from Conditional_GAN.Models import Model_Storage
import numpy as np


## load results from all models of the subject
def getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference=1):
    accuracy_basis, cm_recall_basis = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_basis',
        project='cGAN_Model')
    accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_old',
        project='cGAN_Model')
    accuracy_best, cm_recall_best = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_best',
        project='cGAN_Model')
    accuracy_tf, cm_recall_tf = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_tf',
        project='cGAN_Model')
    accuracy_worst, cm_recall_worst = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_worst',
        project='cGAN_Model')

    accuracy_combine, cm_recall_combine = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_combine',
        project='cGAN_Model', num_reference=num_reference)
    accuracy_compare, cm_recall_compare = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_compare',
        project='cGAN_Model', num_reference=num_reference)
    accuracy_new, cm_recall_new = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_new',
        project='cGAN_Model', num_reference=num_reference)
    if num_reference != 0:
        accuracy_noise, cm_recall_noise = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_noise',
            project='cGAN_Model', num_reference=num_reference)
        accuracy_copy, cm_recall_copy = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_copy',
            project='cGAN_Model', num_reference=num_reference)
    else:
        cm_array = cm_recall_new['delay_0_ms']  # to get the shape of cm_recall array
        accuracy_noise, cm_recall_noise = ({'delay_0_ms': 0}, {'delay_0_ms': np.zeros_like(cm_array)})
        accuracy_copy, cm_recall_copy = ({'delay_0_ms': 0}, {'delay_0_ms': np.zeros_like(cm_array)})

    accuracy_old_mix, cm_recall_old_mix = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_old_mix',
        project='cGAN_Model')
    accuracy_old_copy, cm_recall_old_copy = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_old_copy',
        project='cGAN_Model')

    # accuracy = {'accuracy_best': accuracy_best, 'accuracy_tf': accuracy_tf, 'accuracy_combine': accuracy_combine,
    #     'accuracy_new': accuracy_new, 'accuracy_compare': accuracy_compare, 'accuracy_noise': accuracy_noise,
    #     'accuracy_copy': accuracy_copy, 'accuracy_worst': accuracy_worst, 'accuracy_basis': accuracy_basis, 'accuracy_old': accuracy_old}
    # cm_recall = {'cm_recall_best': cm_recall_best, 'cm_recall_tf': cm_recall_tf, 'cm_recall_combine': cm_recall_combine,
    #     'cm_recall_new': cm_recall_new, 'cm_recall_compare': cm_recall_compare, 'cm_recall_noise': cm_recall_noise,
    #     'cm_recall_copy': cm_recall_copy, 'cm_recall_worst': cm_recall_worst, 'cm_recall_basis': cm_recall_basis, 'cm_recall_old': cm_recall_old}
    # classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}
    accuracy = {'accuracy_worst': accuracy_worst, 'accuracy_copy': accuracy_copy, 'accuracy_noise': accuracy_noise,
        'accuracy_compare': accuracy_compare, 'accuracy_new': accuracy_new, 'accuracy_combine': accuracy_combine,
        'accuracy_tf': accuracy_tf, 'accuracy_best': accuracy_best, 'accuracy_basis': accuracy_basis, 'accuracy_old_mix': accuracy_old_mix,
        'accuracy_old': accuracy_old, 'accuracy_old_copy': accuracy_old_copy}
    cm_recall = {'cm_recall_worst': cm_recall_worst, 'cm_recall_copy': cm_recall_copy, 'cm_recall_noise': cm_recall_noise,
        'cm_recall_compare': cm_recall_compare, 'cm_recall_new': cm_recall_new, 'cm_recall_combine': cm_recall_combine,
        'cm_recall_tf': cm_recall_tf, 'cm_recall_best': cm_recall_best, 'cm_recall_basis': cm_recall_basis,
        'cm_recall_old_mix': cm_recall_old_mix, 'cm_recall_old': cm_recall_old, 'cm_recall_old_copy': cm_recall_old_copy}
    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}

    return classify_results


## load results from benchmark datasets of the subject (worst, best, tf_best, old_best)
def getBenchmarkResults(subject, version, basis_result_set):
    accuracy_basis, cm_recall_basis = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_basis',
        project='cGAN_Model')
    accuracy_best, cm_recall_best = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_best',
        project='cGAN_Model')
    accuracy_tf, cm_recall_tf = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_tf',
        project='cGAN_Model')
    accuracy_worst, cm_recall_worst = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_worst',
        project='cGAN_Model')

    accuracy = {'accuracy_worst': accuracy_worst, 'accuracy_tf': accuracy_tf, 'accuracy_best': accuracy_best, 'accuracy_basis': accuracy_basis}
    cm_recall = {'cm_recall_worst': cm_recall_worst, 'cm_recall_tf': cm_recall_tf, 'cm_recall_best': cm_recall_best, 'cm_recall_basis': cm_recall_basis}
    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}

    return classify_results


## load results of difference reference number for the subject
def getNumOfReferenceResults(subject, version, filter_result_set, num_reference):
    classify_results = {'accuracy': {}, 'cm_recall': {}}
    for reference in num_reference:
        accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_old', project='cGAN_Model',
            num_reference=reference)
        accuracy_new, cm_recall_new = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_new', project='cGAN_Model',
            num_reference=reference)
        accuracy_compare, cm_recall_compare = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_compare',
            project='cGAN_Model', num_reference=reference)
        accuracy_combine, cm_recall_combine = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_combine',
            project='cGAN_Model', num_reference=reference)
        if reference != 0:
            accuracy_noise, cm_recall_noise = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_noise',
                project='cGAN_Model', num_reference=reference)
            accuracy_copy, cm_recall_copy = Model_Storage.loadClassifyResult(subject, version, filter_result_set, 'classify_copy',
                project='cGAN_Model', num_reference=reference)
        else:
            cm_array = cm_recall_new['delay_0_ms']  # to get the shape of cm_recall array
            accuracy_noise, cm_recall_noise = ({'delay_0_ms': 0}, {'delay_0_ms': np.zeros_like(cm_array)})
            accuracy_copy, cm_recall_copy = ({'delay_0_ms': 0}, {'delay_0_ms': np.zeros_like(cm_array)})

        accuracy = {'accuracy_combine': accuracy_combine, 'accuracy_new': accuracy_new, 'accuracy_compare': accuracy_compare,
            'accuracy_noise': accuracy_noise, 'accuracy_copy': accuracy_copy, 'accuracy_old': accuracy_old}
        cm_recall = {'cm_recall_combine': cm_recall_combine, 'cm_recall_new': cm_recall_new, 'cm_recall_compare': cm_recall_compare,
            'cm_recall_noise': cm_recall_noise, 'cm_recall_copy': cm_recall_copy, 'cm_recall_old': cm_recall_old}
        # accuracy = {'accuracy_copy': accuracy_copy, 'accuracy_noise': accuracy_noise, 'accuracy_compare': accuracy_compare,
        #     'accuracy_new': accuracy_new, 'accuracy_combine': accuracy_combine, 'accuracy_old': accuracy_old}
        # cm_recall = {'cm_recall_copy': cm_recall_copy, 'cm_recall_noise': cm_recall_noise, 'cm_recall_compare': cm_recall_compare,
        #     'cm_recall_new': cm_recall_new, 'cm_recall_combine': cm_recall_combine, 'cm_recall_old': cm_recall_old}

        classify_results['accuracy'][f'reference_{reference}'] = accuracy
        classify_results['cm_recall'][f'reference_{reference}'] = cm_recall

    return classify_results


## load combine results of difference reference number for the subject
def getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference):
    classify_results = {'accuracy': {}, 'cm_recall': {}}
    accuracy_basis, cm_recall_basis = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_basis', project='cGAN_Model')
    classify_results['accuracy'][f'accuracy_basis'] = accuracy_basis
    classify_results['cm_recall'][f'cm_recall_basis'] = cm_recall_basis

    for reference in num_reference:
        accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, old_result_set, 'classify_old', project='cGAN_Model',
            num_reference=reference)
        classify_results['accuracy'][f'accuracy_old_{reference}'] = accuracy_old
        classify_results['cm_recall'][f'cm_recall_old_{reference}'] = cm_recall_old
    return classify_results


## load only old results of difference reference number for the subject
def getModeAccuracyResults(subject, version, basis_result_set, combine_result_set, num_reference):
    classify_results = {'accuracy': {}, 'cm_recall': {}}

    accuracy_tf, cm_recall_tf = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_tf', project='cGAN_Model')
    classify_results['accuracy'][f'accuracy_tf'] = accuracy_tf
    classify_results['cm_recall'][f'cm_recall_tf'] = cm_recall_tf

    for reference in num_reference:
        accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, combine_result_set, 'classify_combine', project='cGAN_Model',
            num_reference=reference)
        classify_results['accuracy'][f'accuracy_combine_{reference}'] = accuracy_old
        classify_results['cm_recall'][f'cm_recall_combine_{reference}'] = cm_recall_old

    accuracy_worst, cm_recall_worst = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_worst',
        project='cGAN_Model')
    classify_results['accuracy'][f'accuracy_worst'] = accuracy_worst
    classify_results['cm_recall'][f'cm_recall_worst'] = cm_recall_worst

    accuracy_best, cm_recall_best = Model_Storage.loadClassifyResult(subject, version, basis_result_set, 'classify_best',
        project='cGAN_Model')
    classify_results['accuracy'][f'accuracy_best'] = accuracy_best
    classify_results['cm_recall'][f'cm_recall_best'] = cm_recall_best

    return classify_results
