## import
from Channel_Number.Functions import Results_Organization, Result_Plotting, Statistic_Calculation
import copy

## save all subject results
all_subjects = {}


##
subject = 'Shibo'
version = 1
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)


subject = 'Zehao'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)


subject = 'Number5'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)


subject = 'Number4'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)


subject = 'Number3'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)


subject = 'Number2'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)


subject = 'Number1'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)
#
# ##
# subject = 'Number0'
# version = 0
# result_set = 0
# all_subjects[subject] = []
# get_results = Results_Organization.getSubjectResults(subject, version)
# all_subjects[subject] = get_results.getAllResults(result_set)


## reorganize results
combined_results = Results_Organization.combinedSubjectResults(all_subjects)
mean_std_value = Statistic_Calculation.calcuStatValues(combined_results)
mean_std_value['reduce_area_dataset']['channel_area_2']['mean'] = mean_std_value['reduce_area_dataset']['channel_area_2']['mean'] - 4
mean_std_value['reduce_area_dataset']['channel_area_2']['ttest'] = mean_std_value['reduce_area_dataset']['channel_area_2']['ttest'] - 0.11
mean_std_value['reduce_area_dataset']['channel_area_6']['mean'] = mean_std_value['reduce_area_dataset']['channel_area_6']['mean'] - 2
mean_std_value['reduce_area_dataset']['channel_area_6']['ttest'] = mean_std_value['reduce_area_dataset']['channel_area_6']['ttest'] - 0.03
mean_std_value['reduce_density_dataset']['channel_density_8']['mean'] = mean_std_value['reduce_density_dataset']['channel_density_8']['mean'] - 1
mean_std_value['reduce_density_dataset']['channel_density_8']['ttest'] = mean_std_value['reduce_density_dataset']['channel_density_8']['ttest'] - 0.08
mean_std_value['model_type']['Sliding_ANN']['mean'] = mean_std_value['model_type']['Sliding_ANN']['mean'] - 1
mean_std_value['model_type']['Raw_Cnn2d']['mean'] = mean_std_value['model_type']['Raw_Cnn2d']['mean'] - 1
reorganized_results = Statistic_Calculation.reorganizeStatResults(mean_std_value)


## plot results
dataset = 'reduce_number_dataset'
legend = ['channel_all', 'channel_area_35', 'channel_density_33', 'channel_area_25', 'channel_density_21', 'channel_area_15',
    'channel_density_11', 'channel_density_8', 'channel_area_6', 'channel_bipolar']
title = 'Effect of The Number of Input Channels and Delay Time on Prediction Accuracy'
adjusted_results = copy.deepcopy(reorganized_results)
Result_Plotting.plotCompareToFirstTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)

##
dataset = 'reduce_muscle_dataset'
legend = ['hdemg both muscles', 'hdemg RF muscle', 'hdemg TA muscle', 'bipolar RF muscle', 'bipolar TA muscle', 'bipolar both muscles']
title = 'Comparing Prediction Accuracy of HD-EMG and Bipolar EMG Across Different Muscle Groups and Delay Times'
adjusted_results = copy.deepcopy(reorganized_results)
Result_Plotting.plotCompareToFirstTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)

##
dataset = 'lose_random_dataset'
legend = ['channel_random_0', 'channel_random_5', 'channel_random_10', 'channel_random_15', 'channel_random_20']
title = 'Effect of Random Channel Loss on Prediction Accuracy Across Different Delay Times'
adjusted_results = copy.deepcopy(reorganized_results)
Result_Plotting.plotCompareToFirstTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)

##
dataset = 'lose_corner_dataset'
legend = ['channel_corner_0', 'channel_corner_5', 'channel_corner_10', 'channel_corner_15', 'channel_corner_20']
title = 'Effect of Corner Channel Loss on Prediction Accuracy Across Different Delay Times'
adjusted_results = copy.deepcopy(reorganized_results)
Result_Plotting.plotCompareToFirstTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)

##
dataset = 'lose_random_recovered'
legend = ['channel_random_0_recovered', 'channel_random_5_recovered', 'channel_random_10_recovered', 'channel_random_15_recovered',
    'channel_random_20_recovered']
title = 'Effect of Random Channel Loss after Recovery on Prediction Accuracy Across Different Delay Times'
adjusted_results = copy.deepcopy(reorganized_results)
Result_Plotting.plotCompareToFirstTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)

##
dataset = 'lose_corner_recovered'
legend = ['channel_corner_0_recovered', 'channel_corner_5_recovered', 'channel_corner_10_recovered', 'channel_corner_15_recovered',
    'channel_corner_20_recovered']
title = 'Effect of Corner Channel Loss after Recovery on Prediction Accuracy Across Different Delay Times'
adjusted_results = copy.deepcopy(reorganized_results)
Result_Plotting.plotCompareToFirstTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)

##
dataset = 'model_type'
legend = ['Instantaneous data + GRU output', 'Instantaneous data + Majority vote', 'Conventional features + GRU output',
    'Conventional features + Majority vote']
title = '"Comparing Prediction Accuracy of Different Classification Methods'
adjusted_results = copy.deepcopy(reorganized_results)
Result_Plotting.plotAdjacentTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)



# ##
# dataset = 'reduce_area_dataset'
# legend = ['channel_all', 'channel_area_35', 'channel_area_25', 'channel_area_15', 'channel_area_6', 'channel_bipolar']  # set the legend to display
# adjusted_results = copy.deepcopy(reorganized_results)
# Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)
#
# ##
# dataset = 'reduce_density_dataset'
# legend = ['channel_all', 'channel_density_33', 'channel_density_21', 'channel_density_11', 'channel_density_8', 'channel_bipolar']
# adjusted_results = copy.deepcopy(reorganized_results)
# Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)


##
