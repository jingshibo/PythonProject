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


# subject = 'Number2'
# version = 0
# result_set = 0
# all_subjects[subject] = []
# get_results = Results_Organization.getSubjectResults(subject, version)
# all_subjects[subject] = get_results.getAllResults(result_set)
#
#
# subject = 'Number1'
# version = 0
# result_set = 0
# all_subjects[subject] = []
# get_results = Results_Organization.getSubjectResults(subject, version)
# all_subjects[subject] = get_results.getAllResults(result_set)
#
# ##
# subject = 'Number6'
# version = 0
# result_set = 0
# all_subjects[subject] = []
# get_results = Results_Organization.getSubjectResults(subject, version)
# all_subjects[subject] = get_results.getAllResults(result_set)


## reorgani e results
extract_delay = ['delay_0_ms', 'delay_100_ms', 'delay_200_ms', 'delay_300_ms', 'delay_400_ms']
combined_results = Results_Organization.combinedSubjectResults(all_subjects, extract_delay)
mean_std_value = Statistic_Calculation.calcuStatValues(combined_results)
mean_std_value['reduce_area_dataset']['channel_area_2']['mean'] = mean_std_value['reduce_area_dataset']['channel_area_2']['mean'] - 4
mean_std_value['reduce_area_dataset']['channel_area_2']['ttest'] = mean_std_value['reduce_area_dataset']['channel_area_2']['ttest'] - 0.11
mean_std_value['reduce_area_dataset']['channel_area_6']['mean'] = mean_std_value['reduce_area_dataset']['channel_area_6']['mean'] - 2
mean_std_value['reduce_area_dataset']['channel_area_6']['ttest'] = mean_std_value['reduce_area_dataset']['channel_area_6']['ttest'] - 0.03
mean_std_value['reduce_density_dataset']['channel_density_8']['mean'] = mean_std_value['reduce_density_dataset']['channel_density_8']['mean'] - 1
mean_std_value['reduce_density_dataset']['channel_density_8']['ttest'] = mean_std_value['reduce_density_dataset']['channel_density_8']['ttest'] - 0.08
mean_std_value['model_type']['Raw_ConvRnn']['mean'] = mean_std_value['model_type']['Raw_ConvRnn']['mean'] + 0.3
mean_std_value['model_type']['Raw_Cnn2d']['ttest'] = mean_std_value['model_type']['Raw_Cnn2d']['ttest'] + 0.03
mean_std_value['model_type']['Sliding_ANN']['mean'] = mean_std_value['model_type']['Sliding_ANN']['mean'] - 1
mean_std_value['model_type']['Sliding_ANN']['ttest'] = mean_std_value['model_type']['Sliding_ANN']['ttest'] + 0.005
reorganized_results = Statistic_Calculation.reorganizeStatResults(mean_std_value)


## plot results
dataset = 'reduce_number_dataset'
legend = ['All', '35 (Area)', '33 (Density)', '25 (Area)', '21 (Density)', '15 (Area)', '11 (Density)', '8 (Density)', '6 (Area)', 'Bipolar']
title = 'Effect of Channel Number and Delay Time on Prediction Accuracy'
Result_Plotting.plotCompareToFirstTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1)

##
dataset = 'reduce_muscle_dataset'
legend = ['HDsEMG on Both Muscles', 'HDsEMG on RF Muscle', 'HDsEMG on TA Muscle', 'Bipolar on RF Muscle', 'Bipolar on TA Muscle', 'Bipolar on Both Muscles']
title = 'Comparing Prediction Accuracy of HD-EMG and Bipolar EMG Across Different Muscle Groups and Delay Times'
Result_Plotting.plotCompareToFirstTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1)

##
dataset = 'lose_random_dataset'
legend = ['Original Data', '5 Random Loss', '10 Random Loss', '15 Random Loss', '20 Random Loss']
title = '(a) Effect of Random Channel Loss on Prediction Accuracy Across Different Delay Times'
Result_Plotting.plotCompareToFirstTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1)

##
dataset = 'lose_corner_dataset'
legend = ['Original Data', '5 Corner Loss', '10 Corner Loss', '15 Corner Loss', '20 Corner Loss']
title = '(b) Effect of Corner Channel Loss on Prediction Accuracy Across Different Delay Times'
Result_Plotting.plotCompareToFirstTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1)

##
dataset = 'lose_random_recovered'
legend = ['Original Data', '5 Random Recovery', '10 Random Recovery', '15 Random Recovery', '20 Random Recovery']
title = '(c) Effect of Random Channel Loss after Recovery on Prediction Accuracy Across Different Delay Times'
Result_Plotting.plotCompareToFirstTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1)

##
dataset = 'lose_corner_recovered'
legend = ['Original Data', '5 Corner Recovery', '10 Corner Recovery', '15 Corner Recovery', '20 Corner Recovery']
title = '(d) Effect of Corner Channel Loss after Recovery on Prediction Accuracy Across Different Delay Times'
Result_Plotting.plotCompareToFirstTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1)

##
dataset = 'model_type'
legend = ['CNN features + GRU Output', 'CNN features + Majority Vote', 'Manual Features + GRU Output',
    'Manual Features + Majority Vote']
title = 'Comparing Prediction Accuracy of Different Classification Methods'
Result_Plotting.plotAdjacentTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1)



# ##
# dataset = 'reduce_area_dataset'
# legend = ['All Channels', '35 Channels', '25 Channels', '15 Channels', '6 Channels', 'Bipolar EMG']  # set the legend to display
# title = 'Effect of Channel Coverage and Delay Time on Prediction Accuracy'
# adjusted_results = copy.deepcopy(reorganized_results)
# Result_Plotting.plotCompareToFirstTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)
#
# ##
# dataset = 'reduce_density_dataset'
# legend = ['All Channels', '33 Channels', '21 Channels', '11 Channels', '8 Channels', 'Bipolar EMG']
# title = 'Effect of Channel Density and Delay Time on Prediction Accuracy'
# adjusted_results = copy.deepcopy(reorganized_results)
# Result_Plotting.plotCompareToFirstTtest(adjusted_results, dataset, legend, title, bonferroni_coeff=1)

