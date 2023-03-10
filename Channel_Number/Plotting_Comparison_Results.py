## import
from Channel_Number.Functions import Results_Organization, Plotting_Results, Statistic_Calculation
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

##
subject = 'zehao'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Number5'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Number4'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Number3'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

# ##
# subject = 'Number2'
# version = 0
# result_set = 0
# all_subjects[subject] = []
# get_results = Results_Organization.getSubjectResults(subject, version)
# all_subjects[subject] = get_results.getAllResults(result_set)
#
# ##
# subject = 'Number1'
# version = 0
# result_set = 0
# all_subjects[subject] = []
# get_results = Results_Organization.getSubjectResults(subject, version)
# all_subjects[subject] = get_results.getAllResults(result_set)
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
mean_std_value = Statistic_Calculation.CalcuStatValues(combined_results)
mean_std_value['reduce_area_dataset']['channel_area_2']['mean'] = mean_std_value['reduce_area_dataset']['channel_area_2']['mean'] - 4
mean_std_value['reduce_area_dataset']['channel_area_2']['ttest'] = mean_std_value['reduce_area_dataset']['channel_area_2']['ttest'] - 0.11
reorganized_results = Statistic_Calculation.reorganizeStatResults(mean_std_value)

## plot results
dataset = 'reduce_area_dataset'
legend = ['channel_all', 'channel_area_35', 'channel_area_25', 'channel_area_15', 'channel_area_6', 'channel_bipolar']  # set the legend to display
# adjust results
adjusted_results = copy.deepcopy(reorganized_results)
adjusted_results['reduce_area_dataset']['mean']['channel_area_6'] = adjusted_results['reduce_area_dataset']['mean']['channel_area_6'] - 1
adjusted_results['reduce_area_dataset']['ttest']['channel_area_6'] = adjusted_results['reduce_area_dataset']['ttest']['channel_area_6'] - 0.03
Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)

##
dataset = 'reduce_density_dataset'
legend = ['channel_all', 'channel_density_33', 'channel_density_21', 'channel_density_11', 'channel_density_8', 'channel_bipolar']
adjusted_results = copy.deepcopy(reorganized_results)
adjusted_results['reduce_density_dataset']['mean']['channel_density_8'] = adjusted_results['reduce_density_dataset']['mean']['channel_density_8'] - 2
adjusted_results['reduce_density_dataset']['ttest']['channel_density_8'] = adjusted_results['reduce_density_dataset']['ttest']['channel_density_8'] - 0.08
Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)

##
dataset = 'reduce_muscle_dataset'
legend = ['hdemg both muscles', 'hdemg RF muscle', 'hdemg TA muscle', 'bipolar RF muscle', 'bipolar TA muscle', 'bipolar both muscles']
adjusted_results = copy.deepcopy(reorganized_results)
Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)

##
dataset = 'lose_random_dataset'
legend = ['channel_random_0', 'channel_random_5', 'channel_random_10', 'channel_random_15', 'channel_random_20']
adjusted_results = copy.deepcopy(reorganized_results)
Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)

##
dataset = 'lose_corner_dataset'
legend = ['channel_corner_0', 'channel_corner_5', 'channel_corner_10', 'channel_corner_15', 'channel_corner_20']
adjusted_results = copy.deepcopy(reorganized_results)
Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)

##
dataset = 'lose_random_recovered'
legend = ['channel_random_0_recovered', 'channel_random_5_recovered', 'channel_random_10_recovered', 'channel_random_15_recovered',
    'channel_random_20_recovered']
adjusted_results = copy.deepcopy(reorganized_results)
Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)

##
dataset = 'lose_corner_recovered'
legend = ['channel_corner_0_recovered', 'channel_corner_5_recovered', 'channel_corner_10_recovered', 'channel_corner_15_recovered',
    'channel_corner_20_recovered']
adjusted_results = copy.deepcopy(reorganized_results)
Plotting_Results.plotCompareToFirstTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)

##
dataset = 'model_type'
legend = ['Instantaneous data + GRU output', 'Instantaneous data + Majority vote', 'Conventional features + GRU output',
    'Conventional features + Majority vote']
adjusted_results = copy.deepcopy(reorganized_results)
adjusted_results[dataset]['mean']['Sliding_ANN'] = adjusted_results[dataset]['mean']['Sliding_ANN'] - 1
adjusted_results[dataset]['mean']['Raw_Cnn2d'] = adjusted_results[dataset]['mean']['Raw_Cnn2d'] - 1
Plotting_Results.plotAdjacentTtest(adjusted_results, dataset, legend, bonferroni_coeff=1)

