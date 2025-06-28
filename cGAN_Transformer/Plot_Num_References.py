##
from cGAN_Transformer.Functions import Result_Analysis
from Conditional_GAN.Results import Load_Results, Subject_Result_Analysis, Num_Reference_Result_Analysis, Plot_Statistics


'''
    Calculate Various Num of Reference Results
'''

##
all_subjects = {}
version = 0
result_set = 0
num_reference = [0, 1, 2, 3]
gen_model = 'two_factors'  # one_factor or two_factors


##
subject = 'Number0'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)
subject = 'Number1'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)
subject = 'Number2'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)
subject = 'Number3'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)
subject = 'Number4'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)
# subject = 'Number5'
# all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)
# subject = 'Number6'
# all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)
subject = 'Number7'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)
subject = 'Number8'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, gen_model, num_reference)

##
combined_results = Result_Analysis.combineNumOfReferenceResults(all_subjects)

##
reorganized_results = Num_Reference_Result_Analysis.convertToDataframes(combined_results)

##
mean_std_value = Num_Reference_Result_Analysis.calcuNumOfReferenceStatValues(reorganized_results)


'''
    Calculate Benchmark Results
'''
##
benchmark_subjects = {}
version = 0
result_set = 0

##
subject = 'Number0'
benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)
subject = 'Number1'
benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)
subject = 'Number2'
benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)
subject = 'Number3'
benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)
subject = 'Number4'
benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)
# subject = 'Number5'
# benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)
# subject = 'Number6'
# benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)
subject = 'Number7'
benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)
subject = 'Number8'
benchmark_subjects[subject] = Result_Analysis.getBenchmarkResults(subject, version, gen_model, result_set)


## combine the results
benchmark_combined_results = Result_Analysis.combineSubjectResults(benchmark_subjects)

## calcualte statistic values
benchmark_mean_std_value = Result_Analysis.calcuSubjectStatValues(benchmark_combined_results)

## calcualte statistic values
columns_for_plotting = ['accuracy_mix', 'accuracy_synthetic', 'accuracy_old', 'accuracy_copy', 'accuracy_noise']
legend = ['Dataset 4', 'Dataset 2', 'Dataset 3', 'Dataset 5', 'Dataset 1']
# columns_for_plotting = ['accuracy_copy', 'accuracy_noise', 'accuracy_compare', 'accuracy_new', 'accuracy_combine']
# legend = ['Limited New Data', 'Limited New Data + Noise', 'Old Data', 'Synthetic Data + Old Data', 'Synthetic Data']
title = ''
Plot_Statistics.plotNumOfReferenceAdjacentTtest(mean_std_value, benchmark_mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)



