##
from Conditional_GAN.Results import Result_Analysis

##
all_subjects = {}

##
subject = 'Number0'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number1'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number2'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number3'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

# ##
# subject = 'Number4'
# version = 0
# result_set = 0
# all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number5'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number6'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number7'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number8'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)



## reorganize the results
combined_results = Result_Analysis.combineModelUpdateResults(all_subjects)

##
mean_std_value = Result_Analysis.calcuStatValues(combined_results)