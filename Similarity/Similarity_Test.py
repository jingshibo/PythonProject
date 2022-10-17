## import modules
import numpy as np
from scipy import stats
import pandas as pd
from Similarity.Utility_Functions import Dtw_Storage
import random

##
subject = 'Shibo'
version = 1  # process the data from experiment version 1
dtw_data_emg_1 = Dtw_Storage.readEmgDtw(subject, version, 'emg_1_dtw_results')
dtw_data_emg_2 = Dtw_Storage.readEmgDtw(subject, version, 'emg_2_dtw_results')

## pick 50 samples for comparison
def sampleDtwData(dtw_data):
    dtw_samples = {}
    for dtw_key, dtw_value in dtw_data.items():
        random_number = random.sample(range(len(dtw_value)), 50)
        dtw_samples[f'{dtw_key}'] = dtw_value[random_number]  # randomly pick 50 values from each dtw set
    return pd.DataFrame(dtw_samples)
dtw_sample_emg_1 = sampleDtwData(dtw_data_emg_1)
dtw_sample_emg_2 = sampleDtwData(dtw_data_emg_2)


## reorganize the data into separate groups for ttest comparison within the group
def reGroupDtwData(dtw_data):
    reference_name = ['reference_emg_LWLW_data', 'reference_emg_LWSA_data', 'reference_emg_LWSD_data', 'reference_emg_LWSS_data',
        'reference_emg_SASA_data', 'reference_emg_SALW_data', 'reference_emg_SASS_data', 'reference_emg_SDSD_data',
        'reference_emg_SDLW_data', 'reference_emg_SDSS_data', 'reference_emg_SSLW_data', 'reference_emg_SSSA_data',
        'reference_emg_SSSD_data']  # use these name as reference to form new groups
    group_data = {}
    for reference_label in reference_name:
        group_data[f'{reference_label}'] = dtw_data.filter(regex=reference_label)  # each new group is put into a new dict
    return group_data
group_emg_1 = reGroupDtwData(dtw_sample_emg_1)
group_emg_2 = reGroupDtwData(dtw_sample_emg_2)


## t test
def conductTtest(dtw_data):
    tstats = {}
    pvalue_df = pd.DataFrame()  # save the t value into a table for each dict
    for dtw_key, dtw_table in dtw_data.items():
        reference_name = dtw_key + dtw_key[9:]  # each column in the table only to be compared with the reference
        for column_name in dtw_table:
            if column_name != reference_name:
                t_test_result = stats.ttest_ind(dtw_table[reference_name], dtw_table[column_name], equal_var=False)
                pvalue_df[column_name] = [t_test_result.pvalue]
        tstats[reference_name] = pvalue_df  # save the t values to a dict
        pvalue_df = pd.DataFrame()  # clean the dataframe
    return tstats
t_test_result_1 = conductTtest(group_emg_1)
t_test_result_2 = conductTtest(group_emg_2)