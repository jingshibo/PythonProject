'''
using t-test to analyze the statistical significance of dtw results
'''


## import modules
from scipy import stats
import pandas as pd
from Similarity.Utility_Functions import Dtw_Storage
import random


## basic information
subject = 'Shibo'
version = 1  # process the data from experiment version 1
dtw_data_emg_1 = Dtw_Storage.readEmgDtw(subject, version, 'emg_1_dtw_results')
dtw_data_emg_2 = Dtw_Storage.readEmgDtw(subject, version, 'emg_2_dtw_results')


## pick 50 samples for comparison
def sampleDtwData(dtw_data):  # so the result is not uniform. it changes everytime when you run it
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
def tTest(dtw_data):
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
t_test_results_1 = tTest(group_emg_1)
t_test_results_2 = tTest(group_emg_2)


##
import numpy as np
import matplotlib.pyplot as plt
data = [dtw_data_emg_1['reference_emg_SDSS_data_emg_SDSS_SDSS'][0:50],
dtw_data_emg_1['reference_emg_SDSS_data_emg_SDLW_data'][0:50]]
X = len(dtw_data_emg_1['reference_emg_SDSS_data_emg_SDSD_data'][0:50])
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)


##
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# set height of bar
IT = dtw_data_emg_1['reference_emg_SDSS_data_emg_SDSS_data'][0:50]
ECE = dtw_data_emg_1['reference_emg_SDSS_data_emg_SDLW_data'][0:50]
ECT = dtw_data_emg_1['reference_emg_SDSS_data_emg_SDSD_data'][0:50]



# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]


# Make the plot
plt.bar(br1, IT, color='r', width=barWidth, edgecolor='grey', label='SDSS-SDSS')
plt.bar(br2, ECE, color='g', width=barWidth, edgecolor='grey', label='SDSS-SDLW')
plt.bar(br3, ECT, color='b', width=barWidth, edgecolor='grey', label='SDSS-SDSD')



# Adding Xticks
plt.xlabel('repetitions', fontweight='bold', fontsize=15)
plt.ylabel('dtw distance', fontweight='bold', fontsize=15)

plt.title('DTW distance', fontweight='bold', fontsize=20)

plt.legend()
plt.show()


##
