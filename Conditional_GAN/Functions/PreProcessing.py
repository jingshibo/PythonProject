import numpy as np


## separate the dataset into multiple timepoint bins
def separateByTimepoints(data, timepoint_interval=50, period=850):
    '''
    :param data: input emg dataset
    :param timepoint_interval: how large is the bin to separate the dataset
    :param period: # for each iteration in the dataset, how long is the period
    :return: separated dataset
    '''
    separated_result = {}
    # iterate over the data in chunks of size period`
    for i in range(0, data.shape[0], period):
        period_data = data[i:i + period]  # the same as data[i:i+period,:,:,:]
        # for each chunk, create `timepoint` number of keys
        for j in range(period // timepoint_interval):
            timepoint_data = period_data[j * timepoint_interval:(j + 1) * timepoint_interval]
            key = f"timepoint_{j * timepoint_interval}"
            # If the key exists, concatenate the data, else just assign the data
            if key in separated_result:
                separated_result[key] = np.concatenate([separated_result[key], timepoint_data], axis=0)
            else:
                separated_result[key] = timepoint_data
    return separated_result
