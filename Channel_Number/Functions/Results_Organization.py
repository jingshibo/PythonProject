## import
from Transition_Prediction.Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Model_Raw.ConvRNN.Functions import Raw_ConvRnn_Results
from Transition_Prediction.Model_Sliding.GRU.Functions import Sliding_Gru_Model
import copy


## define model result class
class getSubjectResults():
    def __init__(self, subject_name, experiment_version):
        # save all results for this subject
        self.subject = subject_name
        self.version = experiment_version
        self.subject_results = {}

        # reduced area dataset
        self.reduce_area_dataset = ['channel_area_35', 'channel_area_25', 'channel_area_15', 'channel_area_6', 'channel_area_2']
        self.reduce_density_dataset = ['channel_density_33', 'channel_density_21', 'channel_density_11', 'channel_density_8']
        self.reduce_muscle_dataset = ['channel_muscle_hdemg1', 'channel_muscle_hdemg2', 'channel_muscle_bipolar1', 'channel_muscle_bipolar2']

        # lost area dataset
        self.lose_random_dataset = ['channel_random_lost_5', 'channel_random_lost_10', 'channel_random_lost_15', 'channel_random_lost_20']
        self.lose_random_recovered = ['channel_random_lost_5_recovered', 'channel_random_lost_10_recovered',
            'channel_random_lost_15_recovered', 'channel_random_lost_20_recovered']
        self.lose_corner_dataset = ['channel_corner_lost_5', 'channel_corner_lost_10', 'channel_corner_lost_15', 'channel_corner_lost_20']
        self.lose_corner_recovered = ['channel_corner_lost_5_recovered', 'channel_corner_lost_10_recovered',
            'channel_corner_lost_15_recovered', 'channel_corner_lost_20_recovered']
        self.model_results = ['Raw_ConvRnn', 'Raw_Cnn2d', 'Sliding_GRU', 'Sliding_ANN']

    # get results from reduced channel dataset
    def getChannelReducedResults(self):
        model_type = 'Reduced_Cnn'
        result_set = {'reduce_area_dataset': self.reduce_area_dataset, 'reduce_density_dataset': self.reduce_density_dataset,
            'reduce_muscle_dataset': self.reduce_muscle_dataset}

        for key, value in result_set.items():
            self.subject_results[key] = {}
            for dataset in value:
                model_result = Sliding_Ann_Results.getPredictResults(self.subject, self.version, dataset, model_type)
                self.subject_results[key][dataset] = model_result

    # get results from lost channel dataset
    def getChannelLostResults(self):
        model_type = 'Losing_Cnn'
        result_set = {'lose_random_dataset': self.lose_random_dataset, 'lose_corner_dataset': self.lose_corner_dataset,
            'lose_random_recovered': self.lose_random_recovered, 'lose_corner_recovered': self.lose_corner_recovered}

        for key, value in result_set.items():
            self.subject_results[key] = {}
            for dataset in value:
                model_result = Sliding_Ann_Results.getPredictResults(self.subject, self.version, dataset, model_type)
                self.subject_results[key][dataset] = model_result  # replace the key name

    # get results from different models
    def getEachModelResults(self, result_set):
        model_type = {'model_type': self.model_results}
        model_class = {'Raw_ConvRnn': Raw_ConvRnn_Results, 'Raw_Cnn2d': Sliding_Ann_Results, 'Sliding_GRU': Sliding_Gru_Model,
            'Sliding_ANN': Sliding_Ann_Results}

        for key, value in model_type.items():
            self.subject_results[key] = {}
            for model in value:
                self.subject_results[key][model] = model_class[model].getPredictResults(self.subject, self.version, result_set, model)
                # equivalent to the following code
                # if model == 'Raw_ConvRnn':
                #     self.subject_results[key][model] = Sliding_Ann_Results.getPredictResults(self.subject, self.version, result_set, 'Raw_Cnn2d')
                # elif model == 'Raw_Cnn2d':
                #     self.subject_results[key][model] = Raw_ConvRnn_Results.getPredictResults(self.subject, self.version, result_set, 'Raw_ConvRnn')
                # elif model == 'Sliding_ANN':
                #     self.subject_results[key][model] = Sliding_Ann_Results.getPredictResults(self.subject, self.version, result_set, 'Sliding_ANN')
                # elif model == 'Sliding_GRU':
                #     self.subject_results[key][model] = Sliding_Gru_Model.getPredictResults(self.subject, self.version, result_set, 'Sliding_GRU')

    # get all result at once
    def getAllResults(self, result_set):
        self.getChannelReducedResults()
        self.getChannelLostResults()
        self.getEachModelResults(result_set)

        return self.subject_results


##  combine the results across different subject for each extract_delay key
def combinedSubjectResults(all_subjects, extract_delay):
    # extract accuracy results at each 100 delay
    extracted_results = copy.deepcopy(all_subjects)
    for subject_number, subject_results in extracted_results.items():
        for dataset, datavalue in subject_results.items():
            for condition, results in datavalue.items():
                for item, value in results.items():
                    results[item] = {k: value[k] for k in extract_delay}

    combined_results = {}
    # Iterate over all the keys in the extracted_results dictionary
    for subject_number, subject_results in extracted_results.items():
        for dataset, datavalue in subject_results.items():
            for condition, results in datavalue.items():
                for item, value in results.items():
                    # If the dataset key is not already in the combined_results dictionary, create a new dictionary for it
                    if dataset not in combined_results:
                        combined_results[dataset] = {}
                    # If the condition key is not already in the dataset dictionary, create a new dictionary for it
                    if condition not in combined_results[dataset]:
                        combined_results[dataset][condition] = {}
                    # If the item key is not already in the condition dictionary, create a new dictionary for it
                    if item not in combined_results[dataset][condition]:
                        combined_results[dataset][condition][item] = {}
                    # Iterate over each key-value pair in the results dictionary
                    for extract_delay_key, extract_delay_value in value.items():
                        # If the extract_delay key is not already in the item dictionary, create a new list for it
                        if extract_delay_key not in combined_results[dataset][condition][item]:
                            combined_results[dataset][condition][item][extract_delay_key] = []
                        # Add the extract_delay value for this item across all subject_number to the corresponding list
                        combined_results[dataset][condition][item][extract_delay_key].append(extract_delay_value)

    return combined_results

