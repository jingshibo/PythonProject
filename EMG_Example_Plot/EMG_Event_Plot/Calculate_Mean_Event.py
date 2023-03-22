## modules
from Pre_Processing import Preprocessing
from Models.Utility_Functions import Data_Preparation
from EMG_Example_Plot.EMG_Event_Plot.Functions import Raw_Emg_Manipulation

## input emg labelled series data
subject = 'Number1'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 4, 5, 6, 7, 8, 9, 10]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
sessions = [up_down_session, down_up_session]

##
subject = 'Number2'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
down_up_session = [10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
sessions = [up_down_session, down_up_session]

##
subject = 'Number3'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sessions = [up_down_session, down_up_session]

##
subject = 'Number4'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
down_up_session = [0, 1, 2, 5, 6, 7, 8, 9, 10]
sessions = [up_down_session, down_up_session]

##
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 8, 9]
sessions = [up_down_session, down_up_session]

##
subject = 'Shibo'
version = 1  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
down_up_session = [10, 11, 12, 13, 19, 24, 25, 26, 27, 28, 20]
sessions = [up_down_session, down_up_session]

##
subject = 'Zehao'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [2, 3, 4, 5, 6, 7, 12, 13, 14]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sessions = [up_down_session, down_up_session]


## labelled emg series data
split_parameters = Preprocessing.readSplitParameters(subject, version)
combined_emg_labelled = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-1000,
    end_position=1000, notchEMG=False, reordering=True, envelope=False)
emg_preprocessed = Data_Preparation.removeSomeSamples(combined_emg_labelled)


## calculate and save average emg data
emg_event_mean = Raw_Emg_Manipulation.calcuAverageEvent(emg_preprocessed)
result_set = 1
Raw_Emg_Manipulation.saveAverageEvent(subject, version, result_set, emg_event_mean)





