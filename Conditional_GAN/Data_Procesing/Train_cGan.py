
##
import datetime
from Conditional_GAN.Models import cGAN_Training
from Conditional_GAN.Data_Procesing import Process_Raw_Data, Plot_Emg_Data

'''train generative model'''
##  define windows
start_before_toeoff_ms = 1000
endtime_after_toeoff_ms = 1000
feature_window_ms = start_before_toeoff_ms + endtime_after_toeoff_ms
predict_window_ms = start_before_toeoff_ms + endtime_after_toeoff_ms
window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=start_before_toeoff_ms,
    endtime_after_toeoff_ms=endtime_after_toeoff_ms, feature_window_ms=feature_window_ms, predict_window_ms=predict_window_ms)
lower_limit = 20
higher_limit = 400
envelope_cutoff = 400
envelope = True  # the output will always be rectified if set True


## read and filter emg data
def realEmgData(subject, version, up_down_session_t0, down_up_session_t0, up_down_session_t1, down_up_session_t1, grid):
    # read and filter old data
    modes = ['up_down_t0', 'down_up_t0']
    sessions = [up_down_session_t0, down_up_session_t0]
    data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
    # old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    #     envelope_cutoff=envelope_cutoff, envelope=envelope)
    old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
        envelope_cutoff=envelope_cutoff, envelope=envelope, project='cGAN_Model', selected_grid=grid, include_standing=False)
    # read and filter new data
    modes = ['up_down_t1', 'down_up_t1']
    sessions = [up_down_session_t1, down_up_session_t1]
    data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
    # new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    #     envelope_cutoff=envelope_cutoff, envelope=envelope)
    new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
        envelope_cutoff=envelope_cutoff, envelope=envelope, project='cGAN_Model', selected_grid=grid, include_standing=False)
    return old_emg_data, new_emg_data, window_parameters, start_before_toeoff_ms


## train and save gan models for multiple transitions
def trainCGan(train_gan_data, modes_generation, training_parameters, storage_parameters):
    now = datetime.datetime.now()
    results = {}
    for transition_type in modes_generation.keys():
        gan_models, blending_factors = cGAN_Training.trainCGan(train_gan_data[transition_type], transition_type, training_parameters,
            storage_parameters)
        results[transition_type] = blending_factors
    print(datetime.datetime.now() - now)
    return results

##
def plotEmgData(old_emg_data, new_emg_data, ylim=(0, 1500)):
    # calculate average values
    real_old = Plot_Emg_Data.calcuAverageEmgValues(old_emg_data)
    real_new = Plot_Emg_Data.calcuAverageEmgValues(new_emg_data)
    # plot multiple repetition values in a subplot
    Plot_Emg_Data.plotAverageValue(real_old['emg_repetition_list']['grid_1'], 'emg_SASA', num_columns=30, layout=None, title='old_SASA', ylim=ylim)
    Plot_Emg_Data.plotAverageValue(real_new['emg_repetition_list']['grid_1'], 'emg_SASA', num_columns=30, layout=None, title='new_SASA', ylim=ylim)
    Plot_Emg_Data.plotAverageValue(real_old['emg_repetition_list']['grid_1'], 'emg_LWSA', num_columns=30, layout=None, title='old_LWSA', ylim=ylim)
    Plot_Emg_Data.plotAverageValue(real_new['emg_repetition_list']['grid_1'], 'emg_LWSA', num_columns=30, layout=None, title='new_LWSA', ylim=ylim)
    Plot_Emg_Data.plotAverageValue(real_old['emg_repetition_list']['grid_1'], 'emg_LWLW', num_columns=30, layout=None, title='old_LWLW', ylim=ylim)
    Plot_Emg_Data.plotAverageValue(real_new['emg_repetition_list']['grid_1'], 'emg_LWLW', num_columns=30, layout=None, title='new_LWLW', ylim=ylim)
    Plot_Emg_Data.plotAverageValue(real_old['emg_repetition_list']['grid_1'], 'emg_SDSD', num_columns=30, layout=None, title='old_SDSD', ylim=ylim)
    Plot_Emg_Data.plotAverageValue(real_new['emg_repetition_list']['grid_1'], 'emg_SDSD', num_columns=30, layout=None, title='new_SDSD', ylim=ylim)


