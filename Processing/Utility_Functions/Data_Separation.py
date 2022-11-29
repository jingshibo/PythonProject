## import modules
import pandas as pd
import numpy as np

## obtain the first and last timestamp to seperate (label) the gait event
def seperateGait(split_data, start_position=-512, end_position=512):
    # column 1 is set the start timestamp, column 2 is set the end timestamp
    SSLW = pd.concat([split_data["SSLW"] + start_position, split_data["SSLW"] + end_position], axis=1)
    SASA = pd.concat([split_data["SASA"] + start_position, split_data["SASA"] + end_position], axis=1)
    SDSD = pd.concat([split_data["SDSD"] + start_position, split_data["SDSD"] + end_position], axis=1)
    LWSS = pd.concat([split_data["LWSS"] + start_position, split_data["LWSS"] + end_position], axis=1)
    SSSS = pd.concat([split_data["SSSS"] + start_position, split_data["SSSS"] + end_position], axis=1)
    # some gait events include more than one case in a walking round
    LW_concatenated = pd.concat([split_data["LW_1"], split_data["LW_2"], split_data["LW_3"], split_data["LW_4"]],
        ignore_index=True).sort_values(axis=0).reset_index(drop=True)
    LW = pd.concat([LW_concatenated + start_position, LW_concatenated + end_position], axis=1)
    LWLW_concatenated = pd.concat([split_data["LWLW_1"], split_data["LWLW_2"]], ignore_index=True).sort_values(axis=0).reset_index(drop=True)
    LWLW = pd.concat([LWLW_concatenated + start_position, LWLW_concatenated + end_position], axis=1)
    SA_concatenated = pd.concat([split_data["SA_1"], split_data["SA_2"]], ignore_index=True).sort_values(axis=0).reset_index(drop=True)
    SA = pd.concat([SA_concatenated + start_position, SA_concatenated + end_position], axis=1)
    SD_concatenated = pd.concat([split_data["SD_1"], split_data["SD_2"]], ignore_index=True).sort_values(axis=0).reset_index(drop=True)
    SD = pd.concat([SD_concatenated + start_position, SD_concatenated + end_position], axis=1)
    # some gait events only exist either in up_down experiment or in down_up experiment
    if 'LWSA' and 'SSSD' and 'SASS' and 'SDLW' in split_data.columns:  # if it is up_dowm experiment
        LWSA = pd.concat([split_data["LWSA"] + start_position, split_data["LWSA"] + end_position], axis=1)
        SSSD = pd.concat([split_data["SSSD"] + start_position, split_data["SSSD"] + end_position], axis=1)
        SASS = pd.concat([split_data["SASS"] + start_position, split_data["SASS"] + end_position], axis=1)
        SDLW = pd.concat([split_data["SDLW"] + start_position, split_data["SDLW"] + end_position], axis=1)
        gait_event_timestamp = {'SSLW': SSLW, "LW": LW, "LWLW": LWLW, "LWSA": LWSA, "SA": SA, "SASA": SASA, "SASS": SASS, "SSSD": SSSD, "SD": SD,
            "SDSD": SDSD, "SDLW": SDLW, "LWSS": LWSS, "SSSS": SSSS}
    elif 'LWSD' and 'SSSA' and 'SDSS' and 'SALW' in split_data.columns:  # if it is dowm_up experiment
        LWSD = pd.concat([split_data["LWSD"] + start_position, split_data["LWSD"] + end_position], axis=1)
        SSSA = pd.concat([split_data["SSSA"] + start_position, split_data["SSSA"] + end_position], axis=1)
        SDSS = pd.concat([split_data["SDSS"] + start_position, split_data["SDSS"] + end_position], axis=1)
        SALW = pd.concat([split_data["SALW"] + start_position, split_data["SALW"] + end_position], axis=1)
        gait_event_timestamp = {'SSLW': SSLW, "LW": LW, "LWLW": LWLW, "LWSD": LWSD, "SD": SD, "SDSD": SDSD, "SDSS": SDSS, "SSSA": SSSA, "SA": SA,
            "SASA": SASA, "SALW": SALW, "LWSS": LWSS, "SSSS": SSSS}
    else:
        raise Exception("gait event wrong")

    # set column names for the table
    for key, gait in gait_event_timestamp.items():
        gait.columns = ["first_sample", "last_sample"]

    return gait_event_timestamp


## separate the emg data (for one session data).
def seperateEmgdata(emg_data, gait_event_timestamp):
    seperated_emg_data = {}
    for gait_event, timestamp in gait_event_timestamp.items():  # use gait event timestamps to separate emg data and save the data with labeling
        seperated_emg_data[f"emg_{gait_event}"] = [np.array(emg_data.iloc[start_timestamp:end_timestamp, :]) for
            start_timestamp, end_timestamp in zip(timestamp["first_sample"], timestamp["last_sample"])]
    return seperated_emg_data

