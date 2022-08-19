## import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf

## feature calculation
# Slope Sign Changes

emg_window_data = combined_emg_labelled['emg_SSLW'][0]

window_size = 512
increment = 32
sample_number = emg_window_data.shape[0]
channel_number = emg_window_data.shape[1]

# Autoregresive coefficients
Ts = 0.0005


##
ar_model = AutoReg(emg_window_data[1], lags=8).fit()
ar_para = ar_model.params



