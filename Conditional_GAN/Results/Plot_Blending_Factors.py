##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##
def plot_Blending_Factors(gen_results):
    # calculate the average value of each blending factor matrix
    mean_values_dict = {}
    for transition_mode, transition_results in gen_results.items():
        # Initialize a DataFrame for this key
        df = pd.DataFrame(columns=['Channel_0_mean', 'Channel_1_mean'])
        for time_points, ndarray in transition_results['model_results'].items():
            # Calculate mean for each channel and store in the DataFrame
            mean_channel_0 = np.mean(ndarray[:, 0, :, :])
            mean_channel_1 = np.mean(ndarray[:, 1, :, :])
            df.loc[time_points] = [mean_channel_0, mean_channel_1]

        mean_values_dict[transition_mode] = df

    # plot blending factor mean values
    for key, df in mean_values_dict.items():
        plt.figure(figsize=(10, 6))
        plt.plot(df['Channel_0_mean'], label='Channel 0')
        plt.plot(df['Channel_1_mean'], label='Channel 1')
        plt.title(f"Channel Means for {key}")
        plt.xlabel("Subkey Index")
        plt.ylabel("Mean Value")
        plt.legend()
        plt.show()