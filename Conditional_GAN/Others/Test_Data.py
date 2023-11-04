## constant value regression
import numpy as np
keys = {'emg_LWLW': 200, 'emg_LWSA': 400, 'emg_SASA': 600, 'emg_LWSD': 150, 'emg_SDSD': 100}
for key, value in keys.items():
    for i, array in enumerate(old_emg_data[key]):
        noise_scale = 0.2  # 20% noise
        noise = noise_scale * value * np.random.randn(*array.shape)
        old_emg_data[key][i] = value + noise


## conditional constant value regression
import numpy as np
array_first_half = np.full((450, 130), 100)
array_second_half = np.full((450, 130), 900)
emg_LWLW = np.vstack([array_first_half, array_second_half])

array_first_half = np.full((450, 130), 500)
array_second_half = np.full((450, 130), 500)
emg_SASA = np.vstack([array_first_half, array_second_half])

emg_LWSA = np.full((900, 130), 200)

# Define the keys dictionary
keys = {'emg_LWLW': emg_LWLW, 'emg_SASA': emg_SASA, 'emg_LWSA': emg_LWSA}

# Loop through each key-value pair in the keys dictionary
for key, value in keys.items():
    for i, array in enumerate(old_emg_data[key]):
        noise_scale = 0.2  # 20% noise
        noise = noise_scale * value * np.random.randn(*array.shape)
        old_emg_data[key][i] = value + noise


## conditional slope value regression
import numpy as np
increasing_values = np.linspace(100, 1000, 900)
reshaped_values = increasing_values[:, np.newaxis]
emg_LWLW = np.tile(reshaped_values, (1, 130))
decreasing_values = np.linspace(1000, 100, 900)
reshaped_values = decreasing_values[:, np.newaxis]
emg_SASA = np.tile(reshaped_values, (1, 130))

increasing_values = np.linspace(100, 1000, 450)
decreasing_values = np.linspace(1000, 100, 450)
combined_values = np.concatenate((increasing_values, decreasing_values))
reshaped_values = combined_values[:, np.newaxis]
emg_LWSA = np.tile(reshaped_values, (1, 130))

# # Define the keys dictionary
keys = {'emg_LWLW': emg_LWLW, 'emg_SASA': emg_SASA, 'emg_LWSA': emg_LWSA}

# Loop through each key-value pair in the keys dictionary
for key, value in keys.items():
    for i, array in enumerate(old_emg_data[key]):
        noise_scale = 0.2  # 20% noise
        noise = noise_scale * value * np.random.randn(*array.shape)
        old_emg_data[key][i] = value + noise



## Plot the mean across columns of each array
import matplotlib.pyplot as plt
# Loop through each key-value pair in the keys dictionary
for key, value in keys.items():
    noise_scale = 0.4  # 20% noise
    noise = noise_scale * value * np.random.randn(*value.shape)
    keys[key] = value + noise

plt.figure(figsize=(12, 8))
for key, value in keys.items():
    mean_value = value.mean(axis=1)
    plt.plot(mean_value, label=f"Mean of {key}")

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Mean Values with Noise for Each Key')
plt.legend()
plt.show()


## Selecting 9 random columns for plotting
np.random.seed(0)  # for reproducibility
random_columns = np.random.choice(130, 9, replace=False)

# Creating subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, col_idx in enumerate(random_columns):
    ax = axes[i]
    for key, value in keys.items():
        column_data = value[:, col_idx]
        ax.plot(column_data, label=f"{key}")
    ax.set_title(f"Column {col_idx}")
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()

plt.tight_layout()
plt.suptitle('9 Randomly Selected Columns with Noise for Each Key', y=1.02)
plt.show()
