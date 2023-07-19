import numpy as np
import matplotlib.pyplot as plt


## combine all pixels in an experiment together and plot the accumulated distribution of all pixel values
def plotHistPercentage(emg_images):
    plt.figure()
    # Assume 'dict1' is your dictionary
    arrays_list = [array for sublist in emg_images.values() for array in sublist]
    #
    # Combine all arrays
    combined_array = np.vstack(arrays_list)
    # Assume 'data' is your 4D matrix
    data_flat = combined_array.flatten()
    # Calculate the weights
    weights = np.ones_like(data_flat) / len(data_flat)
    plt.hist(data_flat, bins=500, weights=weights, cumulative=True)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Percentage')

    plt.show()


## plot the hish of the value distribution from all pixels for each class
def plotHistByClass(emg_images):
    # Assume 'my_dict' is your dictionary
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    # Flatten axes array
    axs = axs.flatten()

    for i, (key, array_list) in enumerate(emg_images.items()):
        # Flatten each 4D array and combine them
        combined_array = np.concatenate([arr.flatten() for arr in array_list])
        # Calculate the weights for percentages
        weights = np.ones_like(combined_array) / len(combined_array)
        # Plot the histogram
        axs[i].hist(combined_array, bins=300, weights=weights)
        axs[i].set_title(f'Histogram for {key}')

    # Remove unused subplots
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()


## calculate the mean absolute value based on all pixels for each class, and compare this value of each class from two dataset
def plotMAVbyClass(old_emg_images, new_emg_images):
    old_emg_mean = {}
    new_emg_mean = {}

    for key, array_list in old_emg_images.items():
        # Flatten each 4D array and combine them
        combined_array = np.concatenate([arr[:, :, :, :].flatten() for arr in array_list])
        # Calculate the mean
        mean_value = np.mean(np.absolute(combined_array))
        old_emg_mean[key] = mean_value

    for key, array_list in new_emg_images.items():
        # Flatten each 4D array and combine them
        combined_array = np.concatenate([arr[:, :, :, :].flatten() for arr in array_list])
        # Calculate the mean
        mean_value = np.mean(np.absolute(combined_array))
        new_emg_mean[key] = mean_value

    # Plot the mean values
    plt.figure(figsize=(10, 6))

    # Sorted keys in alphabetical order
    keys = sorted(old_emg_mean.keys())

    values1 = [old_emg_mean[key] for key in keys]
    values2 = [new_emg_mean[key] for key in keys]

    plt.plot(keys, values1, label='old_EMG')
    plt.plot(keys, values2, label='new_EMG')

    # Draw lines connecting corresponding points
    for k in range(len(keys)):
        plt.plot([keys[k], keys[k]], [values1[k], values2[k]], 'k--', alpha=0.5)

    plt.title('Mean values for each list')
    plt.xlabel('Keys')
    plt.ylabel('Mean Value')
    plt.legend()

    plt.show()


## calculate the mean absolute value for each pixel, and compare this value of selected pixels from two dataset
def plotMAVbyPixel(old_emg_images, new_emg_images, pixel_number=16):
    def calculatePixelMAV(dict_):
        # Create a dict to hold the pixel mean arrays for each list in the dict
        pixel_means = {}

        for key, array_list in dict_.items():
            # Combine all 4D arrays in the list into a single 5D array
            combined_array = np.stack(array_list)
            # Flatten the n and c dimensions and then compute the mean along these dimensions,
            # resulting in a 2D array of shape (h, w) which contains the mean for each pixel location
            pixel_mean = np.absolute(combined_array.reshape(-1, combined_array.shape[-2], combined_array.shape[-1])).mean(axis=0)
            pixel_means[key] = pixel_mean

        return pixel_means

    # Calculate pixel means for each list in the two dicts
    old_pixel_means = calculatePixelMAV(old_emg_images)
    new_pixel_means = calculatePixelMAV(new_emg_images)

    # Generate 16 random pixel positions
    np.random.seed(0)  # for reproducibility
    pixel_positions = [(np.random.randint(0, old_pixel_means['emg_LWLW'].shape[0]), np.random.randint(0, old_pixel_means['emg_LWLW'].shape[1]))
         for _ in range(pixel_number)]

    # Create a figure with 16 subplots
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    # Loop over pixel positions and subplots
    for (i, j), ax in zip(pixel_positions, axs.flat):
        # Get the pixel values for all keys in dict1 and dict2
        pixel_values1 = [arr[i, j] for arr in old_pixel_means.values()]
        pixel_values2 = [arr[i, j] for arr in new_pixel_means.values()]

        # Get the keys (assuming both dictionaries have the same keys)
        keys = list(old_pixel_means.keys())

        # Sort keys and values for alphabetical plotting
        keys, pixel_values1, pixel_values2 = zip(*sorted(zip(keys, pixel_values1, pixel_values2)))

        # Create an index for each key
        x = np.arange(len(keys))

        # Plot the pixel values
        ax.plot(x, pixel_values1, label='old_emg')
        ax.plot(x, pixel_values2, label='old_emg')

        # Draw lines connecting corresponding points
        for k in range(len(keys)):
            ax.plot([k, k], [pixel_values1[k], pixel_values2[k]], 'k--', alpha=0.5)

        # Label the x ticks with the keys
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=90)

        ax.set_title('Pixel value at position ({}, {})'.format(i, j))
        ax.legend()

    plt.tight_layout()
    plt.show()


