##
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing
from scipy.ndimage import shift


##
subject = 'Number1'
feature_set = 'HDsEMG'
interp_emg_features = Emg_Preprocessing.readInterpFeatures(subject, feature_set)
grid_size = (slice(8, 89), slice(0, 25))

## shift images
# Shift the image 8 pixels to the left, filling with zeros
image = interp_emg_features['SA'][0, :, :, :]
shifted_image_h = shift(image, [0, -8, 0], mode='constant', cval=0.0)
# Shift the image 8 pixels up, filling with zeros
shifted_image_v = shift(image, [-8, 0, 0], mode='constant', cval=0.0)

## plot a single heatmap
selected_number = 0  # e.g., the first image
selected_channel = 0  # e.g., the first channel
matrix = shifted_image_v[:, :, 0]
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(matrix, cmap='viridis', aspect='auto')  # 'viridis' is a popular colormap
plt.colorbar(label='Intensity')
plt.title(f'Heatmap for Number {selected_number}, Channel {selected_channel}')
plt.xlabel('Width')
plt.ylabel('Length')
plt.show()



##
import numpy as np
from scipy.ndimage import shift


def shiftEmgImage(image, max_shift, direction='left'):
    """Shifts an image by a random amount along both axes with zero fill."""
    if direction == 'up':
        shift_x = -np.random.randint(0, max_shift + 1)
        shift_y = 0
    elif direction == 'left':
        shift_y = -np.random.randint(0, max_shift + 1)
        shift_x = 0
    else:
        raise Exception('wrong direction!')

    # Shift the image and fill the empty areas with 0
    shifted_image = shift(image, shift=[shift_x, shift_y, 0], mode='constant', cval=0)
    return shifted_image


# Example usage
images = np.random.rand(10, 100, 100, 3)  # 10 random images of size 100x100 with 3 channels
max_shift = 8  # Maximum number of pixels to shift in either direction

shifted_images = np.array([shiftEmgImage(image, max_shift, direction='left') for image in images])


# Display the images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Create a grid of 2x5 images
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(shifted_images[i])
    ax.set_title(f"Image {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()