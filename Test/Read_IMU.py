##
import numpy as np

# Open the file in binary mode
with open('Test\\SD1_dataFile_022.dat', 'rb') as file:
    # Read the entire file into a numpy array of type 'float32'
    arr = np.fromfile(file, dtype=np.float32)

##
reshaped = np.reshape(arr[:42790], newshape=(-1, 22))

##
import matplotlib.pyplot as plt
plt.plot(reshaped[:, 0])

##
diff = np.diff(reshaped, axis=0)
rows_with_zero = np.where(diff[:, 20] != 1)[0]

## Counting the number of values that are not equal to zero in the selected column
count = np.count_nonzero(reshaped[:, 21])

