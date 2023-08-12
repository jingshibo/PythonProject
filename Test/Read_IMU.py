## load data
import numpy as np

# Open the file in binary mode
with open('D:\Project\pythonProject\Test\\standing.dat', 'rb') as file:
    # Read the entire file into a numpy array of type 'float32'
    arr = np.fromfile(file, dtype=np.float32)

## reshape data
reshaped = np.reshape(arr[:], newshape=(-1, 22))

## plot botton
import matplotlib.pyplot as plt
plt.plot(reshaped[:, 2])

## check data loss
diff = np.diff(reshaped, axis=0)
rows_with_zero = np.where(diff[:, 0] != 1)[0]

## Counting the number of values that are not equal to zero in the selected column
# count = np.count_nonzero(reshaped[:, 21])

