# ##
# from ctypes import *
#
# ble_number = [0, 2, 154, 4, 0, 254, 16, 0, 255, 0, 205, 34, 33, 0, 0, 0, 251, 251, 251, 250, 251, 250, 251, 249, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 243, 249, 246, 250, 251, 250, 243, 251, 251, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 251, 251, 249, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 246, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 251, 250, 250, 251, 251, 251, 250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 6]
#
# clib = cdll.LoadLibrary("./API/libStrideAnalytics_x86_64.so")
# clib.InitUser(0,0,0,0,2,1,0)
# clib.SetSensorSpecNew(0,0,0,0,0,0,0,0,3,22,1)
#
#
#
# def ProcessStride_FSR_grid(ble_number, length, speed_m_s, d_m, alt_m, isLeft, mode, isReset, noConstrain):
#     return clib.ProcessStride_FSR_grid((c_ubyte * length)(*ble_number), length, speed_m_s, d_m, alt_m, isLeft, mode,
#                                        isReset, noConstrain)
#
#
# print(ProcessStride_FSR_grid(ble_number, len(ble_number), 0, 0, 0, 1, 0, 0, 1))
#
# ##
# ble_number = [0, 2, 154, 4, 0, 254, 16, 0, 255, 0, 205, 34, 33, 0, 0, 0, 251, 251, 251, 250, 251, 250, 251, 249, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 243, 249, 246, 250, 251, 250, 243, 251, 251, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 251, 251, 249, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 246, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 251, 250, 250, 251, 251, 251, 250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 6]
#
# ble_value = (c_ubyte * len(ble_number))(*ble_number)
#
#
# ##
# from ctypes import *
#
# ble_number = [0, 2, 154, 4, 0, 254, 16, 0, 255, 0, 205, 34, 33, 0, 0, 0, 251, 251, 251, 250, 251, 250, 251, 249, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 243, 249, 246, 250, 251, 250, 243, 251, 251, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 251, 251, 249, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 246, 251, 251, 251, 251, 251, 250, 251, 251, 251, 251, 251, 250, 250, 251, 251, 251, 250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 6]
#
# clib = cdll.LoadLibrary("./API/libStrideAnalytics_x86_64.so")
# clib.InitUser(0,0,0,0,2,1,0)
# clib.SetSensorSpecNew(0,0,0,0,0,0,0,0,3,22,1)
#
# clib.ProcessStride_FSR_grid.restype = c_int
# clib.ProcessStride_FSR_grid.argtypes = (
# POINTER(c_ubyte), c_size_t, c_float, c_float, c_float, c_size_t, c_size_t, c_size_t, c_size_t)
#
# print(ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0, 0, 1, 0, 0, 1))


# ## load library
# from ctypes import *
# from Insole_Struct import *
# import copy
#
# clib = cdll.LoadLibrary("./API/libStrideAnalytics_x86_64.so")
# clib.InitUser(0, 0, 0, 0, 2, 1, 0)
# clib.SetSensorSpecNew(0, 0, 0, 0, 0, 0, 0, 0, 3, 22, 0)
# clib.SetSensorSpecNew(0, 0, 0, 0, 0, 0, 0, 0, 3, 22, 1)
#
# # define the return type and the argument type of the imported c function
# clib.ProcessStride_FSR_grid.restype = c_int
# clib.ProcessStride_FSR_grid.argtypes = (
#     POINTER(c_ubyte), c_int, c_float, c_float, c_float, c_int, c_int, c_int, c_int)
# clib.GetStressInfo.restype = POINTER(StressInfo)
# clib.GetStressInfo.argtypes = [c_uint]
# clib.GetNewStrideInfo.restype = POINTER(NewStrideInfo)
# clib.GetNewStrideInfo.argtypes = [c_uint]
# clib.ResetStrideInfo.restype = None
# clib.ResetStrideInfo.argtypes = [c_int]
#
# ##
# ble_number = [0, 0, 18, 92, 3, 2, 15, 0, 255, 255, 0, 0, 0, 0, 0, 0, 252, 250, 250, 252, 252, 252, 252, 252, 252, 252, 252, 248, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 250, 224, 248, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 251, 252, 252, 252, 253, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 249, 252, 252, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 16, 0, 64, 6]
#
#
# error_indicator = clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
#                                               0, 1, 0, 0, 1)
# stride_pointer = clib.GetNewStrideInfo(1)
# stride_value1 = copy.copy(stride_pointer.contents)
# stress_pointer = clib.GetStressInfo(1)
# stress_value1 = copy.copy(stress_pointer.contents)
# ##
# ble_number = [0, 1, 113, 6, 0, 255, 16, 0, 255, 0, 0, 0, 0, 0, 0, 0, 252, 252, 251, 252, 252, 252, 252, 251, 252, 252,
#               252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 241, 247, 252, 252, 252, 251, 252,
#               252, 252, 252, 251, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 251, 252, 251, 251, 252, 251, 252,
#               252, 252, 252, 252, 251, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 251, 251, 252, 252, 252,
#               252, 248, 252, 252, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
#               255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 6]
#
# error_indicator = clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
#                                               0, 1, 0, 0, 1)
# stride_pointer = clib.GetNewStrideInfo(1)
# stride_value2 = copy.copy(stride_pointer.contents)
# stress_pointer = clib.GetStressInfo(1)
# stress_value2 = copy.copy(stress_pointer.contents)

## load library
if __name__ == "__main__":
    from Communication.Insole.Insole_Struct import *
    import copy
    clib = cdll.LoadLibrary("./API/libStrideAnalytics_x86_64.so")
    clib.InitUser(0, 0, 0, 0, 2, 1, 0)
    clib.SetSensorSpecNew(0, 0, 0, 0, 0, 0, 0, 0, 3, 22, 0)
    clib.SetSensorSpecNew(0, 0, 0, 0, 0, 0, 0, 0, 3, 22, 1)

    # define the return type and the argument type of the imported c function
    clib.ProcessStride_FSR_grid.restype = c_int
    clib.ProcessStride_FSR_grid.argtypes = (
        POINTER(c_ubyte), c_int, c_float, c_float, c_float, c_int, c_int, c_int, c_int)
    clib.GetStressInfo.restype = POINTER(StressInfo)
    clib.GetStressInfo.argtypes = [c_uint]
    clib.GetNewStrideInfo.restype = POINTER(NewStrideInfo)
    clib.GetNewStrideInfo.argtypes = [c_uint]
    clib.ResetStrideInfo.restype = None
    clib.ResetStrideInfo.argtypes = [c_int]


    # ble_number = left_data[192]
    ble_number = [0, 0, 17, 68, 255, 254, 15, 255, 255, 255, 0, 0, 0, 0, 0, 0, 252, 252, 252, 252, 252, 252, 252, 252,
                  252, 251, 252, 252, 252, 252, 252, 252, 252, 252, 251, 251, 252, 252, 252, 250, 231, 245, 251, 252,
                  252, 252, 252, 252, 252, 252, 252, 251, 252, 251, 252, 252, 252, 252, 251, 252, 252, 252, 252, 252,
                  252, 252, 251, 252, 252, 251, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
                  252, 252, 252, 251, 252, 252, 249, 252, 252, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    error_indicator = clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
                                                  0, 1, 0, 0, 1)
    stride_pointer_test = clib.GetNewStrideInfo(1)
    stride_value_test = copy.copy(stride_pointer_test.contents)
    stress_pointer_test = clib.GetStressInfo(1)
    stress_value_test = copy.copy(stress_pointer_test.contents)

    pass

# ##
# for i in range(20):
#     for j in range(9):
#         print(left_data[100].load_inst[i][j],end='     ')
#     print("")

# ##
# import matplotlib.pyplot as plt
# import numpy as np
# fig = plt.figure()
# ax = plt.axes()
# for i in range(len(left_data)):
#     msX = np.ctypeslib.as_array(left_data[i].load_inst)
#     im = ax.imshow(msX)
#     plt.colorbar(im)
#     plt.pause(1)
#     plt.clf()
#
# ##
# import matplotlib.pyplot as plt
# import matplotlib
# # matplotlib.use('Qt5Agg')
# print(matplotlib.get_backend())
#
# import numpy as np
#
#
# msX =  np.ctypeslib.as_array(left_data[11].load_inst)
# fig = plt.figure()
# ax = plt.axes()
# im = ax.imshow(msX)
# plt.colorbar(im)

##
my_data = pd.DataFrame({"x1":["a", "b", "c", "b"],         # Create pandas DataFrame
                        "x2":range(16, 20),
                        "x3":range(1, 5),
                        "x4":["a", "b", "c", "d"]})
print(my_data)
my_row = [11, 22, 33, 44]                                  # Create list
print(my_row)                                              # Print list
# [11, 22, 33, 44]
data_new = my_data.copy()                                  # Create copy of DataFrame
data_new.loc[1.5] = my_row                                 # Append list at the bottom
data_new = data_new.sort_index().reset_index(drop = True)  # Reorder DataFrame
print(data_new)
print(data_new)                                            # Print updated DataFrame


##
import numpy as np
import pandas as pd
rng = pd.date_range("1/1/2012", periods=100, freq="S")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts)
ts.resample("5Min").sum()

##
import numpy as np
from scipy import signal
a = np.array([1,5,11,9,13,21])
b = signal.resample(a, len(a) * 2)
print(b)

##
import pandas as pd

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

xtra = {'col1': [3,4]}

df = df.concat(pd.DataFrame(xtra))

## insert missing raw insole data
# def insertMissingRow(raw_insole_data):
#     time_stamp = raw_insole_data.iloc[:, 3]
#     for i in range(len(time_stamp) - 1):  # loop over all rows
#         num_to_insert = int((time_stamp[i + 1] - time_stamp[i]) / T - 1)  # number of rows to insert
#         for j in range(num_to_insert):
#             interplated_value = ((raw_insole_data.iloc[i + 1, 4:] - raw_insole_data.iloc[i, 4:]) / (num_to_insert + 1)
#                                  * (j + 1) + raw_insole_data.iloc[i, 4:]).to_numpy().tolist()  # linear interpolation method
#             row_to_insert = [raw_insole_data.iloc[i, 0], 0, raw_insole_data.iloc[i, 2], time_stamp[i] + T * (j + 1), *interplated_value] # construct a row list
#             raw_insole_data.loc[i + (j + 1) / (num_to_insert + 1)] = row_to_insert  # Append row at the bottom with a given index
#     inserted_left_data = raw_insole_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
#     return inserted_left_data



## upsampling
# upsampled_left_data = pd.DataFrame()
# # For each rows of the initial dataframe
# for i in range(0, len(recovered_left_data.index) - 1):  # do not add rows to the last row
#     # Append the current row to the new dataframe
#     upsampled_left_data = upsampled_left_data.append(recovered_left_data.iloc[[i], :], ignore_index=True)
#     # Add 2000/T-1 empty rows
#     for j in range(0, upsample_ratio - 1):
#         list_to_append = [*(recovered_left_data.iloc[i, 0:3].to_numpy().tolist()),
#                           recovered_left_data.iloc[i, 3] + (upsample_interval * (j + 1)),
#                           *((recovered_left_data.shape[1] - 4) * [np.NaN])]  # construct Nan lists
#         row_to_append = pd.DataFrame([list_to_append])  # convert list to dataframe
#         # upsampled_left_data = pd.concat([upsampled_left_data, row_to_append], ignore_index=True)
#         upsampled_left_data = upsampled_left_data.append(row_to_append, ignore_index=True)
# print("done")

##
# Create DataFrame with out column labels
df=pd.DataFrame([ ["Spark",20000, "30days"],
                 ["Pandas",25000, "40days"],
               ])

# Assign column names to existing DataFrame
column_names=["Courses","Fee",'Duration']
df.columns = column_names
print(df)
column_names=["1","2",'3']
df.columns = column_names


print(df)