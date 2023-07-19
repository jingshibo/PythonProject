from Communication.Insole.Insole_Struct import *


## load library
def bleToStress(ble_number,process_queue, clib):


    error_indicator = clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
                                                  0, 1, 0, 0, 1)
    stride_pointer = clib.GetNewStrideInfo(1)
    stride_value = stride_pointer.contents
    stress_pointer = clib.GetStressInfo(1)
    stress_value = stress_pointer.contents

    process_queue.put(stress_pointer)