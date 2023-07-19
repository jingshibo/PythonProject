##
import asyncio
from bleak import BleakClient, BleakScanner
from datetime import datetime
from functools import partial
from Communication.Insole.Insole_Struct import *
import copy

##
left_insole_address = "FD:87:83:5C:EE:21"
right_insole_address = "CF:6B:F1:97:C6:2C"
read_characteristic = "00002A53-0000-1000-8000-00805f9b34fb"
write_characteristic = "0000ff02-0000-1000-8000-00805f9b34fb"
# global data
initial_time = datetime.now()
last_time = initial_time
left_data = []
right_data = []

## load library
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
clib.GetMatrixLoadInfo.restype = POINTER(MatrixLoadInfo)
clib.GetMatrixLoadInfo.argtypes = [c_uint]
clib.ResetStrideInfo.restype = None
clib.ResetStrideInfo.argtypes = [c_int]

##
async def scanBle():
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)

# def callbackInsole(client, handle, data):
#     global initial_time
#     present_time = datetime.now()
#     if client.address == left_insole_address:
#         left_data.append(data)
#     elif client.address == right_insole_address:
#         right_data.append(data)
#     print("left_data", len(left_data), "right_data", len(right_data))
#     # print(present_time - initial_time)
#     # print(client.address, handle, list(data))

def callbackInsole(client, handle, data):
    global last_time
    present_time = datetime.now()
    ble_number = list(data)
    error_indicator = clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
                                                  0, 1, 0, 0, 1)
    stride_pointer = clib.GetNewStrideInfo(1)
    stride_value = copy.deepcopy(stride_pointer.contents)
    stress_pointer = clib.GetStressInfo(1)
    stress_value = copy.deepcopy(stress_pointer.contents)
    matrix_pointer = clib.GetMatrixLoadInfo(1)
    matrix_load = copy.deepcopy(matrix_pointer.contents)

    left_data.append(stride_value.time)
    run_time = datetime.now()
    print(present_time - last_time, "left_data", len(left_data))
    last_time = present_time

async def connectInsole(address):
    client = BleakClient(address)
    try:
        print("connect to", address)
        await client.connect()
        print("connect to", address)

        # read model number
        # model_number = await client.read_gatt_char(read_characteristic)
        # print("Model Number: {0}".format("".join(map(chr, model_number))))

        # set data rate
        dataRate = 50
        period = round(1000 / dataRate);
        set_dataRate = bytearray([0, 11, period])
        await client.write_gatt_char(write_characteristic, set_dataRate)
        print("connect to", address)

        # callback method
        try:
            await client.start_notify(read_characteristic, partial(callbackInsole, client))
            await asyncio.sleep(5)
            await client.stop_notify(read_characteristic)
        except Exception as e:
            print(e)

    except Exception as e:
        print(e)


async def main(address):
    await asyncio.create_task(connectInsole(address))


if __name__ == "__main__":
    # asyncio.run(scanBle())
    asyncio.run(main(right_insole_address))
