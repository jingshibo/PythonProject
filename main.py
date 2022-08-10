##
import asyncio
import multiprocessing
from bleak import BleakClient, BleakScanner
from datetime import datetime
from functools import partial
from Insole.Insole_Struct import *
import EMG.Connect_Emg
import copy

## Initialization
left_insole_address = "FD:87:83:5C:EE:21"
right_insole_address = "CF:6B:F1:97:C6:2C"
notify_characteristic = "00002A53-0000-1000-8000-00805f9b34fb"
read_characteristic = "00002A00-0000-1000-8000-00805f9b34fb"
write_characteristic = "0000ff02-0000-1000-8000-00805f9b34fb"
# global data
initial_time = datetime.now()
left_data = []
right_data = []
data_list = []
left_timestamp = []
right_timestamp = []

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

## scan bluetooth device
async def scanBle():
    async with BleakScanner(adapter="hci1") as scanner:
        await asyncio.sleep(5.0)
    for d in scanner.discovered_devices:
        print(d)


## callback function
def callbackInsole(client, datalist, handle, ble_data):
    present_time = datetime.now()
    if client.address == left_insole_address:
        ble_number = list(ble_data)
        error_indicator = clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
                                                      0, 1, 0, 0, 1)
        stride_pointer = clib.GetNewStrideInfo(1)
        stride_value = copy.deepcopy(stride_pointer.contents)
        stress_pointer = clib.GetStressInfo(1)
        stress_value = copy.deepcopy(stress_pointer.contents)
        matrix_pointer = clib.GetMatrixLoadInfo(1)
        matrix_load = copy.deepcopy(matrix_pointer.contents)

        left_data.append(matrix_load)
        left_timestamp.append(stride_value.time)
        run_time = datetime.now()
        print(run_time - present_time, "left_data", len(left_data))

    elif client.address == right_insole_address:
        ble_number = list(ble_data)
        error_indicator = clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
                                                      0, 0, 0, 0, 1)
        stride_pointer = clib.GetNewStrideInfo(0)
        stride_value = copy.deepcopy(stride_pointer.contents)
        stress_pointer = clib.GetStressInfo(0)
        stress_value = copy.deepcopy(stress_pointer.contents)
        matrix_pointer = clib.GetMatrixLoadInfo(0)
        matrix_load = copy.deepcopy(matrix_pointer.contents)

        right_data.append(matrix_load)
        right_timestamp.append(stride_value.time)
        run_time = datetime.now()
        print(run_time - present_time, "right_data", len(right_data))
    datalist.append(ble_number)

    # print(present_time - initial_time)
    # print(client.address, handle, list(data))


async def connectInsole(address):
    if address == left_insole_address:
        client = BleakClient(address, adapter="hci0")  # use "busctl tree org.bluez" to obtain adapter name
    elif address == right_insole_address:
        client = BleakClient(address, adapter="hci1")
    try:
        print("connect to", address)
        await client.connect()
        print("connect to", address)

        # set data rate
        dataRate = 50
        period = round(1000 / dataRate);
        set_dataRate = bytearray([0, 11, period])
        await client.write_gatt_char(write_characteristic, set_dataRate)
        print("connect to", address)

        # callback method
        try:
            datalist = []
            await client.start_notify(notify_characteristic, partial(callbackInsole, client, datalist))
            await asyncio.sleep(5)
            await client.stop_notify(notify_characteristic)
            # print(datalist)
        except Exception as e:
            print(e)

        # while True: # loop method
        #     model_number = await client.read_gatt_char(read_characteristic)
        #     name = bytearray.decode(model_number)
        #     print('name', name)

    except Exception as e:
        print(e)


async def main(addresses):
    p = multiprocessing.Process(target=EMG.Connect_Emg.connectEmg)
    p.start()
    await asyncio.gather(*(connectInsole(address) for address in addresses))
    p.join()

if __name__ == "__main__":
    # asyncio.run(scanBle())
    asyncio.run(main([left_insole_address, right_insole_address]))
