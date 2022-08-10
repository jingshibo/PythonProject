##
import asyncio
import multiprocessing
import threading
import os
from bleak import BleakClient, BleakScanner
import datetime
from functools import partial
from Integration.Utility.Insole_Struct import *
from Integration.Utility import Emg_Asyncio_Class
import copy

## Initialization
left_insole_address = "E4:22:A9:C9:32:C0"
right_insole_address = "CF:6B:F1:97:C6:2C"
notify_characteristic = "00002A53-0000-1000-8000-00805f9b34fb"
read_characteristic = "00002A54-0000-1000-8000-00805f9b34fb"
write_characteristic = "0000ff02-0000-1000-8000-00805f9b34fb"
# global data
initial_time = datetime.datetime.now()
left_data = []
right_data = []
data_list = []
left_timestamp = []
right_timestamp = []
# test insole
third_insole_address = "FD:87:83:5C:EE:21"
third_data = []
third_timestamp = []

## load library
clib = cdll.LoadLibrary("/home/jing/PycharmProjects/pythonProject/API/libStrideAnalytics_x86_64.so")
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
    present_time = datetime.datetime.now()
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
        run_time = datetime.datetime.now()
        print("OS time:", run_time - initial_time, "insole time:", stride_value.time, "left_data:", len(left_data))

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
        run_time = datetime.datetime.now()
        print("OS time:", run_time - initial_time, "insole time:", stride_value.time, "right_data:", len(right_data))

    elif client.address == third_insole_address:
        ble_number = list(ble_data)
        error_indicator = clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
                                                      0, 0, 0, 0, 1)
        stride_pointer = clib.GetNewStrideInfo(0)
        stride_value = copy.deepcopy(stride_pointer.contents)
        stress_pointer = clib.GetStressInfo(0)
        stress_value = copy.deepcopy(stress_pointer.contents)
        matrix_pointer = clib.GetMatrixLoadInfo(0)
        matrix_load = copy.deepcopy(matrix_pointer.contents)

        third_data.append(matrix_load)
        third_timestamp.append(stride_value.time)
        run_time = datetime.datetime.now()
        print("OS time:", run_time - initial_time, "insole time:", stride_value.time, "third_data:", len(third_data))
    datalist.append(ble_number)

    # print(present_time - initial_time)
    # print(client.address, handle, list(data))


async def connectInsole(address):
    if address == left_insole_address:
        client = BleakClient(address, adapter="hci0")  # use "busctl tree org.bluez" to obtain adapter name
    elif address == right_insole_address:
        client = BleakClient(address, adapter="hci1")
    elif address == third_insole_address:
        client = BleakClient(address, adapter="hci2")
    try:
        print("connect to", address)
        await client.connect()
        print("connect to", address)

        # set data rate
        dataRate = 25
        period = round(1000 / dataRate)
        set_dataRate = bytearray([0, 11, period])
        await client.write_gatt_char(write_characteristic, set_dataRate)
        print("connect to", address)

        # callback method
        try:
            datalist = []
            await client.start_notify(notify_characteristic, partial(callbackInsole, client, datalist))
            await asyncio.sleep(10)
            await client.stop_notify(notify_characteristic)
            await client.disconnect()
            # print(datalist)
        except Exception as e:
            print(e)

        # # loop method
        # init_time = datetime.datetime.now()
        # end_time = init_time + datetime.timedelta(seconds=10)
        # while datetime.datetime.now() < end_time:
        #     model_number = await client.read_gatt_char(read_characteristic)
        #     if address == left_insole_address:
        #         left_data.append(model_number)
        #         print(datetime.datetime.now() - init_time, "left:", len(left_data))
        #     if address == right_insole_address:
        #         right_data.append(model_number)
        #         print(datetime.datetime.now() - init_time, "right:", len(right_data))
        # await client.disconnect()

    except Exception as e:
        print(e)


def task(addresses, emg_device):
    print("process:", os.getpid(), "thread:", threading.get_ident())
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(
        asyncio.gather(*(connectInsole(address) for address in addresses), emg_device.connect_to_sq()))


def task1(addresses):
    print("process:", os.getpid(), "thread:", threading.get_ident())
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(asyncio.gather(*(connectInsole(address) for address in addresses)))


def task2(emg_device):
    print("process:", os.getpid(), "thread:", threading.get_ident())
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(emg_device.connect_to_sq())


if __name__ == "__main__":
    addresses = [left_insole_address, right_insole_address]
    emg_device = Emg_Asyncio_Class.Sessantaquattro(buffsize=200)

    # process1 = multiprocessing.Process(target=task1, args=(addresses,))
    # process2 = multiprocessing.Process(target=task2, args=(emg_device,))
    # process1.start()
    # process2.start()
    # process1.join()
    # process2.join()

    process = multiprocessing.Process(target=task, args=(addresses, emg_device))
    process.start()
    process.join()

    pass

# async def main(addresses):
#     task1 = asyncio.create_task(connectInsole(addresses[0]))
#     task2 = asyncio.create_task(connectInsole(addresses[1]))
#
#     loop = asyncio.get_running_loop()
#
#     with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
#         f1 = executor.submit(Connect_Emg.connectEmg)
#         await task1
#         await task2
#
#         # coros = [await loop.run_in_executor(executor, partial(connectInsole, addresses[0]))]  # this returns a coroutine object
#         # results = [await f for f in asyncio.as_completed(coros)]
#
#
#         # print(results)
