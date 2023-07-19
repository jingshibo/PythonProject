# This is for two insoles working together

import asyncio
from bleak import BleakClient, BleakScanner
import datetime
from functools import partial
from Communication.Integration.Utility_Functions.Insole_Struct import *
import copy
from ctypes import *
import numpy as np
import csv
import os

# save file path
# root_path = os.environ["HOME"]
# output_file = f"{root_path}\insole.csv"

class Insole:
    def __init__(self, left_insole_address, right_insole_address, subject_number, record_date):

        # device information
        self.left_insole_address = left_insole_address
        self.right_insole_address = right_insole_address
        self.notify_characteristic = "00002A53-0000-1000-8000-00805f9b34fb"
        self.read_characteristic = "00002A54-0000-1000-8000-00805f9b34fb"
        self.write_characteristic = "0000ff02-0000-1000-8000-00805f9b34fb"

        # RX Buffer used to track the incoming data
        self.left_data = []
        self.right_data = []
        self.left_os_time = []
        self.right_os_time = []
        self.left_insole_time = []
        self.right_insole_time = []

        # save file path
        data_dir = './Data'
        left_insole_file = f'left_insole/left_subject{subject_number}_{record_date}.csv'
        right_insole_file = f'right_insole/right_subject{subject_number}_{record_date}.csv'
        self.left_insole_path = os.path.join(data_dir, left_insole_file)
        self.right_insole_path = os.path.join(data_dir, right_insole_file)

    # load library
    def load_lib(self, *address):
        if len(address) == 1:
            raise Exception("two insoles are needed")
        elif len(address) == 2:
            self.clib = cdll.LoadLibrary("/home/jing/PycharmProjects/pythonProject/API/libStrideAnalytics_x86_64.so")
            self.clib.InitUser(0, 0, 0, 0, 2, 1, 0) # two insoles: left and right
            self.clib.SetSensorSpecNew(0, 0, 0, 0, 0, 0, 0, 0, 3, 22, 0)
            self.clib.SetSensorSpecNew(0, 0, 0, 0, 0, 0, 0, 0, 3, 22, 1)

            # define the return type and the argument type of the imported c function
            self.clib.ProcessStride_FSR_grid.restype = c_int
            self.clib.ProcessStride_FSR_grid.argtypes = (
                POINTER(c_ubyte), c_int, c_float, c_float, c_float, c_int, c_int, c_int, c_int)
            self.clib.GetStressInfo.restype = POINTER(StressInfo)
            self.clib.GetStressInfo.argtypes = [c_uint]
            self.clib.GetNewStrideInfo.restype = POINTER(NewStrideInfo)
            self.clib.GetNewStrideInfo.argtypes = [c_uint]
            self.clib.GetMatrixLoadInfo.restype = POINTER(MatrixLoadInfo)
            self.clib.GetMatrixLoadInfo.argtypes = [c_uint]
            self.clib.ResetStrideInfo.restype = None
            self.clib.ResetStrideInfo.argtypes = [c_int]

    # scan bluetooth device
    async def scanBle(self):
        async with BleakScanner(adapter="hci1") as scanner:
            await asyncio.sleep(5.0)
        for d in scanner.discovered_devices:
            print(d)


    # callback function
    def callbackInsole(self, client, handle, ble_data):
        try:
            if client.address == self.left_insole_address:
                os_time = datetime.datetime.now()
                ble_number = list(ble_data)
                error_indicator = self.clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
                                                              0, 1, 0, 0, 1)

                last_time = datetime.datetime.now()

                # read left insole data
                (insole_time, insole_force, region_force, imu_date) = self.retrievInsoleData(1)
                # save a list to csv file
                self.saveInsoleData(self.left_insole_path, os_time, insole_time, insole_force, region_force, imu_date)

                # print results
                self.left_data.append(ble_number)
                print("left:", str(os_time), insole_time, len(self.left_data), datetime.datetime.now()-last_time)

            elif client.address == self.right_insole_address:

                os_time = datetime.datetime.now()
                ble_number = list(ble_data)
                error_indicator = self.clib.ProcessStride_FSR_grid((c_ubyte * len(ble_number))(*ble_number), len(ble_number), 0, 0,
                                                              0, 0, 0, 0, 1)

                last_time = datetime.datetime.now()

                # read right insole data
                (insole_time, insole_force, region_force, imu_date) = self.retrievInsoleData(0)
                # save a list to csv file
                self.saveInsoleData(self.right_insole_path, os_time, insole_time, insole_force, region_force, imu_date)

                # print results
                self.right_data.append(insole_force)
                print("right:", str(os_time), insole_time, len(self.right_data), datetime.datetime.now()-last_time)

        except Exception as e:
            print(e)

    # connect insoles
    async def connectInsole(self, address):
        try:
            if address == self.left_insole_address: # connect to usb ble adaptor
                self.client_left = BleakClient(address, adapter="hci0")  # use "busctl tree org.bluez" to obtain adapter name
                print("connect to", address)
                await self.client_left.connect()
                await self.transmitData(self.client_left)
            elif address == self.right_insole_address: # connect to internal ble adaptor
                self.client_right = BleakClient(address, adapter="hci1")
                print("connect to", address)
                await self.client_right.connect()
                await self.transmitData(self.client_right)
        except Exception as e:
            print(e)

    # read and write data
    async def transmitData(self, client):
        # set data rate
        dataRate = 40
        period = round(1000 / dataRate)
        set_dataRate = bytearray([0, 11, period])
        await client.write_gatt_char(self.write_characteristic, set_dataRate)

        # callback method
        try:
            datalist = []
            await client.start_notify(self.notify_characteristic, partial(self.callbackInsole, client))
            # loop infinitely to read data
            # while True:
            #     await asyncio.sleep(5.0)

            # only run for a given period
            await asyncio.sleep(100)
            await client.stop_notify(self.notify_characteristic)
            await client.disconnect()
        except Exception as e:
            print(e)

    # using insole lib to retrieve insole force value from ble data
    def retrievInsoleData(self, side: int): # side means left or right insole
        stride_pointer = self.clib.GetNewStrideInfo(side)
        stride_value = copy.deepcopy(stride_pointer.contents)
        insole_time = stride_value.time
        imu_data = [stride_value.a_x, stride_value.a_y, stride_value.a_z,
                    stride_value.g_x, stride_value.g_y, stride_value.g_z]

        stress_pointer = self.clib.GetStressInfo(side)
        stress_value = copy.deepcopy(stress_pointer.contents)
        region_force = [stress_value.heel, stress_value.heel2, stress_value.mid,
                        stress_value.arch, stress_value.plantar, stress_value.front,
                        stress_value.front_2, stress_value.hallux, stress_value.toes,
                        stress_value.knee, stress_value.knee_s, stress_value.total]

        matrix_pointer = self.clib.GetMatrixLoadInfo(side)
        matrix_load = copy.deepcopy(matrix_pointer.contents)
        load_value = matrix_load.load_inst
        force_matrix = np.ctypeslib.as_array(load_value)
        insole_force = force_matrix.flatten()  # convert the matrix to a row vector

        return insole_time, insole_force, region_force, imu_data

    def saveInsoleData(self, save_path, os_time, insole_time, insole_force, region_force, imu_date):
        with open(save_path, 'a+') as file:
            write = csv.writer(file)
            write_data = insole_force.tolist()
            write_data.insert(0, os_time.isoformat(sep='_'))  # convert datetime to string
            write_data.insert(1, insole_time)
            write_data.extend(region_force)
            write_data.extend(imu_date)
            write.writerow(write_data)

    # disconnect two insoles
    async def disconnectInsole(self):
        if self.client_left:
            await self.client_left.stop_notify(self.notify_characteristic)
            await self.client_left.disconnect()
            print("Left insole disconnected")
        if self.client_right:
            await self.client_right.stop_notify(self.notify_characteristic)
            await self.client_right.disconnect()
            print("Right insole disconnected")
