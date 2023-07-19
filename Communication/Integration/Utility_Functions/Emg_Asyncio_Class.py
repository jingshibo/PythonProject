#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:27:16 2019

@author: pi
"""

import socket
import numpy as np
import datetime
import asyncio
import csv
import os

COMMAND_LENGTH_IN_BYTES = 2

class Sessantaquattro():
    def __init__(self, subject_number, record_date, trigger=0, wifi_range=0, hpfilter=1, resolution=0, mode=0, nch=3, fsamp=2, buffsize=200):

        # save file path
        data_dir = '/home/jing/PycharmProjects/pythonProject/Data'
        emg_file = f'emg/emg_subject{subject_number}_{record_date}.csv'
        self.emg_path = os.path.join(data_dir, emg_file)

        # device config
        self._buffsize = buffsize
        self._trig = trigger
        self._ext = wifi_range
        self._hpf = hpfilter
        self._hres = resolution
        self._mode = mode
        self._nch = nch
        self._fsamp = fsamp
        number_of_channels = None
        sample_frequency = None
        bytes_in_sample = None

        if nch == 0:
            if mode == 1:
                number_of_channels = 8
            else:
                number_of_channels = 12
        elif nch == 1:
            if mode == 1:
                number_of_channels = 12
            else:
                number_of_channels = 20
        elif nch == 2:
            if mode == 1:
                number_of_channels = 20
            else:
                number_of_channels = 36
        elif nch == 3:
            if mode == 1:
                number_of_channels = 36
            else:
                number_of_channels = 68
        else:
            raise Exception("wrong value for nch. Got: {}".format(nch))

        if fsamp == 0:
            if mode == 3:
                sample_frequency = 2000
            else:
                sample_frequency = 500
        elif fsamp == 1:
            if mode == 3:
                sample_frequency = 4000
            else:
                sample_frequency = 1000
        elif fsamp == 2:
            if mode == 3:
                sample_frequency = 8000
            else:
                sample_frequency = 2000
        elif fsamp == 3:
            if mode == 3:
                sample_frequency = 16000
            else:
                sample_frequency = 4000
        else:
            raise Exception("wrong value for fsamp. Got: {}".format(fsamp))

        if resolution == 1:
            bytes_in_sample = 3
        else:
            bytes_in_sample = 2

        if number_of_channels is None or sample_frequency is None or bytes_in_sample is None:
            raise Exception("Could not set number_of_channels and/or  and/or bytes_in_sample")

        self._buffer_values = np.zeros((self._buffsize, number_of_channels), dtype=np.int32)
        self._one_sample_values = np.zeros((number_of_channels,), dtype=np.int32)
        self._number_of_channels = number_of_channels
        self._sample_frequency = sample_frequency
        self._bytes_in_sample = bytes_in_sample
        print(nch)

    def create_bin_command(self, go, rec=0, getset=0):
        command = 0
        command = command + go
        command = command + rec * 2
        command = command + self._trig * 4
        command = command + self._ext * 16
        command = command + self._hpf * 64
        command = command + self._hres * 128
        command = command + self._mode * 256
        command = command + self._nch * 2048
        command = command + self._fsamp * 8192
        command = command + getset * 32768

        return int(command).to_bytes(COMMAND_LENGTH_IN_BYTES, byteorder="big")

    async def connect_to_sq(self):
        self._ip = "0.0.0.0"
        self._port = 45454

        # Create a TCP server (socket type: SOCK_STREAM)
        await self.listen_once(self.handle_server, self._ip, self._port, family=socket.AF_INET, reuse_address=True)

    async def listen_once(self, handler, *server_args, **server_kwargs):  # avoid automatically reconnecting
        first_connection_completion = asyncio.Future()

        async def wrapped_handler(*args):
            await handler(*args)
            first_connection_completion.set_result(None)

        server = await asyncio.start_server(wrapped_handler, *server_args, **server_kwargs)
        async with server:  # with statement manages the close of connection automatically
            server_task = asyncio.create_task(server.serve_forever())
            await first_connection_completion
            server_task.cancel()

    async def handle_server(self, reader, writer):
        # config device
        start_command = self.create_bin_command(go=1)
        writer.write(start_command)
        await writer.drain()

        # initialize buffer
        bdata = np.zeros((self._buffsize * self._number_of_channels), dtype=np.int32)
        emgarray = np.zeros((self._buffsize, self._number_of_channels), dtype=np.int32)
        buffer_size = self._number_of_channels * self._bytes_in_sample * self._buffsize

        # init_time = datetime.datetime.now()
        # end_time = init_time + datetime.timedelta(seconds=10)
        emg_list = []

        # while datetime.datetime.now() < end_time:
        while True:
            try:
                # read binary emg data
                os_time = datetime.datetime.now()
                raw_data_stream = await reader.readexactly(buffer_size)

                # Read emg data row by row
                dt = np.dtype(np.int16)  # each sample is 16 bits
                dt = dt.newbyteorder('big')  # data received from TCP/IP us big Endian order
                bdata = np.frombuffer(buffer=raw_data_stream, dtype=dt, count=self._buffsize * self._number_of_channels)
                emgarray = np.reshape(bdata, [self._buffsize, self._number_of_channels], order='C')

                # save emg data to .csv file
                self.saveEmgData(self.emg_path, emgarray, os_time)

                # print emg_list
                emg_list.append(emgarray)
                print("time:", datetime.datetime.now() - os_time, "length:", len(emg_list) * 200)

            except Exception as e:
                print(e)

        # close connection (because reading data is an infinite loop, the following lines will not be executed)
        stop_command = self.create_bin_command(go=0)
        writer.write(stop_command)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        print("EMG connection closed")

    def saveEmgData(self, save_path, emg_data, os_time):
        with open(save_path, 'a+') as file:
            write = csv.writer(file)
            write_data = emg_data.tolist()
            for i in range(len(write_data)):
                sample_time = (os_time + datetime.timedelta(milliseconds=0.5 * i)).isoformat(sep='_')
                write_data[i].insert(0, sample_time)
                write.writerow(write_data[i])
