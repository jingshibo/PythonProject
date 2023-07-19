import socket
import asyncio
import numpy as np
import datetime
from Communication.SyncStation import CRC8_Calculation


class SyncStation():
    def __init__(self, buffsize=200):

        self.buffer_num = buffsize  # number of samples
        self.sample_rate = 2000  # for emg data
        self.bytes_in_sample = 2  # for emg data

        DeviceEN = [0] * 16
        EMG = [0] * 16
        Mode = [0] * 16

        # ---------- muovi 1 ------------------------------------------------------
        DeviceEN[0] = 0  # 1=Muovi enabled, 0 = Muovi disabled
        EMG[0] = 1  # 1=EMG, 0=EEG
        Mode[0] = 0  # 0=32Ch Monop, 1=16Ch Monp, 2=32Ch ImpCk, 3=32Ch Test
        # ---------- muovi 2 ------------------------------------------------------
        DeviceEN[1] = 0  # 1=Muovi enabled, 0 = Muovi disabled
        EMG[1] = 1  # 1=EMG, 0=EEG
        Mode[1] = 3  # 0=32Ch Monop, 1=16Ch Monp, 2=32Ch ImpCk, 3=32Ch Test
        # ---------- muovi 3 ------------------------------------------------------
        DeviceEN[2] = 0  # 1=Muovi enabled, 0 = Muovi disabled
        EMG[2] = 1  # 1=EMG, 0=EEG
        Mode[2] = 0  # 0=32Ch Monop, 1=16Ch Monp, 2=32Ch ImpCk, 3=32Ch Test
        # ---------- muovi 4 ------------------------------------------------------
        DeviceEN[3] = 0  # 1=Muovi enabled, 0 = Muovi disabled
        EMG[3] = 1  # 1=EMG, 0=EEG
        Mode[3] = 3  # 0=32Ch Monop, 1=16Ch Monp, 2=32Ch ImpCk, 3=32Ch Test
        # ---------- sessantaquattro 1 --------------------------------------------
        DeviceEN[4] = 1  # 1=Sessantaquattro enabled, 0 = Sessantaquattro disabled
        EMG[4] = 1  # 1=EMG, 0=EEG
        Mode[4] = 0  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Test
        # ---------- sessantaquattro 2 --------------------------------------------
        DeviceEN[5] = 0  # 1 = Sessantaquattro enabled, 0 = Sessantaquattro disabled
        EMG[5] = 1  # 1=EMG, 0=EEG
        Mode[5] = 0  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Testt
        # ---------- dueplus 1 ----------------------------------------------------
        DeviceEN[6] = 0  # 1 = due enabled, 0 = due disabled
        EMG[6] = 1  # 1=EMG, 0=EEG
        Mode[6] = 0  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Test
        # ---------- dueplus 2 ----------------------------------------------------
        DeviceEN[7] = 0  # 1 = due enabled, 0 = due disabled
        EMG[7] = 1  # 1=EMG, 0=EEG
        Mode[7] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Testt
        # ---------- dueplus 3 ----------------------------------------------------
        DeviceEN[8] = 0  # 1 = due enabled, 0 = due disabled
        EMG[8] = 1  # 1=EMG, 0=EEG
        Mode[8] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Test
        # ---------- dueplus 4 ----------------------------------------------------
        DeviceEN[9] = 0  # 1 = due enabled, 0 = due disabled
        EMG[9] = 1  # 1=EMG, 0=EEG
        Mode[9] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Testt
        # ---------- dueplus 5 ----------------------------------------------------
        DeviceEN[10] = 0  # 1 = due enabled, 0 = due disabled
        EMG[10] = 1  # 1=EMG, 0=EEG
        Mode[10] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Test
        # ---------- dueplus 6 ----------------------------------------------------
        DeviceEN[11] = 0  # 1 = due enabled, 0 = due disabled
        EMG[11] = 1  # 1=EMG, 0=EEG
        Mode[11] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Testt
        # ---------- dueplus 7 ----------------------------------------------------
        DeviceEN[12] = 0  # 1 = due enabled, 0 = due disabled
        EMG[12] = 1  # 1=EMG, 0=EEG
        Mode[12] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Test
        # ---------- dueplus 8 ----------------------------------------------------
        DeviceEN[13] = 0  # 1 = due enabled, 0 = due disabled
        EMG[13] = 1  # 1=EMG, 0=EEG
        Mode[13] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Testt
        # ---------- dueplus 9 ----------------------------------------------------
        DeviceEN[14] = 0  # 1 = due enabled, 0 = due disabled
        EMG[14] = 1  # 1=EMG, 0=EEG
        Mode[14] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Test
        # ---------- dueplus 10 ----------------------------------------------------
        DeviceEN[15] = 0  # 1 = due enabled, 0 = due disabled
        EMG[15] = 1  # 1=EMG, 0=EEG
        Mode[15] = 3  # 0=64Ch Monop, 1=32Ch Monp, 2=64Ch ImpCk, 3=64Ch Testt

        # Number of acquired channel depending on the acquisition mode
        NumChan = [38, 38, 38, 38, 70, 70, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

        # check if all settings are legal
        Error = 0
        if not all(i <= 1 for i in DeviceEN):
            Error = 1
            raise Exception("Error, set DeviceEN values equal to 0 or 1")
        elif not all(i <= 1 for i in EMG):
            Error = 1
            raise Exception("Error, set EMG values equal to 0 or 1")
        elif not all(i <= 3 for i in Mode):
            Error = 1
            raise Exception("Error, set Mode values between to 0 and 3")

        SizeComm = 0
        for i in range(16):
            SizeComm = SizeComm + DeviceEN[i]  # calculate how many devices to be connected

        self.TotNumChan = 0  # total number of channels
        self.ConfString = []  # in total 18 control commands

        self.ConfString.append(SizeComm * 2 + 1)
        ConfStrLen = 1  # total control command length

        for i in range(16):  # loop over each emg device
            if DeviceEN[i] == 1:  # if the device i is to be used
                self.ConfString.append(i * 16 + EMG[i] * 8 + Mode[i] * 2 + 1)  # set the CONTROL BYTE for device i

                self.TotNumChan = self.TotNumChan + NumChan[i]  # total number of channels increase
                if EMG[i] == 1:
                    self.sample_rate = 2000
                ConfStrLen = ConfStrLen + 1  # control string length increases for 1

        # SyncStatChan = TotNumChan + 1:TotNumChan + 6
        self.TotNumChan = self.TotNumChan + 6  # 6 additional channels from SyncStation
        self.ConfString.append(CRC8_Calculation.CRC8(self.ConfString, ConfStrLen))

        print("SyncStation initialization done")

    # Open the TCP socket
    def start_syncstation(self):
        ip = "192.168.76.1"
        port = 54320
        self.ss_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ss_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ss_socket.connect((ip, port))
        # send start commands
        for command in self.ConfString:
            self.ss_socket.send(int(command).to_bytes(1, byteorder="big"))

    # close the TCP socket
    def close_syncstation(self):
        if self.ss_socket is not None:
            ConfString = []
            ConfString.append(0)
            ConfString.append(CRC8_Calculation.CRC8(ConfString, 1))

            for command in ConfString:
                self.ss_socket.send(int(command).to_bytes(1, byteorder="big"))
            self.ss_socket.shutdown(2)
            self.ss_socket.close()
        else:
            raise Exception("Can't disconnect from because the connection is not established")

    # receive emg data using loop
    def loop_read_sync_emg(self):
        buffer_size = self.TotNumChan * self.bytes_in_sample * self.buffer_num
        raw_data_stream = self.ss_socket.recv(buffer_size, socket.MSG_WAITALL)

        # Read the data row by row
        dt = np.dtype(np.uint16)  # each sample is 16 bits
        dt = dt.newbyteorder('big')  # data received from TCP/IP use big Endian order
        bdata = np.frombuffer(buffer=raw_data_stream, dtype=dt, count=self.buffer_num * self.TotNumChan)
        emgarray = np.reshape(bdata, [self.buffer_num,  self.TotNumChan], order='C')

        emg_data = emgarray.tolist()
        return emgarray, raw_data_stream

    # receive emg data using asyncio
    async def async_read_sync_emg(self):
        _ip = "192.168.76.1"
        _port = 54320

        # Create a TCP server (socket type: SOCK_STREAM)
        reader, writer = await asyncio.open_connection(_ip, _port, family=socket.AF_INET)

        # send command to start connection
        for command in self.ConfString:
            writer.write(int(command).to_bytes(1, byteorder="big"))
            await writer.drain()

        # initialize buffer
        buffer_size = self.TotNumChan * self.bytes_in_sample * self.buffer_num

        init_time = datetime.datetime.now()
        last_time = init_time
        end_time = init_time + datetime.timedelta(seconds=10)
        emg_list = []

        while datetime.datetime.now() < end_time:
            last_time = datetime.datetime.now()
            raw_data_stream = await reader.readexactly(buffer_size)

            # Read the data row by row
            dt = np.dtype(np.uint16)  # each sample is 16 bits
            dt = dt.newbyteorder('big')  # data received from TCP/IP use big Endian order
            bdata = np.frombuffer(buffer=raw_data_stream, dtype=dt, count=self.buffer_num * self.TotNumChan)
            emgarray = np.reshape(bdata, [self.buffer_num, self.TotNumChan], order='C')

            # return emgarray, raw_data_stream
            emg_list.append(emgarray)
            print("time:", datetime.datetime.now() - init_time, "length:", len(emg_list) * 200)  # print(emgarray[0])

        # close connection
        print("Close the connection")
        ConfString = []
        ConfString.append(0)
        ConfString.append(CRC8_Calculation.CRC8(ConfString, 1))
        for command in ConfString:
            writer.write(int(command).to_bytes(1, byteorder="big"))
            await writer.drain()

        writer.close()
        await writer.wait_closed()
        print("Connection closed")


