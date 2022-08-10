##
import Emg_Asyncio_Class
import datetime
import asyncio
import socket

##
window_size = 200  # 100ms
emg_device = Emg_Asyncio_Class.Sessantaquattro(buffsize=window_size)

if __name__ == "__main__":
    asyncio.run(emg_device.connect_to_sq())

