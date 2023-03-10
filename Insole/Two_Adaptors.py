import asyncio
import os, sys
from bleak import BleakClient, BleakScanner
from datetime import datetime
from functools import partial

##
left_insole_address = "FD:87:83:5C:EE:21"
right_insole_address = "CF:6B:F1:97:C6:2C"
notify_characteristic = "00002A53-0000-1000-8000-00805f9b34fb"
read_characteristic = "00002A00-0000-1000-8000-00805f9b34fb"
write_characteristic = "0000ff02-0000-1000-8000-00805f9b34fb"
# global data
initial_time = datetime.now()
left_data = []
right_data = []
left_timestamp = []
right_timestamp = []

## scan bluetooth device
async def scanBle():
    async with BleakScanner(adapter="hci1") as scanner:
        await asyncio.sleep(5.0)
    for d in scanner.discovered_devices:
        print(d)

##

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

## callback function
def callbackInsole(client, datalist, handle, data):
    present_time = datetime.now()
    if client.address == left_insole_address:
        left_data.append(data)
        print(present_time-initial_time,"left_data", len(left_data))
    elif client.address == right_insole_address:
        right_data.append(data)
        right_timestamp.append(present_time-initial_time)
        print(present_time-initial_time,"right_data", len(right_data))
    datalist.append(data)

    # print(present_time - initial_time)
    # print(client.address, handle, list(data))


async def connectInsole(address):
    if address == left_insole_address:
        client = BleakClient(address,adapter="hci0") # use "busctl tree org.bluez" to obtain adapter name
    elif address == right_insole_address:
        client = BleakClient(address,adapter="hci1")
    try:
        print("connect to", address)
        await client.connect()
        print("connect to", address)

        # set data rate
        dataRate = 100
        period = round(1000 / dataRate);
        set_dataRate = bytearray([0, 11, period])
        await client.write_gatt_char(write_characteristic, set_dataRate)
        print("connect to", address)

        # callback method
        try:
            datalist = []
            await client.start_notify(notify_characteristic, partial(callbackInsole, client, datalist))
            await asyncio.sleep(10)
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
    await asyncio.gather(*(connectInsole(address) for address in addresses))


if __name__ == "__main__":
    # asyncio.run(scanBle())
    asyncio.run(main([left_insole_address, right_insole_address]))
