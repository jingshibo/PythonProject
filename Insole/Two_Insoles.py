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

def callLeftInsole(client, handle, data):
    present_time = datetime.now()
    left_data.append(data)
    print(present_time-initial_time,"left_data", len(left_data))

def callRightInsole(client, handle, data):
    present_time = datetime.now()
    right_data.append(data)
    print(present_time - initial_time, "right_data", len(right_data))



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
        dataRate = 100;
        period = round(1000 / dataRate);
        set_dataRate = bytearray([0, 11, period])
        await client.write_gatt_char(write_characteristic, set_dataRate)
        print("connect to", address)

        # callback method
        try:
            if address == left_insole_address:
                await client.start_notify(notify_characteristic, partial(callLeftInsole, client))
            elif address == right_insole_address:
                await client.start_notify(notify_characteristic, partial(callRightInsole, client))
            await asyncio.sleep(10)
            await client.stop_notify(notify_characteristic)
        except Exception as e:
            print(e)

    except Exception as e:
        print(e)

async def main(addresses):
    await asyncio.gather(*(connectInsole(address) for address in addresses))

if __name__ == "__main__":
    # asyncio.run(scanBle())
    asyncio.run(main([left_insole_address, right_insole_address]))
