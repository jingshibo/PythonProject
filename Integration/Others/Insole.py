## import modules
import asyncio
import datetime
from Integration.Utility_Functions import Insole_Asyncio_Class


## main body
async def main():
    # basic information
    subject = 'Number1'
    record_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # insole addresses (change it using your insoles' address)
    left_insole_address = "C0:9B:A7:0E:14:17"
    right_insole_address = "CE:3A:51:F2:4B:76"
    addresses = [left_insole_address, right_insole_address]

    # initialize devices
    insole_device = Insole_Asyncio_Class.Insole(*addresses, subject, record_date)
    insole_device.load_lib(*addresses)  # initialize insole library
    await asyncio.gather(*(insole_device.connectInsole(address) for address in addresses))


## run the program
if __name__ == "__main__":
    asyncio.run(main())

