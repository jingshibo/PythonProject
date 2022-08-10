##
import asyncio
import multiprocessing
import datetime
from Integration.Utility import Insole_Asyncio_Class  # for two insoles working
from Integration.Utility import Insole_Single_Class  # for only one insole working
from Integration.Utility import Emg_Asyncio_Class # by asyncio method reading emg data
from Integration.Utility import Connect_Emg # by loop method reading emg data


def task(insole_device, addresses, emg_device): # read both insole and emg using asyncio
    insole_device.load_lib(*addresses) # initialize insole library
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(asyncio.gather(*(insole_device.connectInsole(address) for address in addresses),
                           emg_device.connect_to_sq()))
    except KeyboardInterrupt:
        print("stopping the program")
        loop.run_until_complete(insole_device.disconnectInsole())
    finally:
        print("Program stopped")

def task1(insole_device, addresses): # read only insole using asyncio
    insole_device.load_lib(*addresses) # initialize insole library
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(asyncio.gather(*(insole_device.connectInsole(address) for address in addresses)))
    except KeyboardInterrupt:
        print("stopping the program")
        loop.run_until_complete(insole_device.disconnectInsole())
    finally:
        print("Program stopped")

def task2(emg_device): # read only emg using asyncio
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(emg_device.connect_to_sq())

def task3(emg_device): # read only emg using loop method
    Connect_Emg.connectEmg()


def main():

    # basic information
    subject_number = 0
    record_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # insole addresses
    left_insole_address = "E4:22:A9:C9:32:C0"
    right_insole_address = "CF:6B:F1:97:C6:2C"
    # left_insole_address = "DC:AB:60:7C:AB:C3"
    # right_insole_address = "CD:3F:5F:AB:49:C2"
    addresses = [left_insole_address, right_insole_address]

    # initialize devices
    insole_device = Insole_Asyncio_Class.Insole(*addresses, subject_number, record_date)
    insole_device1 = Insole_Single_Class.Insole(*addresses, subject_number, record_date)
    insole_device2 = Insole_Single_Class.Insole(*addresses, subject_number, record_date)
    emg_device = Emg_Asyncio_Class.Sessantaquattro(subject_number, record_date, buffsize=200)

    # data recording
    # multiprocessing method
    # process1 = multiprocessing.Process(target=task1, args=(insole_device, addresses)) # two insoles
    process1 = multiprocessing.Process(target=task1, args=(insole_device1, [left_insole_address])) # left insole
    process2 = multiprocessing.Process(target=task1, args=(insole_device2, [right_insole_address])) # right insole
    process3 = multiprocessing.Process(target=task2, args=(emg_device,))  # emg device
    process1.start()
    process2.start()
    process3.start()
    process1.join()
    process2.join()
    process3.join()

    # asyncio method
    # process = multiprocessing.Process(target=task, args=(insole_device, addresses, emg_device))
    # process.start()
    # process.join()

if __name__ == "__main__":
    main()

