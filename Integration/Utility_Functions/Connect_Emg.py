##
from Integration.Utility_Functions import Emg_Sq_Class
import datetime

def connectEmg():
    window_size = 200  # 100ms

    emg_device = Emg_Sq_Class.Sessantaquattro(buffsize=window_size)

    emg_device.connect_to_sq()

    init_time = datetime.datetime.now()
    last_time = init_time
    emg_list = []
    end_time = init_time + datetime.timedelta(seconds=10)

    while datetime.datetime.now() < end_time:
        emg_data, bin_data = emg_device.read_emg()
        emg_list.append(emg_data)
        print("time:", datetime.datetime.now()-last_time, "length:", len(emg_list)*200)
        last_time = datetime.datetime.now()

    emg_device.disconnect_from_sq()

if __name__ == "__main__":
    connectEmg()

