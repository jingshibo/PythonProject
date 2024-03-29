from Communication.SyncStation import SyncStation_Class
import datetime


if __name__ == "__main__":
    window_size = 200  # 100ms
    sync_station = SyncStation_Class.SyncStation(buffsize=window_size)
    sync_station.start_syncstation()

    init_time = datetime.datetime.now()
    last_time = init_time
    emg_list = []
    end_time = init_time + datetime.timedelta(seconds=10)

    while datetime.datetime.now() < end_time:
        emg_data, bin_data = sync_station.loop_read_sync_emg()
        emg_list.append(emg_data)
        print("time:", datetime.datetime.now() - last_time, "length:", len(emg_list) * 200)
        last_time = datetime.datetime.now()

    sync_station.close_syncstation()



##
import matplotlib.pyplot as plt
import numpy as np
a = np.concatenate((emg_list[1], emg_list[2]), axis=0)
for element in emg_list:
    a = np.concatenate((a, element), axis=0)
plt.plot(a[:, 73])

