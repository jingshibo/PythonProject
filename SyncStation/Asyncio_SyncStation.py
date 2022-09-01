from SyncStation import SyncStation_Class
import asyncio

##
window_size = 200  # 100ms
sync_station = SyncStation_Class.SyncStation(buffsize=window_size)

if __name__ == "__main__":
    asyncio.run(sync_station.async_read_sync_emg())

