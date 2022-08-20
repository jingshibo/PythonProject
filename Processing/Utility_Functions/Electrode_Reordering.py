import pandas as pd

## adjust electrode order to match the physical EMG grid
def reorderElectrodes(emg_data):
    emg_data.insert(0, "blank_clomun", "")
    columnNames = list(range(0, 65))
    emg_data.columns = columnNames
    emg_data.loc[:, 0] = emg_data.loc[:, [1, 24, 25]].mean(axis=1)

    table1 = emg_data.loc[:, 0:12]
    table2 = emg_data.loc[:, 13:25].iloc[:, ::-1]  # reverse the electrodes between 13 and 25
    table3 = emg_data.loc[:, 26:38]
    table4 = emg_data.loc[:, 39:51].iloc[:, ::-1]  # reverse the electrodes between 39 and 51
    table5 = emg_data.loc[:, 52:64]
    emg_reordered = pd.concat([table1, table2, table3, table4, table5], axis=1)  # reoder the emg data
    emg_reordered.columns = columnNames

    return emg_reordered