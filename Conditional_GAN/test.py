## check the number of emg data
emg_timestamp = pd.to_datetime(recovered_emg_data[0], format='%Y-%m-%d_%H:%M:%S.%f')
expected_number = (emg_timestamp.iloc[-2000] - emg_timestamp.iloc[2000]).total_seconds() * 1000 * 2  # ignore the first and last 2000 data
real_number = len(emg_timestamp) - 4000
print("expected emg number:", expected_number, "real emg number:", real_number, "missing emg number:", expected_number - real_number)


## recover emg signal (both deleting duplicated data and add missing data)
# how to deal with duplicated data: delete all these data and use the first data and the last data as reference to count the number of
# missing data, then insert Nan of this number
now = datetime.datetime.now()
# delete duplicated data
start_index = 588863  # the index before the first duplicated data
end_index = 590748  # the index after the last duplicated data.
# Note: it only drop data between start_index+1 ~ end_index-1. the last index will not be dropped.
dropped_emg_data = raw_emg_data.drop(raw_emg_data.index[range(start_index+1, end_index)])
inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(dropped_emg_data, start_index, end_index)

# how to deal with lost data: calculate the number of missing data and then insert Nan of the same number
start_index = 588775  # the last index before the missing data
end_index = 588776  # the first index after the missing data
inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(inserted_emg_data, start_index, end_index)

# # interpolate emg data
reindex_emg_data = inserted_emg_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
recovered_emg_data = reindex_emg_data.interpolate(method='cubic', limit_direction='forward', axis=0)
print(datetime.datetime.now() - now)




##
# Final Refined Logic:
# If the absolute value of the difference is in the hundreds or thousands:
# Positive Difference: Classify as missing data.
# Negative Difference: Classify as duplicated data.
# If the absolute value of the difference is close to 60000:
# This likely indicates an anomaly occurring around the time of a wraparound.
# Negative Difference: Classify as missing data (since it occurred at the wraparound, the missing data makes it negative).
# Positive Difference: Classify as duplicated data (since it occurred at the wraparound, the duplicated data makes it positive).

# Function to insert NaN rows at specific indices
def insert_nan_rows(df, indices):
    nan_row = pd.Series([float('nan')]*df.shape[1], index=df.columns)
    for index in indices:
        df.loc[index] = nan_row
    return df.sort_index().reset_index(drop=True)

# Step 1: Remove Duplicated Data
import copy
emg = copy.deepcopy(raw_emg_data)

# Identify start points of duplicated segments using wrong_timestamp
duplicate_start_points = wrong_timestamp_emg.index[
    (wrong_timestamp_emg['count_difference'].abs().between(100, 9999) & (wrong_timestamp_emg['count_difference'] < 0)) | (
                wrong_timestamp_emg['count_difference'].abs() >= 59000) & (wrong_timestamp_emg['count_difference'] > 0)].tolist()
# Identify duplicated data based on the refined logic

# Drop the rows corresponding to duplicated data
df_cleaned = emg.loc[~duplicated_data_mask].reset_index(drop=True)

# Step 2: Identify Missing Data

# Re-calculate differences between adjacent timestamps in the cleaned DataFrame
df_cleaned['diff'] = df_cleaned['timestamp'].diff()

# Identify missing data based on the refined logic
missing_data_mask = ((df_cleaned['diff'].abs() <= 9999 & (df_cleaned['diff'] > 0)) |
                      (df_cleaned['diff'].abs() >= 59000) & (df_cleaned['diff'] < 0))

# Get indices where missing data is identified
missing_data_indices = df_cleaned.index[missing_data_mask].tolist()

# Insert NaN rows at the missing data indices
df_with_nans = insert_nan_rows(df_cleaned, missing_data_indices)

# Step 3: Interpolate Missing Data

# Interpolate missing data using cubic method
df_interpolated = df_with_nans.interpolate(method='cubic')




## Function to identify and remove all duplicated data based on each starting point of duplication
def identify_and_remove_all_duplicates(raw_emg_data, duplicate_start_points):
    raw_emg = copy.deepcopy(raw_emg_data)
    # List to store the indices of rows to remove
    rows_duplicated = []

    # Name of the last column, which is assumed to contain the timestamps
    timestamp_col = raw_emg.columns[-1]

    # Loop through the list of starting points
    for start_row_index in duplicate_start_points:
        start_timestamp = raw_emg.iloc[start_row_index, -1]

        # Find the closest point where duplication begins by tracing back to previous timestamps
        closest_duplication_start = raw_emg[raw_emg.index < start_row_index].loc[raw_emg[timestamp_col] == start_timestamp].index.max()

        # Initialize variables to track the expected timestamp and the end point of duplication
        expected_timestamp = start_timestamp
        end_row_index = None

        # Identify all duplicated data from the starting point until the duplication ends
        for row_index in range(start_row_index, len(raw_emg)):
            if raw_emg.iloc[row_index, -1] == expected_timestamp:
                end_row_index = row_index  # Update the end point of the duplicated segment
                expected_timestamp += 1  # Update the expected timestamp for the next iteration
            else:
                break  # Exit the loop if the data no longer match

        # Add the indices of the duplicated rows to the list
        rows_duplicated.extend(range(closest_duplication_start, end_row_index + 1))

    # Remove the duplicated rows
    df_cleaned = raw_emg.drop(rows_duplicated).reset_index(drop=True)

    return df_cleaned, rows_duplicated

df_cleaned, rows_duplicated = identify_and_remove_all_duplicates(raw_emg_data, duplicate_start_points)



## view raw emg and insole data
# plot emg and insole data
Insole_Emg_Alignment.plotAllSensorData(recovered_emg_data, recovered_left_data, recovered_right_data, 0, -1)
# check the number of emg data
emg_timestamp = pd.to_datetime(recovered_emg_data[0], format='%Y-%m-%d_%H:%M:%S.%f')
expected_number = (emg_timestamp.iloc[-2000] - emg_timestamp.iloc[2000]).total_seconds() * 1000 * 2  # ignore the first and last 2000 data
real_number = len(emg_timestamp) - 4000
print("expected emg number:", expected_number, "real emg number:", real_number, "missing emg number:", expected_number - real_number)
# check missing channels (all values are zeros)
for column in recovered_emg_data:
    if (recovered_emg_data[column] == 0).all() and column != 69 and column != 70 and column != 139 and column != 140:
        print("missing channels:", column)