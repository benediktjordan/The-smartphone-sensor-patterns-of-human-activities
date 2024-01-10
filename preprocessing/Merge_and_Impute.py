#region import
import pandas as pd
import pickle
import time
import numpy as np

#endregion

# this class creates merging and NaN deletion functions
class Merge_and_Impute:

    #merge timeseries or features
    # NOTE:
    ## timedelta HAS TO BE a string in format "10s", otherwise it doesnt work
    ## merging is done using the setting "nearest" which means the nearest timestamp is taken (doesnt matter if before or after)
    def merge(df_base, df_tomerge, sensor_merge, timedelta, columns_to_delete, path_intermediate_files = None, add_prefix_to_merged_columns = False):

        df_final = pd.DataFrame()

        # harmonize column names
        ## if there is a column named "2" in df_tomerge, rename it to "device_id"
        if "2" in df_base.columns:
            df_base = df_base.rename(columns={"2": "device_id"})
        if "2" in df_tomerge.columns:
            df_tomerge = df_tomerge.rename(columns={"2": "device_id"})
        if "device_id" not in df_base.columns:
            for col in df_base.columns:
                if "device_id" in col:
                    df_base = df_base.rename(columns={col: "device_id"})
        if "device_id" not in df_tomerge.columns:
            for col in df_tomerge.columns:
                if "device_id" in col:
                    df_tomerge = df_tomerge.rename(columns={col: "device_id"})

        ## if there is a column named "sensor_timestamp", rename it to "timestamp"
        if "sensor_timestamp" in df_base.columns:
            df_base.rename(columns={"sensor_timestamp": "timestamp"}, inplace=True)
        if "sensor_timestamp" in df_tomerge.columns:
            df_tomerge = df_tomerge.rename(columns={"sensor_timestamp": "timestamp"})

        user_count = 1
        total_users = len(df_base['device_id'].unique())
        # iterate through participants and ESM_timestamps
        for user in df_base['device_id'].unique():
            if path_intermediate_files != None:

                # check if for this user, there is already a df_final
                if os.path.exists(path_intermediate_files + "df_final_" + str(user_count) + ".pkl"):
                    #check if also for next user_count, there is a df_final
                    if os.path.exists(path_intermediate_files + "df_final_" + str(user_count + 1) + ".pkl"):
                        print("df_final_" + str(user_count) + ".csv already exists")
                        # if yes: increase user_count and continue
                        user_count += 1
                        continue
                    user_count += 1
                    continue
                # if not: load df_final
                if user_count != 1:
                    print("df_final_" + str(user_count-1) + " will be used now")
                    df_final = pd.read_pickle(path_intermediate_files + "df_final_" + str(user_count - 1) + ".pkl")

            print("User with ID " + str(user) + " is being processed.")
            print("This is user " + str(user_count) + " of " + str(total_users) + ".")
            time_start = time.time()

            df_base_user = df_base[df_base['device_id'] == user]
            df_tomerge_user = df_tomerge[df_tomerge['device_id'] == user]

            # make sure that timestamp columns are in datetime format with using .loc
            df_base_user['timestamp'] = pd.to_datetime(df_base_user['timestamp'])
            df_tomerge_user['timestamp'] = pd.to_datetime(df_tomerge_user['timestamp'])

            # sort dataframes by timestamp
            df_base_user = df_base_user.sort_values(by='timestamp')
            df_tomerge_user = df_tomerge_user.sort_values(by='timestamp')

            # duplicate timestamp column for test purposes and ESM_timestamp column in order to get ESM_timestamp for rows later where only data from df_tomerge is in
            df_tomerge_user['timestamp_merged'] = df_tomerge_user['timestamp'].copy()
            df_tomerge_user['ESM_timestamp_merged'] = df_tomerge_user['ESM_timestamp'].copy()

            # delete columns from df_tomerger_user that dont have to be merged
            df_tomerge_user = df_tomerge_user.drop(columns= columns_to_delete)
            #df_tomerge_user = df_tomerge_user.drop(columns=['device_id'])

            # delete columns "Unnamed: 0", "0", "1" and "2" from df_sensor_user_event: all the information of these
            # columns is already contained in the JSON data
            #df_sensor_user_event = df_sensor_user_event.drop(columns=['Unnamed: 0', '0', '1', '2'])

            #if sensor timeseries are merged: add prefix of three first letters of sensor to column names
            # in order that the "double_values_X" columns of different sensors can be differentiated
            if add_prefix_to_merged_columns == True:
                # add prefix to column names except for "timestamp" and "device_id"
                for col in df_tomerge_user.columns:
                    if col != "timestamp" and col != "device_id" and col != "timestamp_merged" and col != "ESM_timestamp" and col != "ESM_timestamp_merged":
                        df_tomerge_user = df_tomerge_user.rename(columns={col: sensor_merge[:3] + "_" + col})

            # merge dataframes
            df_merged = pd.merge_asof(df_base_user, df_tomerge_user, on= "timestamp",
                                      tolerance=pd.Timedelta(timedelta), direction='nearest')
            # Explanation: this function looks for every entry in the left timeseries if there is a
            # entry in the right timeseries which is max. "timedelta" time apart; direction = "nearest" means
            # that the closest entry is chosen
            # TODO: include functionality so that also sensors with lesser frequency can be merged (i.e.
            #  locations, open_wheather etc.)

            # delete from df_tomerge_user all rows which have a timestamp that is in df_merged["timestamp_merged"]
            df_tomerge_user = df_tomerge_user[~df_tomerge_user['timestamp_merged'].isin(df_merged["timestamp_merged"])]
            print("df_tomerge_user.shape after deleting rows that are already in df_merged: ", df_tomerge_user.shape)

            #concatenate df_merged and df_tomerge_user
            ## Note: this adds to df_merged all the records from df_tomerge_user which couldn´t be merged
            df_merged = pd.concat([df_merged, df_tomerge_user], axis=0)

            # for all rows in df_merged where the ESM_timestamp is NaN, fill it with the ESM_timestamp_merged
            ## Note: this is necessary because the ESM_timestamp is NaN for all rows which were not merged but merely concatenated
            df_merged = df_merged.reset_index(drop=True)
            for index, row in df_merged.iterrows():
                if pd.isna(row["ESM_timestamp"]):
                    df_merged.loc[index, "ESM_timestamp"] = row["ESM_timestamp_merged"]
            df_merged = df_merged.drop(columns=['ESM_timestamp_merged'])

            # concatenate df_merged to df_final
            df_final = pd.concat([df_final, df_merged], axis=0)

            # implement going in several steps since its crashing otherwise
            ## save intermediate df_final to csv
            if path_intermediate_files != None:
                #save with pickle highest protocol
                with open(path_intermediate_files + "df_final_" + str(user_count) + ".pkl", 'wb') as f:
                    pickle.dump(df_final, f, pickle.HIGHEST_PROTOCOL)

            #print("Time for User " + str(user_count) + "/" + str(total_users) + " was " + str((time.time()-time_start)/60) + " minutes")
            user_count += 1

        # if sensor timeseries are merged: add prefix to "timestamp_merged" column
        #if add_prefix_to_merged_columns == True:
        #    df_final = df_final.rename(columns={"timestamp_merged": sensor_merge[:3] + "_timestamp_merged"})
        df_final = df_final.rename(columns={"timestamp_merged": sensor_merge[:3] + "_timestamp_merged"})

        return df_final

    # delete rows with missing values
    def impute_deleteNaN(df_final):
        # delete rows with NaN values
        df_final = df_final.dropna(axis=0, how='any')
        return df_final


#region OUTDATED
#testsection
df_test = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/linear_accelerometer/activity-label_human motion - general_sensor-linear_accelerometer_timeperiod-30 s_featureselection.pkl")

# region create merged timeseries
#TODO integrate these functions into "Merge_and_Impute" class (more exact: check if Merge_and_Impute class works for timeseries merging)
## import data
sensors_included = ["linear_accelerometer", "rotation"]
timedelta = "100ms"
col_to_delete = ['Unnamed: 0', '0', '1', "ESM_timestamp"]
# check which sensor file contains the most rows; this will be the base file
# (the other files will be merged into this one)
max_rows = 0
sensors_to_merge = []
for sensor in sensors_included:
    df = pd.read_pickle(
        "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/" + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.pkl")
    if len(df) > max_rows and max_rows == 0:
        max_rows = len(df)
        base_sensor = sensor
        print("Base sensor is " + base_sensor)
        print("Number of rows: " + str(max_rows))
    elif len(df) > max_rows and max_rows != 0:
        sensors_to_merge.append(base_sensor)
        base_sensor = sensor
        print("The new base sensor is " + sensor)
        print("Number of rows: " + str(len(df)))
    else:
        #add sensor to list of sensors to merge
        sensors_to_merge.append(sensor)
del df

#sensors = ["accelerometer", "gravity",
#           "gyroscope", "magnetometer", "rotation"]

## iterate through sensors
#TODO: also merge sensors with lesser frequency (i.e. locations, open_weather etc.)

df_base = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/" + str(
    base_sensor) + "_esm_timeperiod_5 min.csv_JSONconverted.pkl")
sensors_included_string = ""
for sensor in sensors_to_merge:
    time_begin = time.time()
    sensors_included_string += "-" + sensor
    print("Current sensor is: ", sensor)
    df_sensor = pd.read_pickle(
        "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/" + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.pkl")
    #df_base = merge_unaligned_timeseries(df_base, df_tomerge=df_sensor, merge_sensor=sensor)
    df_base = Merge_and_Impute.merge(df_base, df_tomerge=df_sensor, timedelta=timedelta, columns_to_delete= col_to_delete)
    df_base.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/timeseries_merged/" + str(base_sensor) + "_with" + str(sensors_included_string) + "_esm_timeperiod_5 min_TimeseriesMerged.csv", index=False)
    time_end = time.time()
    print("Time for sensor ", sensor, " is: ", time_end - time_begin)

# endregion

##region merging function (OUTDATED; I AM USING NOW THE MERGE_AND_IMPUTE CLASS)
def merge_unaligned_timeseries(df_base, df_tomerge, merge_sensor):
    df_final = pd.DataFrame()
    user_count = 0

    #convert timestamp column to datetime format
    df_base['timestamp'] = pd.to_datetime(df_base['timestamp'])
    df_tomerge['timestamp'] = pd.to_datetime(df_tomerge['timestamp'])
    df_base["ESM_timestamp"] = pd.to_datetime(df_base["ESM_timestamp"])
    df_tomerge["ESM_timestamp"] = pd.to_datetime(df_tomerge["ESM_timestamp"])

    # iterate through devices and ESM_timestamps
    number_of_events = len(df_base['ESM_timestamp'].unique())
    counter = 1
    for event in df_base['ESM_timestamp'].unique():
        print("Event " + str(counter) + "/" + str(number_of_events))
        counter += 1

        # get data for specific user and ESM event
        df_base_event = df_base[df_base['ESM_timestamp'] == event]
        df_tomerge_event = df_tomerge[df_tomerge['ESM_timestamp'] == event]

        # sort dataframes by timestamp
        df_base_event = df_base_event.sort_values(by='timestamp')
        df_tomerge_event = df_tomerge_event.sort_values(by='timestamp')

        # duplicate timestamp column for test purposes
        df_tomerge_event['timestamp_' + str(merge_sensor)] = df_tomerge_event['timestamp'].copy()

        # delete all ESM-related columns in df_sensor_user_event (otherwise they would be duplicated)
        #df_sensor_user_event = df_sensor_user_event.drop(
        #    columns=['ESM_timestamp', "ESM_location", "ESM_location_time",
        #             "ESM_bodyposition", "ESM_bodyposition_time",
        #             "ESM_activity", "ESM_activity_time",
        #             "ESM_smartphonelocation", "ESM_smartphonelocation_time",
        #             "ESM_aligned", "ESM_aligned_time"])
        # delete columns "Unnamed: 0", "0", "1" and "2" from df_sensor_user_event: all the information of these
        # columns is already contained in the JSON data
        df_tomerge_event = df_tomerge_event.drop(columns=['Unnamed: 0', '0', '1', '2', "ESM_timestamp"])

        # merge dataframes
        df_merged = pd.merge_asof(df_base_event, df_tomerge_event, on='timestamp',
                                  tolerance=pd.Timedelta("100ms"))
        # TODO: include functionality so that also sensors with lesser frequency can be merged (i.e.
        #  locations, open_wheather etc.)

        # add merged data to 00_general dataframe
        df_final = pd.concat([df_final, df_merged], axis=0)
    return df_final
#endregion

#region temporary
df_base = pd.read_csv(
    "/Users/benediktjordan/Documents/MTS/Iteration01/Data/magnetometer_esm_timeperiod_5 min_TimeseriesMerged.csv",
    parse_dates=['timestamp', "1", "timestamp_accelerometer"], infer_datetime_format=True)

test = df_base["1"] - df_base["timestamp"]
test2 = df_base["1"] - df_base["timestamp_accelerometer"]
test.describe()
test2.describe()

df_base = df_base.drop(columns=['ESM_timestamp_y', "ESM_location_y", "ESM_location_time_y",
                                "ESM_bodyposition_y", "ESM_bodyposition_time_y",
                                "ESM_activity_y", "ESM_activity_time_y",
                                "ESM_smartphonelocation_y", "ESM_smartphonelocation_time_y",
                                "ESM_aligned_y", "ESM_aligned_time_y",
                                "Unnamed: 0_y", "0_y", "1_y", "2_y", "3_y"])
df_base = df_base.rename(columns={"Unnamed: 0_x": "Unnamed: 0", "0_x": "0", "1_x": "1", "2_x": "2", "3_x": "3",
                                  "ESM_timestamp_x": "ESM_timestamp", "ESM_location_x": "ESM_location",
                                  "ESM_location_time_x": "ESM_location_time", "ESM_bodyposition_x": "ESM_bodyposition",
                                  "ESM_bodyposition_time_x": "ESM_bodyposition_time", "ESM_activity_x": "ESM_activity",
                                  "ESM_activity_time_x": "ESM_activity_time",
                                  "ESM_smartphonelocation_x": "ESM_smartphonelocation",
                                  "ESM_smartphonelocation_time_x": "ESM_smartphonelocation_time",
                                  "ESM_aligned_x": "ESM_aligned",
                                  "ESM_aligned_time_x": "ESM_aligned_time"})

# endregion temporary
#endregion

