# Explanation: this class should contain the following methods:
# 1. __init__(): this method should contain the following attributes:

# 2. delete_duplicates(): this method should delete duplicates
## drop duplicates
# check for duplicates
# drop columns "Unnahmed: 0" and "0"
df = df.drop(columns=["Unnamed: 0", "0"])
df.df().sum()
# drop duplicate rows
df = df.drop_duplicates()


# 3. merge(): this method should contain the following steps:
#     a. merge the high-frequency features with the low-frequency features
#     b. return the merged dataframe
#

# 4. merge_unaligned_timeseries(): this method should contain the following attributes:
#   - self.df_base: the dataframe that should be merged with the other dataframe
#   - self.df_tomerge: the dataframe that should be merged with the base dataframe
#   - self.merge_sensor: the sensor that should be merged with the base dataframe
#   - self.sensor: the sensor that should be merged with the base dataframe

# 5. merge_sensors(): this method should contain the following attributes:
#   - self.dir_folders: the directory where the folders of different databased are stored
#   - self.list_sensors: the list of sensors that should be used

# 6. merge_participantIDs(): this method merges the ID´s of participants

#region import
import pandas as pd
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json
import pickle
import pyarrow.feather as feather
#endregion


# create class
class Merge_Transform:

    # join sensorfiles from different databases
    def join_sensor_files(dir_databases, sensor, sensor_appendix = None, participant_id = None):
        counter = 0
        for root, dirs, files in os.walk(dir_databases):  # iterate through different subfolders
            for subfolder in dirs:
                path_sensor = dir_databases + subfolder + "/" + sensor + ".csv"
                if sensor_appendix != None:
                    path_sensor = dir_databases + subfolder + "/" + sensor + sensor_appendix + ".csv"
                if os.path.exists(path_sensor):
                    print("In Subfolder " + subfolder + " the file " + sensor + " exists")
                    if counter == 0:
                        sensor_all = pd.read_csv(path_sensor)
                        # if participant_id is not None, only keep rows with participant_id
                        if participant_id != None:
                            sensor_all = sensor_all[sensor_all["2"] == participant_id]
                        counter += 1
                        print("This file was used as the base: " + str(path_sensor))
                    else:
                        #concatenate sensorfiles
                        df_sensor = pd.read_csv(path_sensor)
                        # if participant_id is not None, only keep rows with participant_id
                        if participant_id != None:
                            df_sensor = df_sensor[df_sensor["2"] == participant_id]
                        sensor_all = pd.concat([sensor_all, df_sensor])
                        print("len of sensor_all is : " + str(len(sensor_all)))
                        print("This files was added: " + str(path_sensor))
                        counter += 1
                else:
                    continue

        if counter == 0:
            print("No sensorfiles were found for sensor " + sensor)
            return None

        return sensor_all

    # transform JSON column into multiple columns
    def convert_json_to_columns(df, sensor):
        # create prefix from first three letters of sensor and, if "_" is in sensor, add the part after "_"
        if "_" in sensor:
            prefix = sensor[:3] + "_" + sensor.split("_")[1]
        else:
            prefix = sensor[:3]

        # if len(df) > 1000000: convert JSON in chunks
        if len(df) > 1000000:
            print("JSON will be converted in chunks")
            # create list of chunks
            list_chunks = []
            list_chunks_new = []
            for i in range(0, len(df), 1000000):
                list_chunks.append(df[i:i+1000000])
            # iterate through chunks
            counter = 0
            for chunk in list_chunks:
                # convert JSON in chunk
                chunk["3"] = chunk["3"].apply(lambda x: json.loads(x))
                chunk = pd.concat([chunk, chunk["3"].apply(pd.Series).add_prefix(prefix + "_")], axis=1)
                # add chunk to list
                list_chunks_new.append(chunk)
                print("chunk " + str(counter) + " of " + str(len(list_chunks)) + " converted")
                counter += 1
            # concatenate chunks
            df = pd.concat(list_chunks_new)
        else:
            df["3"] = df["3"].apply(lambda x: json.loads(x))
            # add sensor identifier to every JSON column
            df = pd.concat([df, df["3"].apply(pd.Series).add_prefix(prefix + "_")], axis=1)
        return df

    # merge IDs of participants which have several IDs
    def merge_participantIDs(df, user_database, device_id_col = None, include_cities = False):
        # if there is no column "device_id" in df, find the column that contains the device_id
        if device_id_col == None:
            if "device_id" not in df.columns:
                for col in df.columns:
                    if "device_id" in col:
                        #rename column into "device_id"
                        df = df.rename(columns={col: "device_id"})
                        device_id_col = "device_id"
                        break
            else:
                device_id_col = "device_id"

        # replace UserIDs in "device_id" column based on mapping in user_databases (ID -> new_ID)
        df[device_id_col] = df[device_id_col].replace(user_database["ID"].values, user_database["new_ID"].values)
        df[device_id_col] = df[device_id_col].astype(int)

        # if include_cities = True: add column "city" to df based on mapping in user_databases (ID -> city)
        if include_cities == True:
            df["city"] = df[device_id_col].replace(user_database["new_ID"].values, user_database["City"].values)

        return df

    # function to convert a single UNIX timestamp to local timezone
    def convert_timestamp_to_local_timezone(timestamp, time_zone):
        timestamp = timestamp / 1000  # in order to convert to milliseconds
        dt = datetime.datetime.fromtimestamp(timestamp, time_zone)
        return dt

    # create local timestamp column
    # Note: IMPORTANT: the timestamp column must be in Unix format!
    def add_local_timestamp_column(df, users):
        # if timestamp is not in columns, create it from column which has timestamp in it
        if "timestamp" not in df.columns:
            for col in df.columns:
                if "timestamp" in col:
                    #rename column into "timestamp"
                    df = df.rename(columns={col: "timestamp"})

        if "device_id" not in df.columns:
            for col in df.columns:
                if "device_id" in col:
                    #rename column into "timestamp"
                    df = df.rename(columns={col: "device_id"})

        ## create from users a dictionary with device_id as key and timezone as value
        time_zones = {}
        for index, row in users.iterrows():
            time_zones[row["new_ID"]] = pytz.timezone(row["Timezone"])

        ## add "timestamp_local" column
        df['timestamp_local'] = df.apply(
            lambda x: Merge_Transform.convert_timestamp_to_local_timezone(x['timestamp'], time_zones[x['device_id']]), axis=1)
        # change format from object to datetime for timestamp_local
        df["timestamp_local"] = df["timestamp_local"].astype(str)
        df["timestamp_local"] = df["timestamp_local"].str.rsplit("+", expand=True)[0]
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])

        return df


    # add beginning, end, and duration of smartphone session around each ESM event

    def add_smartphone_session_start_end_to_esm(df_esm, df_screen):
        df_esm["session_start"] = None
        df_esm["session_end"] = None
        # convert timestamp columns to datetime and rename to src_timestamp to "timestamp"
        df_esm["timestamp"] = pd.to_datetime(df_esm["timestamp"], unit="ms")
        df_screen["timestamp"] = pd.to_datetime(df_screen["scr_timestamp"], unit="ms")
        df_screen = df_screen.drop(columns=["scr_timestamp"])
        for index, row in df_esm.iterrows():
            # only use screen-sensor data from the same user_id
            df_screen_user = df_screen[df_screen["2"] == row["device_id"]]

            # get the record of the screen sensor which is closest before the ESM event and the "scr_screen_status" == 3 with merge_asof
            # check if if row["location_time"] is empty
            if pd.isnull(row["location_time"]):
                df_screen_before = df_screen_user[df_screen_user["timestamp"] <= row["timestamp"]]
            else:
                df_screen_before = df_screen_user[df_screen_user["timestamp"] <= row["location_time"]]
            df_screen_before = df_screen_before[df_screen_before["scr_screen_status"] == 3]
            df_screen_before = df_screen_before.sort_values(by="timestamp", ascending=False)
            df_screen_before = df_screen_before.head(1)
            # get the record of the screen sensor which is closest after the ESM event and the "scr_screen_status" == 3 with merge_asof
            if pd.isnull(row["location_time"]):
                df_screen_after = df_screen_user[df_screen_user["timestamp"] >= row["timestamp"]]
            else:
                df_screen_after = df_screen_user[df_screen_user["timestamp"] >= row["location_time"]]
            df_screen_after = df_screen_after[df_screen_after["scr_screen_status"] == 3]
            df_screen_after = df_screen_after.sort_values(by="timestamp", ascending=True)
            df_screen_after = df_screen_after.head(1)

            # add the session start and end to the ESM event (and check first if there is actually a start and end time)
            if df_screen_before.empty:
                df_esm.at[index, "session_start"] = np.nan
                df_esm.at[index, "session_end"] = df_screen_after["timestamp"].values[0]
            elif df_screen_after.empty:
                df_esm.at[index, "session_end"] = np.nan
                df_esm.at[index, "session_start"] = df_screen_before["timestamp"].values[0]
            else:
                df_esm.at[index, "session_start"] = df_screen_before["timestamp"].values[0]
                df_esm.at[index, "session_end"] = df_screen_after["timestamp"].values[0]

        # compute the duration of the smartphone session
        df_esm["smartphone_session_duration (s)"] = (df_esm["session_end"] - df_esm["session_start"]) / np.timedelta64(
            1, 's')

        return df_esm

    # delete sensor data outside of the active smartphone session
    def delete_sensor_data_outside_smartphone_session(df_sensor, df_esm_with_smartphone_sessions, frequency):
        # make convert string timestamp columns to datetime
        df_sensor["ESM_timestamp"] = pd.to_datetime(df_sensor["ESM_timestamp"])
        df_sensor["timestamp"] = pd.to_datetime(df_sensor["timestamp"])
        df_esm_with_smartphone_sessions["timestamp"] = pd.to_datetime(df_esm_with_smartphone_sessions["timestamp"])
        df_esm_with_smartphone_sessions["session_start"] = pd.to_datetime(
            df_esm_with_smartphone_sessions["session_start"])
        df_esm_with_smartphone_sessions["session_end"] = pd.to_datetime(df_esm_with_smartphone_sessions["session_end"])

        # #iterate through unique ESM_timestamps of df_sensor
        counter = 1
        for esm_timestamp in df_sensor["ESM_timestamp"].unique():
            print("Start with ESM_timestamp: " + str(esm_timestamp) + " which is: " + str(counter) + "/" + str(
                len(df_sensor["ESM_timestamp"].unique())))
            # get the corresponding smartphone session start and end
            smartphone_session_start = df_esm_with_smartphone_sessions[
                df_esm_with_smartphone_sessions["timestamp"] == pd.Timestamp(esm_timestamp)]["session_start"]
            smartphone_session_end = df_esm_with_smartphone_sessions[
                df_esm_with_smartphone_sessions["timestamp"] == pd.Timestamp(esm_timestamp)]["session_end"]

            # check if there is a smartphone session start and end
            if len(smartphone_session_start) == 0 or len(smartphone_session_end) == 0:
                print("No smartphone session start or end found for ESM_timestamp: " + str(esm_timestamp))
                counter += 1
                continue
            smartphone_session_start = smartphone_session_start.iloc[0]
            smartphone_session_end = smartphone_session_end.iloc[0]

            # analytics: number of seconds which will be deleted
            seconds_before_event = (pd.Timestamp(esm_timestamp) - smartphone_session_start).total_seconds()
            seconds_after_event = (smartphone_session_end - pd.Timestamp(esm_timestamp)).total_seconds()

            # check if either one is nan
            if np.isnan(seconds_before_event) or np.isnan(seconds_after_event):
                print("Either seconds_before_event or seconds_after_event is nan")
                counter += 1
                continue

            if seconds_before_event > 300:
                seconds_before_event = 300
            if seconds_after_event > 300:
                seconds_after_event = 300
            seconds_which_will_be_deleted = 600 - seconds_before_event - seconds_after_event

            # check if nothing has to be deleted
            if seconds_which_will_be_deleted == 0:
                print("Nothing to delete")
                counter += 1
                continue
            records_which_will_be_deleted = seconds_which_will_be_deleted * frequency
            n_records_before_deletion = len(df_sensor)

            # delete all sensor data which is not inside the smartphone session
            df_sensor_timestamp = df_sensor[df_sensor["ESM_timestamp"] == esm_timestamp]
            ## get indices of sensor data which is not inside the smartphone session
            indices_to_delete = df_sensor_timestamp[(df_sensor_timestamp["timestamp"] < smartphone_session_start) | (
                        df_sensor_timestamp["timestamp"] > smartphone_session_end)].index
            ## delete sensor data which is not inside the smartphone session
            df_sensor = df_sensor.drop(indices_to_delete)

            # analytics: percentage of records which which have been correctly deleted
            n_records_after_deletion = len(df_sensor)
            percentage_of_records_which_have_been_deleted = (
                                                                        n_records_before_deletion - n_records_after_deletion) / records_which_will_be_deleted * 100
            print(
                "Percentage of records which have been deleted: " + str(percentage_of_records_which_have_been_deleted))
            print("Length of df_sensor: " + str(len(df_sensor)))
            counter += 1
        return df_sensor

    # label sensor data with ESM data

test = df_frequentlocations_day.copy()
df = test.copy()
    # endregion


# merge all locations files & seperate JSON column
dir_databases = "/Volumes/INTENSO/In Usage new/Databases"

merge_locations = Merge_Transform(dir_databases, "locations")
df_locations = merge_locations.join_sensor_files()
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.pkl")
df_locations = merge_locations.convert_json_to_columns(df_locations)
df_locations.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.pkl")

# merge all ESM files
merge_esm = Merge(dir_databases, "esm")
df_esm = merge_esm.join_sensor_files()
df_esm.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.pkl")


# compare saving time
df_locations = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.csv")
start = time.time()
df_locations_pandas.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all2.pkl")
end = time.time()
print("saving time to_pickle: " + str(end - start))

start = time.time()
# save wiht pickle.dump
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all3.pkl", "wb") as f:
    pickle.dump(df_locations_pandas, f)
end = time.time()
print("saving time pickle_dump: " + str(end - start))


#compare loading time
start = time.time()
df_locations_pandas = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.pkl")
end = time.time()
print("loading time pd.read_pickle: " + str(end - start))

start = time.time()
# load with pickle.load
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.pkl", "rb") as f:
    df_locations_pickle = pickle.load(f)
end = time.time()
print("loading time pickle_load: " + str(end - start))




#experimenting with Feather
df_locations = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.csv")

#compare saving times
time_start = time.time()
df_locations_3.to_feather("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.feather")
time_end = time.time()
print("saving time to_feather: " + str(time_end - time_start))

start = time.time()
df_locations_2.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all_2.pkl")
end = time.time()
print("saving time to_pickle: " + str(end - start))

start = time.time()
# save wiht pickle.dump
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all3.pkl", "wb") as f:
    pickle.dump(df_locations_2, f)
end = time.time()
print("saving time pickle_dump: " + str(end - start))

time_start = time.time()
df_locations_3.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.csv")
time_end = time.time()
print("saving time to_csv: " + str(time_end - time_start))



#compar loading times
time_start = time.time()
df_locations_feather = feather.read_feather("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.feather")
#readFrame = pd.read_feather("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.feather", columns=None, use_threads=True);
time_end = time.time()
print("loading time to_feather: " + str(time_end - time_start))

start = time.time()
df_locations_2 = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.pkl")
end = time.time()
print("loading time pd.read_pickle: " + str(end - start))

start = time.time()
# load with pickle.load
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.pkl", "rb") as f:
    df_locations_3 = pickle.load(f)
end = time.time()
print("loading time pickle_load: " + str(end - start))

start = time.time()
df_locations_4 = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.csv")
end = time.time()
print("loading time pd.read_csv: " + str(end - start))


##results (20889633, 5)
saving time to_feather: 67.74425292015076
saving time to_pickle: 137.4927520751953
saving time pickle_dump: 54.08499884605408

loading time to_feather: 116.26049423217773
loading time pd.read_pickle: 9.52783989906311
loading time pickle_load: 17.951920986175537

##results for big DF (20889633, 15)
loading time pd.read_pickle: 786.913162946701
loading time pickle_load: 529.042601108551

saving time to_csv: 273.6814148426056
loading time pd.read_csv: 85.18825125694275

