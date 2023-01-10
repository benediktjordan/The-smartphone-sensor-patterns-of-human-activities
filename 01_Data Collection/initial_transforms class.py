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
    def __init__(self, dir_databases, sensor):
        self.dir_databases = dir_databases
        self.sensor = sensor

    # join sensorfiles from different databases
    def join_sensor_files(dir_databases, sensor, sensor_appendix = None):
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
                        counter += 1
                        print("This file was used as the base: " + str(path_sensor))
                    else:
                        #concatenate sensorfiles
                        sensor_all = pd.concat([sensor_all, pd.read_csv(path_sensor)])
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
    def convert_json_to_columns(self, df):
        # create prefix from first three letters of sensor and, if "_" is in sensor, add the part after "_"
        if "_" in self.sensor:
            prefix = self.sensor[:3] + "_" + self.sensor.split("_")[1]
        else:
            prefix = self.sensor[:3]

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

