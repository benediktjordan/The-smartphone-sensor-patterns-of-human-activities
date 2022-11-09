# Explanation: this class should contain the following methods:
# 1. __init__(): this method should contain the following attributes:

# 2. merge_unaligned_timeseries(): this method should contain the following attributes:
#   - self.df_base: the dataframe that should be merged with the other dataframe
#   - self.df_tomerge: the dataframe that should be merged with the base dataframe
#   - self.merge_sensor: the sensor that should be merged with the base dataframe
#   - self.sensor: the sensor that should be merged with the base dataframe

# 3. merge_sensors(): this method should contain the following attributes:
#   - self.dir_folders: the directory where the folders of different databased are stored
#   - self.list_sensors: the list of sensors that should be used

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

# create class
class Merge_Transform:
    def __init__(self, dir_databases, sensor):
        self.dir_databases = dir_databases
        self.sensor = sensor

    # join sensorfiles from different databases
    def join_sensor_files(self):
        counter = 0
        for root, dirs, files in os.walk(self.dir_databases):  # iterate through different subfolders
            for subfolder in dirs:
                path_sensor = self.dir_databases + "/" + subfolder + "/" + self.sensor + ".csv"
                if os.path.exists(path_sensor):
                    if counter == 0:
                        sensor_all = pd.read_csv(path_sensor)
                        counter += 1
                        print("This file was used as the base: " + str(path_sensor))
                    else:
                        sensor_all = sensor_all.append(pd.read_csv(path_sensor))
                        print("len of sensor_all is : " + str(len(sensor_all)))
                        print("This files was added: " + str(path_sensor))
                        counter += 1
                else:
                    continue

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
start = time.time()
df_locations.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all2.pkl")
end = time.time()
print("saving time to_pickle: " + str(end - start))

start = time.time()
# save wiht pickle.dump
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all3.pkl", "wb") as f:
    pickle.dump(df_locations, f)
end = time.time()
print("saving time pickle_dump: " + str(end - start))


#compare loading time
start = time.time()
df_locations_pandas = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.pkl")
end = time.time()
print("loading time pd.read_pickle: " + str(end - start))

start = time.time()
# load with pickle.load
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all3.pkl", "rb") as f:
    df_locations_pickle = pickle.load(f)
end = time.time()
print("loading time pickle_load: " + str(end - start))

