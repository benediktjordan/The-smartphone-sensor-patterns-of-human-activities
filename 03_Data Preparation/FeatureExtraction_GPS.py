#region import
import pandas as pd
import json
from datetime import datetime
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# computing distance
import geopy.distance

#endregion

# load data
## complete dataset
df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.csv", nrows=10000)

##data around events
df_locations_events = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/locations_esm_timeperiod_5 min.csv_JSONconverted.pkl")


# calculate distance, speed and acceleration
def calculate_distance_speed_acceleration(df):
    # if there is a column "loc_timestamp": rename to "timestamp"
    if "loc_timestamp" in df.columns:
        df = df.rename(columns={"loc_timestamp": "timestamp"})

    # if timestamp is in unix format: proceed differently than if it is in datetime format
    if type(df["timestamp"].iloc[0]) is not str:

        df_new = pd.DataFrame()
        for participant in df["loc_device_id"].unique():
            # create dataset for the participant
            df_participant = df.loc[df["loc_device_id"] == participant]
            # sort by time
            df_participant = df_participant.sort_values(by="timestamp")
            # reset index
            df_participant = df_participant.reset_index(drop=True)
            # calculate distance
            df_participant["distance (m)"] = 0
            for i in range(1, len(df_participant)):
                if i == 0:
                    df_participant.loc[i, "distance (m)"] = np.nan
                    continue
                df_participant.loc[i, "distance (m)"] = geopy.distance.geodesic((df_participant.loc[i, "loc_double_latitude"], df_participant.loc[i, "loc_double_longitude"]), (df_participant.loc[i-1, "loc_double_latitude"], df_participant.loc[i-1, "loc_double_longitude"])).m

            # calculate speed
            df_participant["speed (km/h)"] = 0
            for i in range(1, len(df_participant)):
                if i == 0:
                    df_participant.loc[i, "speed (km/h)"] = np.nan
                    continue
                df_participant.loc[i, "speed (km/h)"] = (df_participant.loc[i, "distance (m)"] / (df_participant.loc[i, "timestamp"] - df_participant.loc[i-1, "timestamp"])) * (1000*60*60/1000) # *1000*60*60 to convert from milliseconds to hours; /1000 to convert km to m

            # calculate acceleration
            df_participant["acceleration (m/s^2)"] = 0
            for i in range(1, len(df_participant)):
                if i == 0:
                    df_participant.loc[i, "acceleration (m/s^2)"] = np.nan
                    continue
                df_participant.loc[i, "acceleration (m/s^2)"] = ((df_participant.loc[i, "speed (km/h)"] - df_participant.loc[i-1, "speed (km/h)"]) / (df_participant.loc[i, "loc_timestamp"] - df_participant.loc[i-1, "loc_timestamp"])) * (1000*1000) # *1000 to convert km to m and another *1000 to convert ms to s

            # concatenate to df_new
            df_new = pd.concat([df_new, df_participant])

            print("Participant " + str(participant) + " done")

    else:
        # convert to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df_new = pd.DataFrame()
        for participant in df["loc_device_id"].unique():
            # create dataset for the participant
            df_participant = df.loc[df["loc_device_id"] == participant]
            # sort by time
            df_participant = df_participant.sort_values(by="timestamp")
            # reset index
            df_participant = df_participant.reset_index(drop=True)
            # calculate distance
            df_participant["distance (m)"] = 0
            for i in range(1, len(df_participant)):
                if i == 0:
                    df_participant.loc[i, "distance (m)"] = np.nan
                    continue
                df_participant.loc[i, "distance (m)"] = geopy.distance.geodesic(
                    (df_participant.loc[i, "loc_double_latitude"], df_participant.loc[i, "loc_double_longitude"]), (
                    df_participant.loc[i - 1, "loc_double_latitude"],
                    df_participant.loc[i - 1, "loc_double_longitude"])).m

            # calculate speed
            df_participant["speed (km/h)"] = 0
            for i in range(1, len(df_participant)):
                if i == 0:
                    df_participant.loc[i, "speed (km/h)"] = np.nan
                    continue
                df_participant.loc[i, "speed (km/h)"] = (df_participant.loc[i, "distance (m)"] / (
                            df_participant.loc[i, "timestamp"] - df_participant.loc[i - 1, "timestamp"]).seconds) * (
                                                                    60 * 60 / 1000) # *60*60 to convert from seconds to hours; /1000 to convert km to m

            # calculate acceleration
            df_participant["acceleration (m/s^2)"] = 0
            for i in range(1, len(df_participant)):
                if i == 0:
                    df_participant.loc[i, "acceleration (m/s^2)"] = np.nan
                    continue
                df_participant.loc[i, "acceleration (m/s^2)"] = ((df_participant.loc[i, "speed (km/h)"] -
                                                                  df_participant.loc[i - 1, "speed (km/h)"]) / (
                                                                             df_participant.loc[i, "timestamp"] -
                                                                             df_participant.loc[
                                                                                 i - 1, "timestamp"]).seconds) * (
                                                                            1000 )  # *1000 to convert km to m and another

            # concatenate to df_new
            df_new = pd.concat([df_new, df_participant])

            print("Participant " + str(participant) + " done")
    #reset index
    df_new = df_new.reset_index(drop=True)
    return df_new

time_start = time.time()
df_features = calculate_distance_speed_acceleration(df_locations_events)
df_features.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/locations-aroundevents_features-distance-speed-acceleration.csv")
time_end = time.time()
print("Time needed in minutes: " + str((time_end - time_start)/60))



# calculate distance to most frequent locations (for day and nights)
df = df_locations_events.copy()
# get most frequent locations
df_frequent_locations_night = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/hours-0-6_freqquent_locations_summary.csv")
df_frequent_locations_day =pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/hours-9-18_freqquent_locations_summary.csv")

def calculate_distance_to_frequent_locations(df, df_frequent_locations_day, df_frequent_locations_night):
    #iterate through all participants
    df_new = pd.DataFrame()
    for participant in df["loc_device_id"].unique():
        # create dataset for the participant
        df_participant = df.loc[df["loc_device_id"] == participant]

        # iterate through all records of the participant
        for index, row in df_participant.iterrows():

            # compute distance of the record to the most frequent locations in night
            distances = []
            for i in range(1, df_frequent_locations_night[df_frequent_locations_night["participant"] == participant]["number_clusters"].shape[0]+1):
                distance = geopy.distance.geodesic((row["loc_double_latitude"], row["loc_double_longitude"]), (df_frequent_locations_night[df_frequent_locations_night["participant"] == participant]["cluster_" + str(i) +"_latitude"].values[0], df_frequent_locations_night[df_frequent_locations_night["participant"] == participant]["cluster_" + str(i) + "_longitude"].values[0])).m
                distances.append(distance)
            #get the minimum distance
            min_distance = min(distances)
            #add the minimum distance to the record
            df_participant.loc[index, "distance_to_most_frequent_locations_night (m)"] = min_distance

            # compute distance of the record to the most frequent locations in day
            distances = []
            for i in range(1, df_frequent_locations_day[df_frequent_locations_day["participant"] == participant]["number_clusters"].shape[0]+1):
                distance = geopy.distance.geodesic((row["loc_double_latitude"], row["loc_double_longitude"]), (df_frequent_locations_day[df_frequent_locations_day["participant"] == participant]["cluster_" + str(i) +"_latitude"].values[0], df_frequent_locations_day[df_frequent_locations_day["participant"] == participant]["cluster_" + str(i) + "_longitude"].values[0])).m
                distances.append(distance)
            #get the minimum distance
            min_distance = min(distances)
            #add the minimum distance to the record
            df_participant.loc[index, "distance_to_most_frequent_locations_day (m)"] = min_distance
        print("Participant " + str(participant) + " done")

    # concatenate to df_new
    df_new = pd.concat([df_new, df_participant])

    #reset index
    df_new = df_new.reset_index(drop=True)
    return df_new


