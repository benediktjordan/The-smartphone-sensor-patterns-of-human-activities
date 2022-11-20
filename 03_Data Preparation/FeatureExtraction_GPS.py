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

# class feature extraction of location data
class FeatureExtraction_GPS:
    # calculate distance, speed & acceleration
    def calculate_distance_speed_acceleration(self, df):
        # if there is a column "loc_timestamp": rename to "timestamp"
        if "loc_timestamp" in df.columns:
            df = df.rename(columns={"loc_timestamp": "timestamp"})

        num_events = df["ESM_timestamp"].unique().shape[0]
        event_counter = 0
        df_new = pd.DataFrame()

        # if timestamp is in unix format: proceed differently than if it is in datetime format
        if type(df["timestamp"].iloc[0]) is not str:
            for event in df["ESM_timestamp"].unique():
                # create dataset for the participant
                df_event = df.loc[df["ESM_timestamp"] == event]
                # sort by time
                df_event = df_event.sort_values(by="timestamp")
                # reset index
                df_event = df_event.reset_index(drop=True)
                # calculate distance
                df_event["distance (m)"] = 0
                for i in range(0, len(df_event)):
                    if i == 0:
                        df_event.loc[i, "distance (m)"] = np.nan  # first entry has no distance
                        continue
                    df_event.loc[i, "distance (m)"] = geopy.distance.geodesic(
                        (df_event.loc[i, "loc_double_latitude"], df_event.loc[i, "loc_double_longitude"]),
                        (df_event.loc[i - 1, "loc_double_latitude"], df_event.loc[i - 1, "loc_double_longitude"])).m

                # calculate speed
                df_event["speed (km/h)"] = 0
                for i in range(0, len(df_event)):
                    if i == 0:
                        df_event.loc[i, "speed (km/h)"] = np.nan
                        continue
                    df_event.loc[i, "speed (km/h)"] = (df_event.loc[i, "distance (m)"] / (
                                df_event.loc[i, "timestamp"] - df_event.loc[i - 1, "timestamp"])) * (
                                                                  1000 * 60 * 60 / 1000)  # *1000*60*60 to convert from milliseconds to hours; /1000 to convert km to m

                # calculate acceleration
                df_event["acceleration (m/s^2)"] = 0
                for i in range(0, len(df_event)):
                    if i == 0:
                        df_event.loc[i, "acceleration (m/s^2)"] = np.nan
                        continue
                    df_event.loc[i, "acceleration (m/s^2)"] = ((df_event.loc[i, "speed (km/h)"] - df_event.loc[
                        i - 1, "speed (km/h)"]) / (df_event.loc[i, "loc_timestamp"] - df_event.loc[
                        i - 1, "loc_timestamp"])) * (
                                                                          1000 * 1000)  # *1000 to convert km to m and another *1000 to convert ms to s

                # concatenate to df_new
                df_new = pd.concat([df_new, df_event])

                print(str(event_counter) + "/" + str(num_events) + " Events is done")
                event_counter += 1

        else:
            # convert to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            for event in df["ESM_timestamp"].unique():
                # create dataset for the participant
                df_event = df.loc[df["ESM_timestamp"] == event]
                # sort by time
                df_event = df_event.sort_values(by="timestamp")
                # reset index
                df_event = df_event.reset_index(drop=True)
                # calculate distance
                df_event["distance (m)"] = 0
                for i in range(1, len(df_event)):
                    if i == 0:
                        df_event.loc[i, "distance (m)"] = np.nan
                        continue
                    df_event.loc[i, "distance (m)"] = geopy.distance.geodesic(
                        (df_event.loc[i, "loc_double_latitude"], df_event.loc[i, "loc_double_longitude"]), (
                            df_event.loc[i - 1, "loc_double_latitude"],
                            df_event.loc[i - 1, "loc_double_longitude"])).m

                # calculate speed
                df_event["speed (km/h)"] = 0
                for i in range(0, len(df_event)):
                    if i == 0:
                        df_event.loc[i, "speed (km/h)"] = np.nan
                        continue
                    df_event.loc[i, "speed (km/h)"] = (df_event.loc[i, "distance (m)"] / (
                            df_event.loc[i, "timestamp"] - df_event.loc[i - 1, "timestamp"]).total_seconds()) * (
                                                              60 * 60 / 1000)  # *60*60 to convert from seconds to hours; /1000 to convert km to m

                # calculate acceleration
                df_event["acceleration (m/s^2)"] = 0
                for i in range(0, len(df_event)):
                    if i == 0:
                        df_event.loc[i, "acceleration (m/s^2)"] = np.nan
                        continue
                    df_event.loc[i, "acceleration (m/s^2)"] = ((df_event.loc[i, "speed (km/h)"] -
                                                                df_event.loc[i - 1, "speed (km/h)"]) / (
                                                                       df_event.loc[i, "timestamp"] -
                                                                       df_event.loc[
                                                                           i - 1, "timestamp"]).total_seconds()) * (
                                                                  1000)  # *1000 to convert km to m and another

                # concatenate to df_new
                df_new = pd.concat([df_new, df_event])

                print(str(event_counter) + "/" + str(num_events) + " Events is done")
                event_counter += 1
        # reset index
        df_new = df_new.reset_index(drop=True)
        return df_new

    def frequentlocations(df_locations_alltimes):

    def frequentlocations_classify(self):
        # classify locations
        df_locations_alltimes = self.df_locations_alltimes
        df_locations_alltimes["loc_class"] = np.nan
        df_locations_alltimes.loc[df_locations_alltimes["loc_type"] == "home", "loc_class"] = "home"
        df_locations_alltimes.loc[df_locations_alltimes["loc_type"] == "work", "loc_class"] = "work"
        df_locations_alltimes.loc[df_locations_alltimes["loc_type"] == "school", "loc_class"] = "school


    def calculate_distance_to_frequent_locations(df, df_frequent_locations_day, df_frequent_locations_night):
        # iterate through all participants
        df_new = pd.DataFrame()
        for participant in df["loc_device_id"].unique():
            # create dataset for the participant
            df_participant = df.loc[df["loc_device_id"] == participant]

            # iterate through all records of the participant
            for index, row in df_participant.iterrows():

                # compute distance of the record to the most frequent locations in night
                distances = []
                for i in range(1,
                               df_frequent_locations_night[df_frequent_locations_night["participant"] == participant][
                                   "number_clusters"].shape[0] + 1):
                    distance = geopy.distance.geodesic((row["loc_double_latitude"], row["loc_double_longitude"]), (
                    df_frequent_locations_night[df_frequent_locations_night["participant"] == participant][
                        "cluster_" + str(i) + "_latitude"].values[0],
                    df_frequent_locations_night[df_frequent_locations_night["participant"] == participant][
                        "cluster_" + str(i) + "_longitude"].values[0])).m
                    distances.append(distance)
                # get the minimum distance
                min_distance = min(distances)
                # add the minimum distance to the record
                df_participant.loc[index, "distance_to_most_frequent_locations_night (m)"] = min_distance

                # compute distance of the record to the most frequent locations in day
                distances = []
                for i in range(1, df_frequent_locations_day[df_frequent_locations_day["participant"] == participant][
                                      "number_clusters"].shape[0] + 1):
                    distance = geopy.distance.geodesic((row["loc_double_latitude"], row["loc_double_longitude"]), (
                    df_frequent_locations_day[df_frequent_locations_day["participant"] == participant][
                        "cluster_" + str(i) + "_latitude"].values[0],
                    df_frequent_locations_day[df_frequent_locations_day["participant"] == participant][
                        "cluster_" + str(i) + "_longitude"].values[0])).m
                    distances.append(distance)
                # get the minimum distance
                min_distance = min(distances)
                # add the minimum distance to the record
                df_participant.loc[index, "distance_to_most_frequent_locations_day (m)"] = min_distance
            print("Participant " + str(participant) + " done")

        # concatenate to df_new
        df_new = pd.concat([df_new, df_participant])

        # reset index
        df_new = df_new.reset_index(drop=True)
        return df_new


# calculate distance to most frequent locations (for day and nights)
df_locations_events = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/locations_esm_timeperiod_5 min.csv_JSONconverted.pkl")
df = df_locations_events.copy()
# get most frequent locations
df_frequent_locations_night = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/hours-0-6_freqquent_locations_summary.csv")
df_frequent_locations_day =pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/hours-9-18_freqquent_locations_summary.csv")



