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
        if (type(df["timestamp"].iloc[0]) is not str) and (type(df["timestamp"].iloc[0]) is not pd.Timestamp):
            print("Timestamp is in unix format. ADAPT THE CODE FOR ACCELERATION FOR THIS FORMAT. Not yet in correct format.")
            #break the function here
            return None
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
                        i - 1, "loc_timestamp"])) * (1000 * 1000)  # *1000 to convert km to m and another *1000 to convert ms to s

                # concatenate to df_new
                df_new = pd.concat([df_new, df_event])

                print(str(event_counter) + "/" + str(num_events) + " Events is done")
                event_counter += 1

        else:
            # convert to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # rename columns which contain  "double_longitude" in some part of the name to "loc_double_longitude"
            # and rename columns which contain  "double_latitude" in some part of the name to "loc_double_latitude"
            ## this is only done to synchronize the versions of laboratory to naturalistic data
            for col in df.columns:
                if "double_longitude" in col:
                    df = df.rename(columns={col: "loc_double_longitude"})
                if "double_latitude" in col:
                    df = df.rename(columns={col: "loc_double_latitude"})

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
                df_event["acceleration (km/h/s)"] = 0
                for i in range(0, len(df_event)):
                    if i == 0:
                        df_event.loc[i, "acceleration (km/h/s)"] = np.nan
                        continue
                    df_event.loc[i, "acceleration (km/h/s)"] = ((df_event.loc[i, "speed (km/h)"] -
                                                                df_event.loc[i - 1, "speed (km/h)"]) / ((
                                                                       df_event.loc[i, "timestamp"] -
                                                                       df_event.loc[i - 1, "timestamp"]).total_seconds()))

                # concatenate to df_new
                df_new = pd.concat([df_new, df_event])

                print(str(event_counter) + "/" + str(num_events) + " Events is done")
                event_counter += 1
        # reset index
        df_new = df_new.reset_index(drop=True)
        return df_new

    # calculate for how many days there are records
    def gps_features_count_days_with_enough_data(df_locations, min_gps_points=60):
        # Group the data by date
        grouped = df_locations.groupby(pd.Grouper(key='timestamp', freq='D'))

        # Count the number of datapoints in each group
        count = grouped.size()

        # Return the number of days where the count is greater than or equal to the minimum number of datapoints
        return (count >= min_gps_points).sum()

    # compute stays and stay features
    def compute_stays_and_stays_features(df, freq, duration_threshold, min_records_fraction):
        # Sort the dataframe by timestamp in ascending order
        df = df.sort_values(by='timestamp')

        # Add columns for the results
        df['stays'] = np.nan
        df['stay_first_record'] = 0
        df['stay_last_record'] = 0
        df['stay_duration'] = np.nan
        df['stay_first_record_trusted'] = 0
        df['stay_last_record_trusted'] = 0
        df['stay_duration_trusted'] = np.nan

        # Keep track of the current stay and its start and end indices
        current_stay = None
        start_index = None
        end_index = None

        # Keep track of the next and previous sets of records for each stay
        next_set = None
        prev_set = None

        # Iterate through chunks: each chunk contains  records for a single stay AND all records are exactly next to each other
        df['chunk_start'] = (df['cluster_label'].diff() != 0).astype(int)
        df['chunk_id'] = df['chunk_start'].cumsum()

        stays = []
        stay_counter = 0
        for chunk_id, chunk in df.groupby('chunk_id'):
            chunk_start = chunk['timestamp'].iloc[0]
            chunk_end = chunk['timestamp'].iloc[-1]
            chunk_duration = (chunk_end - chunk_start).total_seconds()
            expected_records = int(chunk_duration * freq)
            if chunk_duration > duration_threshold and len(chunk) >= expected_records * min_records_fraction:
                stay_counter += 1
                stay_counter_array = np.full(len(chunk), stay_counter)
                stays.extend(stay_counter_array)
                df.loc[chunk.index[0], 'stay_first_record'] = 1
                df.loc[chunk.index[-1], 'stay_last_record'] = 1
                df.loc[chunk.index[-1], 'stay_duration'] = chunk_duration

                # Check if the minute directly before the first record in the chunk contains at least 50% of expected records
                start_time = chunk_start - timedelta(seconds=60)
                end_time = chunk_start
                prev_minute_records = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]
                prev_minute_expected_records = int(60 * freq)
                if len(prev_minute_records) >= prev_minute_expected_records * min_records_fraction:
                    df.loc[chunk.index[0], 'stay_first_record_trusted'] = 1

                # Check if the minute directly after the last record in the chunk contains at least 50% of expected records
                start_time = chunk_end
                end_time = chunk_end + timedelta(seconds=60)
                next_minute_records = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]
                next_minute_expected_records = int(60 * freq)
                if len(next_minute_records) >= next_minute_expected_records * min_records_fraction:
                    df.loc[chunk.index[-1], 'stay_last_record_trusted'] = 1

                # stay_duration_trusted: check if there are "stay_first_record_trusted" and "stay_last_record_trusted" in the chunk; if yes, then compute duration
                if df.loc[chunk.index[0], 'stay_first_record_trusted'] == 1 and df.loc[
                    chunk.index[-1], 'stay_last_record_trusted'] == 1:
                    df.loc[chunk.index[-1], 'stay_duration_trusted'] = chunk_duration

                # create stays_trusted column: if there is a stay_duration_trusted, then set stays_trusted to stay_counter for all records in that stay
                if df.loc[chunk.index[-1], 'stay_duration_trusted'] > 0:
                    stays_trusted = np.full(len(chunk), stay_counter)
                    df.loc[chunk.index, 'stays_trusted'] = stays_trusted



            else:
                stays.extend([np.nan] * len(chunk))

        df['stays'] = stays

        return df


    ############################ FEATURES FOR PLACES ############################
    # compute number of visits per day
    def gps_features_places_visits_per_day(df_participant, participant, place_id):
        # get days which are included in df_locations
        days = df_participant["timestamp"].dt.date.unique()
        visits_per_day = []
        visits_per_day_trusted = []
        for day in days:
            # calculate how many "stay_first_record" are in df_locations for that day and that place and that participant
            visit_that_day = df_participant[
                (df_participant["timestamp"].dt.date == day) & (df_participant["stay_first_record"] == 1) & (
                            df_participant["device_id"] == participant) & (
                            df_participant["cluster_label"] == place_id)].shape[0]
            visits_per_day.append(visit_that_day)


            # calculate how many "stay_first_record_trusted" are in df_locations for that day and that place and that participant
            visit_that_day_trusted = df_locations[
                (df_locations["timestamp"].dt.date == day) & (df_locations["stay_first_record_trusted"] == 1) & (
                            df_locations["device_id"] == participant) & (
                            df_locations["cluster_label"] == place_id)].shape[0]
            visits_per_day_trusted.append(visit_that_day_trusted)

        # calculate means
        visits_per_day_mean = np.mean(visits_per_day)
        visits_per_day_trusted_mean = np.mean(visits_per_day_trusted)

        return visits_per_day_mean, visits_per_day_trusted_mean

    # compute fraction of visits on place of all places
    def gps_features_places_visits_fraction(df_participant, participant, place_id):
        #  for participant and place_id: calculate fraction of visits on place of all places
        visits_on_place = df_participant[(df_participant["device_id"] == participant) & (df_participant["cluster_label"] == place_id) & (df_participant["stay_first_record"] == 1)].shape[0]
        visits_on_all_places = df_participant[(df_participant["device_id"] == participant) & (df_participant["stay_first_record"] == 1)].shape[0]
        visits_fraction = visits_on_place / visits_on_all_places

        #  for participant and place_id: calculate fraction of visits on place of all places (trusted)
        visits_on_place_trusted = df_participant[(df_participant["device_id"] == participant) & (df_participant["cluster_label"] == place_id) & (df_participant["stay_first_record_trusted"] == 1)].shape[0]
        visits_on_all_places_trusted = df_participant[(df_participant["device_id"] == participant) & (df_participant["stay_first_record_trusted"] == 1)].shape[0]
        if visits_on_all_places_trusted == 0:
            visits_fraction_trusted = 0
        else:
            visits_fraction_trusted = visits_on_place_trusted / visits_on_all_places_trusted

        return visits_fraction, visits_fraction_trusted

    #compute time spent on place per day
    def gps_features_places_time_per_day(df_participant, participant, place):
        # get days which are included in df_locations
        days = df_participant["timestamp"].dt.date.unique()
        time_per_day = []
        time_per_day_trusted = []
        for day in days:
            # calculate how many seconds are spent on place for that day and that place and that participant
            time_on_place_that_day = df_participant[(df_participant["timestamp"].dt.date == day) & (df_participant["device_id"] == participant) & (df_participant["cluster_label"] == place)]["stay_duration"].sum()
            time_per_day.append(time_on_place_that_day)

            # calculate how many seconds are spent on place for that day and that place and that participant (trusted)
            time_on_place_that_day_trusted = df_participant[(df_participant["timestamp"].dt.date == day) & (df_participant["device_id"] == participant) & (df_participant["cluster_label"] == place)]["stay_duration_trusted"].sum()
            time_per_day_trusted.append(time_on_place_that_day_trusted)

        # calculate means
        time_per_day_mean = np.mean(time_per_day)
        time_per_day_trusted_mean = np.mean(time_per_day_trusted)

        return time_per_day_mean, time_per_day_trusted_mean

    ####### compute features for different timeslots #########

    # calculate arrive, leave, and intersection_percentage for different timeslots
    def compute_arrive_leave_intersecting_percentage(df_participant, participant, place, timeslot, timeslot_type):
        # compute arrive_percentage
        try:
            arrive_percentage = df_participant[(df_participant["cluster_label"] == place) & (
                    df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot)][
                                    "stay_first_record"].sum() / df_participant[
                                    (df_participant["cluster_label"] == place) & (
                                                df_participant["device_id"] == participant)][
                                    "stay_first_record"].sum()
        except ZeroDivisionError:
            arrive_percentage = 0

        try:
            arrive_percentage_trusted = df_participant[(df_participant["cluster_label"] == place) & (
                        df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot)][
                                            "stay_first_record_trusted"].sum() / df_participant[
                                            (df_participant["cluster_label"] == place) & (
                                                        df_participant["device_id"] == participant)][
                                            "stay_first_record_trusted"].sum()
        except ZeroDivisionError:
            arrive_percentage_trusted = 0

        # compute leave_percentage
        try:
            leave_percentage = df_participant[(df_participant["cluster_label"] == place) & (
                        df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot)][
                                   "stay_last_record"].sum() / df_participant[(df_participant["cluster_label"] == place) & (
                        df_participant["device_id"] == participant)]["stay_last_record"].sum()
        except ZeroDivisionError:
            leave_percentage = 0

        try:
            leave_percentage_trusted = df_participant[(df_participant["cluster_label"] == place) & (
                        df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot)][
                                           "stay_last_record_trusted"].sum() / df_participant[
                                           (df_participant["cluster_label"] == place) & (
                                                       df_participant["device_id"] == participant)][
                                           "stay_last_record_trusted"].sum()
        except ZeroDivisionError:
            leave_percentage_trusted = 0

        # compute intersecting_percentage
        try:
            intersecting_percentage = df_participant[(df_participant["cluster_label"] == place) & (
                        df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot)][
                                          "stays"].nunique() / df_participant[(
                        (df_participant["cluster_label"] == place) & df_participant["device_id"] == participant)][
                                          "stays"].nunique()
        except ZeroDivisionError:
            intersecting_percentage = 0
        try:
            intersecting_percentage_trusted = df_participant[(df_participant["cluster_label"] == place) & (
                        df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot)][
                                                  "stays_trusted"].nunique() / df_participant[
                                                  (df_participant["cluster_label"] == place) & (
                                                              df_participant["device_id"] == participant)][
                                                  "stays_trusted"].nunique()
        except ZeroDivisionError:
            intersecting_percentage_trusted = 0

        return arrive_percentage, arrive_percentage_trusted, leave_percentage, leave_percentage_trusted, intersecting_percentage, intersecting_percentage_trusted

    # compute time fraction, time per day, time per week for place
    def compute_time_fraction_time_per_day(df_participant, participant, place, timeslot, timeslot_type, frequency):
        # compute time fraction per day
        time_fraction = len(df_participant[(df_participant["cluster_label"] == place) & (
                    df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot) & (
                                                     df_participant["stays"] != np.nan)]) / len(df_participant[(
                                                                                                                       df_participant[
                                                                                                                           "device_id"] == participant) & (
                                                                                                                       df_participant[
                                                                                                                           "cluster_label"] == place) & (
                                                                                                                       df_participant[
                                                                                                                           "stays"] != np.nan)])
        time_fraction_trusted = len(df_participant[(df_participant["cluster_label"] == place) & (
                    df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot) & (
                                                             df_participant["stays_trusted"] != np.nan)]) / len(
            df_participant[(df_participant["device_id"] == participant) & (df_participant["cluster_label"] == place) & (
                        df_participant["stays_trusted"] != np.nan)])

        # compute time per day
        ## which days are in df_locations
        days = df_participant["timestamp"].dt.date.unique()
        time_spent_in_timeslot = []
        time_spent_in_timeslot_trusted = []
        for day in days:
            # compute time spent in timeslot
            time_spent_in_timeslot_day = frequency * len(df_participant[(df_participant["cluster_label"] == place) & (
                        df_participant["device_id"] == participant) & (df_participant[timeslot_type] == timeslot) & (
                                                                                  df_participant[
                                                                                      "timestamp"].dt.date == day) & (
                                                                                  df_participant["stays"] != np.nan)])
            time_spent_in_timeslot.append(time_spent_in_timeslot_day)

            # compute time spent in timeslot_trusted
            time_spent_in_timeslot_day_trusted = frequency * len(df_participant[
                                                                     (df_participant["cluster_label"] == place) & (
                                                                                 df_participant[
                                                                                     "device_id"] == participant) & (
                                                                                 df_participant[
                                                                                     timeslot_type] == timeslot) & (
                                                                                 df_participant[
                                                                                     "timestamp"].dt.date == day) & (
                                                                                 df_participant[
                                                                                     "stays_trusted"] != np.nan)])
            time_spent_in_timeslot_trusted.append(time_spent_in_timeslot_day_trusted)

        time_per_day = np.mean(time_spent_in_timeslot)
        time_per_day_trusted = np.mean(time_spent_in_timeslot_trusted)

        return time_fraction, time_fraction_trusted, time_per_day, time_per_day_trusted

    ######### compute features to compare different other features ###########

    # compute rank ascend and descend: they put the features into relation to each other across places
    def gps_features_places_rank_ascent_descent(df_results, feature_list):
        #reset index
        df_results = df_results.reset_index(drop=True)
        for participant in df_results["device_id"].unique():
            df_participant = df_results[df_results["device_id"] == participant]
            # iterate through all features
            for feature in feature_list:
                # sort df_results by feature
                df_results_sorted = df_participant.sort_values(by=feature, ascending=False)
                # get rank of each place: ascending and descending
                df_results_sorted[feature + "_rank_ascent"] = df_results_sorted[feature].rank(ascending=False)
                df_results_sorted[feature + "_rank_descent"] = df_results_sorted[feature].rank(ascending=True)

                # add rank ascend and descend to df_results based on index
                for index, row in df_results_sorted.iterrows():
                    # add rank ascend and descend to df_results for index
                    df_results.loc[index, feature + "_rank_ascent"] = row[feature + "_rank_ascent"]
                    df_results.loc[index, feature + "_rank_descent"] = row[feature + "_rank_descent"]

        return df_results











    #def frequentlocations(df_locations_alltimes):















    def frequentlocations_classify(self):
        # classify locations
        df_locations_alltimes = self.df_locations_alltimes
        df_locations_alltimes["loc_class"] = np.nan
        df_locations_alltimes.loc[df_locations_alltimes["loc_type"] == "home", "loc_class"] = "home"
        df_locations_alltimes.loc[df_locations_alltimes["loc_type"] == "work", "loc_class"] = "work"
        df_locations_alltimes.loc[df_locations_alltimes["loc_type"] == "school", "loc_class"] = "school"


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



