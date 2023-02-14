#region import
import pandas as pd
import geopy.distance


#endregion

class GPS_computations:

    # delete accuracies above gps_min_accuracy and delete NaN in GPS values
    def GPS_delete_accuracy(df_locations, min_gps_accuracy):
        # define analytics dataframe
        df_analytics = pd.DataFrame(
            columns=["event", "n_rows", "n_columns", "n_events", "percentage_of_rows_deleted", "percentage_of_columns_deleted", "percentage_of_events_deleted"])

        if "ESM_timestamp" in df_locations.columns:
            df_analytics.loc[0] = ["before", df_locations.shape[0], df_locations.shape[1], df_locations["ESM_timestamp"].nunique(),  np.nan, np.nan, np.nan]
        else:
            df_analytics.loc[0] = ["before", df_locations.shape[0], df_locations.shape[1], np.nan, np.nan, np.nan, np.nan]

        # rename columns
        for column in df_locations.columns:
            # if "accuracy" is in column name, rename column to "gps_accuracy"
            if "accuracy" in column:
                df_locations = df_locations.rename(columns={column: "accuracy"})
            # if "latitude" is in column name, rename column to "latitude"
            if "latitude" in column:
                df_locations = df_locations.rename(columns={column: "latitude"})
            # if "longitude" is in column name, rename column to "longitude"
            if "longitude" in column:
                df_locations = df_locations.rename(columns={column: "longitude"})
            if "2" in column:
                # check if any of the columns is already named "ESM_timestamp"
                if "ESM_timestamp" in df_locations.columns:
                    pass
                else:
                    df_locations = df_locations.rename(columns={column: "ESM_timestamp"})

        # delete rows which have less accuracy
        df_locations = df_locations[df_locations["accuracy"] <= min_gps_accuracy]

        #include information in df_analytics
        ## if ESM_timestamp is included: include also information about number of events
        if "ESM_timestamp" in df_locations.columns:
            df_analytics.loc[1] = ["after deleting rows with accuracy < " + str(min_gps_accuracy), df_locations.shape[0],
                                   df_locations.shape[1], df_locations["ESM_timestamp"].nunique(),
                                   (df_locations.shape[0] - df_analytics.loc[0, "n_rows"]) / df_analytics.loc[0, "n_rows"],
                                   (df_locations.shape[1] - df_analytics.loc[0, "n_columns"]) / df_analytics.loc[
                                       0, "n_columns"], (df_locations["ESM_timestamp"].nunique() - df_analytics.loc[0, "n_events"]) / df_analytics.loc[0, "n_events"]]
        else:
            df_analytics.loc[1] = ["after deleting rows with accuracy < " + str(min_gps_accuracy), df_locations.shape[0],
                                   df_locations.shape[1], np.nan,
                                   (df_locations.shape[0] - df_analytics.loc[0, "n_rows"]) / df_analytics.loc[0, "n_rows"],
                                   (df_locations.shape[1] - df_analytics.loc[0, "n_columns"]) / df_analytics.loc[
                                       0, "n_columns"], np.nan]
        return df_locations, df_analytics

    # only keep first sensor event after each event
    def first_sensor_record_after_event(df_sensor):
        ## delete, for every event, all records with a timestamp earlier than the ESM timestamp
        df_sensor = df_sensor[df_sensor["timestamp"] >= df_sensor["ESM_timestamp"]]

        ## now keep only one record per event, the one with the timestamp closest to the ESM timestamp (which is the first record after the event)
        df_sensor = df_sensor.sort_values(by=['ESM_timestamp', 'timestamp'])
        df_sensor = df_sensor.drop_duplicates(subset=['ESM_timestamp'], keep='first')

        return df_sensor







    # the subsequent functions are used in the first round of classification and might be outdated

    # get locations closest to events
    #TODO could not only take the closest location at time of event, but build a cluster of locations around the event and then take the cluster centroid
    def get_locations_for_labels(df, label_column_name):
        # initialize lists
        participants = []
        ESM_events = []
        dates = []
        times = []
        labels = []
        location_latitude = []
        location_longitude = []

        # delete rows in which label is nan
        number_of_events = len(df["ESM_timestamp"].unique())
        df = df.dropna(subset=[label_column_name])
        print("Number of Events which have been dropped because of missing label: {}".format(
            number_of_events - (len(df["ESM_timestamp"].unique()))))
        print("Number of Events which have been kept: {}".format(len(df["ESM_timestamp"].unique())))

        for participant in df.loc_device_id.unique():
            for event in df[df["loc_device_id"] == participant]["ESM_timestamp"].unique():
                # get the location just at the event
                df_event = df[(df["loc_device_id"] == participant) & (df["ESM_timestamp"] == event)]
                # get record with timestamp closest to ESM_timestamp
                df_event["timestamp"] = pd.to_datetime(df_event["timestamp"])
                df_event["ESM_timestamp"] = pd.to_datetime(df_event["ESM_timestamp"])
                df_event = df_event.iloc[(df_event["timestamp"] - df_event["ESM_timestamp"]).abs().argsort()[:1]]

                # add to lists
                participants.append(participant)
                ESM_events.append(df_event["ESM_timestamp"].values[0])
                dates.append(df_event["ESM_timestamp"].dt.date.values[0])
                times.append(df_event["ESM_timestamp"].dt.time.values[0])
                labels.append(df_event[label_column_name].values[0])
                location_latitude.append(df_event["loc_double_latitude"].values[0])
                location_longitude.append(df_event["loc_double_longitude"].values[0])

        # create dataframe
        df_locations_for_labels = pd.DataFrame(
            {"participant": participants, "ESM_timestamp":ESM_events, "date": dates, "time": times, "label": labels, "latitude": location_latitude,
             "longitude": location_longitude})

        return df_locations_for_labels

    # classify locations based on a list of location classifications
    def classify_locations(df_locations, location_classifications):
        # iterate through df_locations and compute distance between location and the locations classifications in location_classifications for this participant
        for index, row in df_locations.iterrows():
            # find "device_id" column
            if "loc_device_id" in df_locations.columns:
                name_participant_column = "loc_device_id"
                latitude_column = "loc_double_latitude"
                longitude_column = "loc_double_longitude"
            elif "device_id" in df_locations.columns:
                name_participant_column = "device_id"
            elif "participant" in df_locations.columns:
                name_participant_column = "participant"
                latitude_column = "latitude"
                longitude_column = "longitude"
            else:
                print("No column with participant information found")

            # get participant
            participant = row[name_participant_column]
            # get location
            location = (row[latitude_column], row[longitude_column])
            # get location classifications for this participant
            location_classifications_participant = locations_classifications[participant]

            # compute distance between location and location classifications with geopy geodesic distance
            name_of_location_classifications = []
            distances = []
            for index_classification, row_classification in location_classifications_participant.iterrows():
                name_of_location_classifications.append(row_classification["location_name"])
                location_classification = (row_classification["latitude"], row_classification["longitude"])
                distances.append(geopy.distance.geodesic(location, location_classification).m)

            # get location classification with minimum distance
            df_locations.at[index, "location_classification"] = name_of_location_classifications[distances.index(min(distances))]
            df_locations.at[index, "distance_to_location_classification (m)"] = min(distances)
            # also add a column with the distance to the second closest location classification
            distances.remove(min(distances))
            df_locations.at[index, "location_classification_second_closest_location"] = name_of_location_classifications[distances.index(min(distances))]
            df_locations.at[index, "distance_to_second_closest_location_classification (m)"] = min(distances)

        return df_locations








# get location
df_locations = pd.read_csv()
label_column_name =
df_locations = labeling_sensor_df(df_locations, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp")
df_locations_of_locations = GPS_computations.get_locations_for_labels(df_locations, label_column_name)
# get for every participant a list of true locations


#get the locations for labels for before & after sleep for every participant
# add label column
label_column_name = "label_before and after sleep"
df_locations_events = labeling_sensor_df(df_locations_events, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp")
#df for this function must contain location data and the label column
df_locations_for_labels = get_locations_for_labels(df_locations_events, label_column_name)


#endregion