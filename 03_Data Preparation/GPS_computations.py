#region import
import pandas as pd


#endregion

class GPS_computations:

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