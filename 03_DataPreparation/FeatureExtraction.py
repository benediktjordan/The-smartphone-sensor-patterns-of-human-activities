import pandas as pd
import tsfresh
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh import extract_relevant_features

from tsfresh.utilities.dataframe_functions import impute
import os
import numpy as np
import time
from tqdm import tqdm
import pickle
import datetime

#NOTE: this script is a copy of the one in "pythonProject", since the deployment
# of the tsfresh code needs another runtime environment
# therefore this script needs to be updated regularly (as well as the part of "main" referring to it)
class computeFeatures:
    ## create feature extraction function (using tsfresh)
    def feature_extraction(df_sensor, sensor_column_names, feature_segment, time_column_name="timestamp",
                           ESM_event_column_name="ESM_timestamp"):

        df_final = pd.DataFrame()
        counter = 1

        # create list of columns which are used for feature extraction
        feature_extraction_columns = sensor_column_names.copy()
        feature_extraction_columns.append(time_column_name)
        feature_extraction_columns.append(ESM_event_column_name)
        if "2" in df_sensor.columns: # the device ID in the iteration 01 data
            #rename column "2" to "device_id"
            df_sensor = df_sensor.rename(columns={"2": "device_id"})
        if "1" in df_sensor.columns: # timestamp of the sensor collection
            feature_extraction_columns.append("1")
        feature_extraction_columns.append("device_id")

        # get only sensor columns and also label columns
        df_sensor = df_sensor[feature_extraction_columns]

        # count number of rows with nan value
        nan_rows = df_sensor.isnull().sum(axis=1)
        nan_rows = nan_rows[nan_rows > 0]
        print(f"Number of rows with nan values: {len(nan_rows)}")
        print(f"Percentage of rows with nan values: {len(nan_rows)/df_sensor.shape[0]}")

        # remove rows with NaN values & reset index
        df_sensor = df_sensor.dropna()
        df_sensor = df_sensor.reset_index(drop=True)
        # check: if there are less then segment_size records left, skip this iteration
        if (df_sensor.shape[0] < 10):
            print("Less than 10 records left, skip this iteration")
            return pd.DataFrame()

        # create ID column: used by feature extraction to identify different segments
        # TODO change this ID selection so that it doesn´t rely on counts but on the timestamp (for the
        # cases that the sampling frequency is not all the time 100% correct)
        #iterate through the unique ESM_timestamps
        df_sensor["ID"] = ""
        counter_id = 1
        #convert the time_column_name to datetime objects
        df_sensor[time_column_name] = pd.to_datetime(df_sensor[time_column_name])
        df_sensor[ESM_event_column_name] = pd.to_datetime(df_sensor[ESM_event_column_name])

        #iterate through users and ESM_timestamps: necessary to iterate also through users, since we have the case in
        # iteration 02 data that two users have the same ESM_timestamp :)
        for user in df_sensor["device_id"].unique():
            for ESM_timestamp in df_sensor[ESM_event_column_name].unique():
                # iterate through the part of df_sensor with this ESM_timestamp and get earliest "timestamp" value. Then
                # create ID for every segment of size segment_size
                # get the smallest timestamp for the user and ESM_timestamp
                start_timestamp = df_sensor[(df_sensor["device_id"] == user) & (df_sensor[ESM_event_column_name] == ESM_timestamp)][time_column_name].min()
                end_timestamp = start_timestamp + datetime.timedelta(seconds=feature_segment)

                while end_timestamp < df_sensor[(df_sensor["device_id"] == user) & (df_sensor[ESM_event_column_name] == ESM_timestamp)][time_column_name].max():
                    # create ID colum; the ESM_timestamp has to be equal to ESM_timestamp
                    df_sensor.loc[(df_sensor[ESM_event_column_name] == ESM_timestamp) & (
                                df_sensor["device_id"] == user) & (
                                df_sensor[time_column_name] >= start_timestamp) & (
                                df_sensor[time_column_name] < end_timestamp), "ID"] = counter_id

                    # analytics: count how many rows this ID has
                    #print("ID: ", counter_id, "number of rows: ", len(df_sensor[df_sensor["ID"] == counter_id]))
                    #print("start_timestamp: ", start_timestamp, "end_timestamp: ", end_timestamp)

                    # increase counter
                    counter_id += 1
                    # increase start_timestamp and end_timestamp
                    start_timestamp = end_timestamp
                    end_timestamp = start_timestamp + datetime.timedelta(seconds=feature_segment)

        #outdated version: this version of creating the ID didn´t take the timestamps into account but did it by using only the frequency
        #df_sensor['ID'] = df_sensor.index // segment_size

        # make sure that in one segment, there is only one ESM event (check with ESM_timestamp);
        # if there are more than one ESM events in one segment, check which ESM has the most entries in the segment and delete the other
        #get_indices = df_sensor.groupby(['ID', 'ESM_timestamp'])["ESM_timestamp"].count().reset_index(name="count")
        # delete all rows with segment_size value
        #get_indices = get_indices[get_indices["count"] != segment_size]

        # for every ID, get the ESM_timestamp which has the most entries in the segment
        # and delete all other ESM_timestamps for that ID
        #for ID in get_indices["ID"].unique():
            # check if there is only one ESMTimestamp for that ID (is the case if lenght of the last
            # segment is not segment_size).
        #    if len(get_indices[get_indices["ID"] == ID]["ESM_timestamp"].unique()) == 1:
                # If so, check if the count is >1; if not, delete the row
        #        if int(get_indices[get_indices["ID"] == ID]["count"]) == 1:
                    # drop row with this ID from df_sensor
        #            df_sensor = df_sensor.drop(df_sensor[df_sensor["ID"] == ID].index)
        #        else:
        #            continue
        #        continue
                # print("ID with only one ESM_timestamp: ", ID)
            # get ESM_timestamp with least entries in segment
        #    ESM_timestamp = get_indices[get_indices["ID"] == ID]["count"].sort_values(ascending=False).index[1]
            # get indices of all rows with ID and ESM_timestamp
        #    indices = df_sensor[(df_sensor["ID"] == ID) & (df_sensor["ESM_timestamp"] == ESM_timestamp)].index
            # delete all rows with ID and ESM_timestamp
        #    df_sensor = df_sensor.drop(indices)

        # create dataset which contains only device_id, timestamp and  ESM_timestamp (needed for later)
        df_intermediate = df_sensor[[ESM_event_column_name, "device_id", "ID", time_column_name]].copy()

        # sort by timestamp
        df_sensor = df_sensor.sort_values(by=time_column_name)

        # drop unnecessary columns (i.e. columns which are not used for feature extraction)
        df_sensor = df_sensor.drop(columns=[ESM_event_column_name, "device_id"])
        if "1" in df_sensor.columns:
            df_sensor = df_sensor.drop(columns=["1"])
        if "2" in df_sensor.columns:
            df_sensor = df_sensor.drop(columns=["2"])

        # drop all rows with ID = ""
        # print: number of rows with ID = "" and percentage of rows with ID = ""
        print("number of rows with ID = \"\": ", len(df_sensor[df_sensor["ID"] == ""]))
        print("percentage of rows with ID = \"\": ", len(df_sensor[df_sensor["ID"] == ""]) / len(df_sensor))

        df_sensor = df_sensor[df_sensor["ID"] != ""]

        # extract feature
        extracted_features = extract_features(df_sensor, column_id='ID', column_sort=time_column_name)

        # impute missing values
        # impute(extracted_features)

        # apply feature selection
        # create label pd.Series from df_event[label_column_name].iloc[0] with length of extracted_features
        # features_filtered = select_features(extracted_features, label_series)

        # apply feature extraction, imputing (deleting missing values) and feature selection in one step
        # features_filtered_direct = extract_relevant_features(df_sensor, label_series, column_id='ID', column_sort=time_column_name)

        # add label, device_id and ESM_timestamp to features_filtered
        #sort df_intermediate by timestamp
        df_intermediate = df_intermediate.sort_values(by="timestamp")
        extracted_features["sensor_timestamp"] = df_intermediate.groupby("ID")[time_column_name].first()
        extracted_features[ESM_event_column_name] = df_intermediate.groupby("ID")[ESM_event_column_name].first()
        extracted_features["device_id"] = df_intermediate.groupby("ID")["device_id"].first()

        return extracted_features

    ## create function to select features
    def feature_selection(df_features):
        # check how many columns are in df_features
        num_features_before_selection = df_features.shape[1]

        # delete columns which contain only one value
        print("shape before deleting columns with only one value: ", df_features.shape)
        #df_features = df_features.loc[:, df_features.nunique() != 1] #computational complexity too high
        #df_features = df_features.loc[:, df_features.apply(pd.Series.nunique) != 1] #computational complexity too high
        drop_cols = df_features.columns[df_features.nunique() == 1]
        df_features.drop(columns=drop_cols, inplace=True)
        print("shape after deleting columns with only one value: ", df_features.shape)

        # delete column if more than 90% of values are missing
        print("shape before deleting columns with more than 90% missing values: ", df_features.shape)
        #print(df_features.columns[df_features.isnull().mean() > 0.9 ])
        drop_cols = df_features.columns[df_features.isnull().mean() > 0.9]
        df_features.drop(columns=drop_cols, inplace=True)
        # df_features = df_features.loc[:, df_features.isnull().mean() < .9] #computational complexity too high
        print("shape after deleting columns with more than 90% missing values: ", df_features.shape)

        # delete unnecessary columns
        device_id = df_features["device_id"]
        ESM_timestamp = df_features[ESM_identifier_column]
        sensor_timestamp = df_features["sensor_timestamp"]
        df_features = df_features.drop(columns = ["ESM_timestamp", "sensor_timestamp", "device_id", "sensor_timestamp"])
        print("unnecessary columns deleted")

        # impute missing values
        impute(df_features)
        #iterate through df_features in chunks of 1000 rows and impute missing values and save temporarily
        #for i in tqdm(range(0, df_features.shape[0], 10000)):
            #impute(df_features.iloc[i:i+10000, :])
        #    print("missing values imputed")
        #    df_features.iloc[i:i+10000, :] = df_features.iloc[i:i+10000, :].apply(lambda x: x.fillna(x.mean()), axis=0)
        print("imputation done")

        # add label & delete missing values in label
        # add label column
        df_features[ESM_identifier_column] = ESM_timestamp
        df_features["device_id"] = device_id
        df_features["sensor_timestamp"] = sensor_timestamp
        df_features = labeling_sensor_df(df_features, dict_label, label_column_name, ESM_identifier_column)
        # count missing values in df_features and groupby ESM_timestamp
        print("missing values in df_features: ", df_features.isnull().sum())
        print("shape of df_features before deleting missing values in label: ", df_features.shape)
        # delete rows which contain nan in label_column_name
        df_features.dropna(subset=[label_column_name], inplace=True)
        print("shape of df_features after deleting missing values in label: ", df_features.shape)

        # create updated series for label, device_id and ESM_timestamp
        label = pd.Series(df_features[label_column_name])
        sensor_timestamp = df_features["sensor_timestamp"]
        ESM_timestamp = df_features[ESM_identifier_column]
        device_id = df_features["device_id"]

        # delete label & ESM_timestamp column again
        df_features.drop(columns=[label_column_name, ESM_identifier_column, "device_id", "sensor_timestamp"], inplace=True)
        print("label created")

        # apply feature selection
        features_filtered = select_features(df_features, label)
        print("feature selection done")
        print("shape of df_features after feature selection : ", features_filtered.shape)
        #print how many features have been deleted
        num_features_after_selection = features_filtered.shape[1]
        print("number of features deleted: ", num_features_before_selection - num_features_after_selection)

        # add device id, ESM_timestamp and label column again
        features_filtered["device_id"] = device_id
        features_filtered[label_column_name] = label
        features_filtered[ESM_identifier_column] = ESM_timestamp
        features_filtered["sensor_timestamp"] = sensor_timestamp
        print("device id and label added")
        print("time needed for feature selection: ", time.time() - t0)

        return features_filtered

    ## visualization: count columns & get column names
    def count_feature_columns_and_list_columnnames(df_features, sensor_list):
        df_results = pd.DataFrame(columns=["timeperiod", "sensor", "number_features", "number_records"])
        dict_columns = {}
        axes = ["double_values_0", "double_values_1", "double_values_2" ]
        # iterate through sensors, take only first three letters and count how many columns in df_features start with that three letters
        # save the number of columns and rows in a list
        for sensor in sensors:
            for axis in axes:
                # count how many columns start with first three letters of that sensor and the axis
                number_features = len([col for col in df_features.columns if col.startswith(sensor[:3] + "_" + axis)])
                # save number of features and number of records in df_results using concat
                df_results = pd.concat([df_results,
                                        pd.DataFrame([[timeperiod, sensor, axis, number_features, df_features.shape[0]]],
                                                     columns=["timeperiod", "sensor", "axis", "number_features",
                                                              "number_records"])], axis=0)
            # save list of columns in dict_columns
            dict_columns[sensor] = [col for col in df_features.columns if col.startswith(sensor[0:3])]
            print("sensor finished: ", sensor)

        return df_results, dict_columns

    # for battery_charges
    ## calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
    ## battery double_end_timestamp" timestamp
    def battery_charges_features(self, df):
        # calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
        # battery double_end_timestamp" timestamp

        return df


