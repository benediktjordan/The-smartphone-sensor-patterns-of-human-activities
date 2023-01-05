#region import
import numpy as np
import pandas as pd
import pickle
import os
#endregion

class InitialTransforms_Iteration02:

    # label sensorfiles
    ## + adjust time to Berlin timezone (1 hour difference)
    ## + drop rows which have NaN value in label column
    def label_sensor_data(path_sensorfile, dict_label):
        try:
            df_sensor = pd.read_csv(path_sensorfile)
        except:  # for one file, pd.read_csv() does not work; here is the workaround
            with open(path_sensorfile, 'r') as temp_f:
                # get No of columns in each line
                col_count = [len(l.split(",")) for l in temp_f.readlines()]
            column_names = [i for i in range(0, max(col_count))]
            df_sensor = pd.read_csv(path_sensorfile, header=None,
                                    delimiter=",", names=column_names)
            df_sensor.columns = df_sensor.iloc[0]
            df_sensor = df_sensor.iloc[1:]

        # check if sensorfile is empty (no data)
        if len(df_sensor) == 0:
            print("Sensorfile is empty")
            # return empty dataframe and stop function
            return None


        # create datetime timestamp column and add one hour (in order to align to Berlin timezone)
        df_sensor["timestamp_unix"] = df_sensor["timestamp"].copy()
        df_sensor["timestamp"] = pd.to_datetime(df_sensor["timestamp_unix"], unit="ms") + pd.Timedelta(hours=1)

        # label the data
        for activity in dict_label.keys():
            df_sensor.loc[(df_sensor["timestamp"] >= dict_label[activity]["start_session"]) & (
                        df_sensor["timestamp"] <= dict_label[activity][
                    "end_session"]), "label_human motion - general"] = activity

        # delete all rows with nan as label
        df_sensor = df_sensor.dropna(subset=["label_human motion - general"])

        return df_sensor

    # add ESM timestamps

