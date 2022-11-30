# region import
import numpy as np
import pandas as pd
# endregion

#region load, transform & label sensor data
dir_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/perfectdata"
list_sensors =  ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "locations","plugin_ios_activity_recognition"]

# create dictionary with start and end times of different activities; start_session and end_session should be datetime objects
dict_label = {"standing": {"start_session": pd.Timestamp("2022-11-30 15:44:00"), "end_session": pd.Timestamp("2022-11-30 15:46:00")},
                "walking_mediumspeed": {"start_session": pd.Timestamp("2022-11-30 15:46:00"), "end_session": pd.Timestamp("2022-11-30 15:48:00")},
                "walking_lowspeed": {"start_session": pd.Timestamp("2022-11-30 15:48:00"), "end_session": pd.Timestamp("2022-11-30 15:50:00")},
              "sitting_onthecouch": {"start_session": pd.Timestamp("2022-11-30 15:50:00"), "end_session": pd.Timestamp("2022-11-30 15:52:00")},
                "walking_fastspeed": {"start_session": pd.Timestamp("2022-11-30 15:52:00"), "end_session": pd.Timestamp("2022-11-30 15:54:00")},
                "cycling": {"start_session": pd.Timestamp("2022-11-30 15:56:00"), "end_session": pd.Timestamp("2022-11-30 15:58:00")},
            "cycling_includingstops": {"start_session": pd.Timestamp("2022-11-30 15:58:00"), "end_session": pd.Timestamp("2022-11-30 16:01:00")},
                "running": {"start_session": pd.Timestamp("2022-11-30 16:03:00"), "end_session": pd.Timestamp("2022-11-30 16:05:00")}}

for sensor in list_sensors:
    print("Name of Sensor " + sensor)
    try:
        df_sensor = pd.read_csv(dir_dataset + "/" + sensor + ".csv")
    except:
        # Loop the data lines
        with open(dir_dataset + "/" + sensor + ".csv", 'r') as temp_f:
            # get No of columns in each line
            col_count = [len(l.split(",")) for l in temp_f.readlines()]

        ### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
        column_names = [i for i in range(0, max(col_count))]
        # endregion
        df_sensor = pd.read_csv(dir_dataset + "/" + sensor + ".csv", header=None,
                                delimiter=",", names=column_names)
        # convert first row into column names and delete first row
        df_sensor.columns = df_sensor.iloc[0]
        df_sensor = df_sensor.iloc[1:]

    #create datetime timestamp column and add one hour (in order to align to Berlin timezone)
    df_sensor["timestamp_datetime"] = pd.to_datetime(df_sensor["timestamp"], unit="ms") + pd.Timedelta(hours=1)
    # label the data
    for activity in dict_label.keys():
        df_sensor.loc[(df_sensor["timestamp_datetime"] >= dict_label[activity]["start_session"]) & (df_sensor["timestamp_datetime"] <= dict_label[activity]["end_session"]), "label"] = activity
    # save the data
    df_sensor.to_csv(dir_dataset + "/" + sensor + "_labeled.csv", index=False)
#endregion

#region visualize data

#endregion
