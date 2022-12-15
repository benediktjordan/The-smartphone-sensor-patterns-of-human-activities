# region import
import numpy as np
import pandas as pd
import pickle
# endregion


#region load, transform & label sensor data
dir_dataset = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben"
list_sensors =  ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "locations","plugin_ios_activity_recognition"]
path_save = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben/"

# create dictionary with start and end times of different activities; start_session and end_session should be datetime objects
dict_label_iteration02_Ben = {"standing": {"start_session": pd.Timestamp("2022-11-30 15:44:00"), "end_session": pd.Timestamp("2022-11-30 15:46:00")},
                "walking_mediumspeed": {"start_session": pd.Timestamp("2022-11-30 15:46:00"), "end_session": pd.Timestamp("2022-11-30 15:48:00")},
                "walking_lowspeed": {"start_session": pd.Timestamp("2022-11-30 15:48:00"), "end_session": pd.Timestamp("2022-11-30 15:50:00")},
              "sitting_onthecouch": {"start_session": pd.Timestamp("2022-11-30 15:50:00"), "end_session": pd.Timestamp("2022-11-30 15:52:00")},
                "walking_fastspeed": {"start_session": pd.Timestamp("2022-11-30 15:52:00"), "end_session": pd.Timestamp("2022-11-30 15:54:00")},
                "cycling": {"start_session": pd.Timestamp("2022-11-30 15:56:00"), "end_session": pd.Timestamp("2022-11-30 15:58:00")},
            "cycling_includingstops": {"start_session": pd.Timestamp("2022-11-30 15:58:00"), "end_session": pd.Timestamp("2022-11-30 16:01:00")},
                "running": {"start_session": pd.Timestamp("2022-11-30 16:03:00"), "end_session": pd.Timestamp("2022-11-30 16:05:00")},
                              "on the toilet": {"start_session": pd.Timestamp("2022-12-06 12:01:00"), "end_session": pd.Timestamp("2022-12-06 12:06:00")},
                                "lying_phoneinfront": {"start_session": pd.Timestamp("2022-12-06 12:07:30"), "end_session": pd.Timestamp("2022-12-06 12:12:30")},
                                "lying_phoneoverhead": {"start_session": pd.Timestamp("2022-12-06 12:12:30"), "end_session": pd.Timestamp("2022-12-06 12:17:30")},
                                "lying_phoneonbed": {"start_session": pd.Timestamp("2022-12-06 12:17:30"), "end_session": pd.Timestamp("2022-12-06 12:22:30")},
                                "sitting_attable_phoneinhand": {"start_session": pd.Timestamp("2022-12-06 12:23:00"), "end_session": pd.Timestamp("2022-12-06 12:28:00")},
                                "sitting_attable_phoneontable": {"start_session": pd.Timestamp("2022-12-06 12:28:30"), "end_session": pd.Timestamp("2022-12-06 12:33:30")},
                              }

# save dictionary as pickle file
with open(path_save + "/dict_label_iteration02_Ben.pkl", "wb") as handle:
    pickle.dump(dict_label_iteration02_Ben, handle, protocol=pickle.HIGHEST_PROTOCOL)

#label the data
for sensor in list_sensors:
    print("Name of Sensor " + sensor)
    try:
        df_sensor = pd.read_csv(dir_dataset + "/" + sensor + ".csv")
    except: #for one file, pd.read_csv() does not work; here is the workaround
        with open(dir_dataset + "/" + sensor + ".csv", 'r') as temp_f:
            # get No of columns in each line
            col_count = [len(l.split(",")) for l in temp_f.readlines()]
        column_names = [i for i in range(0, max(col_count))]
        df_sensor = pd.read_csv(dir_dataset + "/" + sensor + ".csv", header=None,
                                delimiter=",", names=column_names)
        df_sensor.columns = df_sensor.iloc[0]
        df_sensor = df_sensor.iloc[1:]

    #create datetime timestamp column and add one hour (in order to align to Berlin timezone)
    df_sensor["timestamp_unix"] = df_sensor["timestamp"].copy()
    df_sensor["timestamp"] = pd.to_datetime(df_sensor["timestamp_unix"], unit="ms") + pd.Timedelta(hours=1)
    # label the data
    for activity in dict_label.keys():
        df_sensor.loc[(df_sensor["timestamp"] >= dict_label_iteration02_Ben[activity]["start_session"]) & (df_sensor["timestamp"] <= dict_label[activity]["end_session"]), "label_human motion - general"] = activity

    #delete all rows with nan as label
    df_sensor = df_sensor.dropna(subset=["label_human motion - general"])

    # save the data
    df_sensor.to_csv(dir_dataset + "/" + sensor + "_labeled.csv", index=False)
#endregion

#region visualize data

#endregion
