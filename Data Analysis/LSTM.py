#region import libraries
## for Keras LSTM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
#endregion

#region data preparation
## import data
base_sensor = "linear_accelerometer"
sensors = ["gravity",
           "gyroscope", "magnetometer", "rotation"]
#sensors = ["accelerometer", "gravity",
#           "gyroscope", "magnetometer", "rotation"]

## merging function
def merge_unaligned_timeseries(df_base, df_tomerge, merge_sensor):
    df_final = pd.DataFrame()

    # iterate through devices and ESM_timestamps
    for user in df_base['2'].unique():
        print("Current user is: ", user)
        for event in df_base['ESM_timestamp'].unique():
            print("Current event is: ", event)
            # get data for specific user and ESM event
            df_base_user_event = df_base[(df_base['2'] == user) & (df_base['ESM_timestamp'] == event)]
            df_sensor_user_event= df_tomerge[(df_tomerge['2'] == user) & (df_tomerge['ESM_timestamp'] == event)]

            # sort dataframes by timestamp
            df_base_user_event = df_base_user_event.sort_values(by='timestamp')
            df_sensor_user_event = df_sensor_user_event.sort_values(by='timestamp')

            # duplicate timestamp column for test purposes
            df_sensor_user_event['timestamp_'+str(merge_sensor)] = df_sensor_user_event['timestamp']

            # delete all ESM-related columns in df_sensor_user_event (otherwise they would be duplicated)
            df_sensor_user_event = df_sensor_user_event.drop(columns=['ESM_timestamp', "ESM_location", "ESM_location_time",
                                                                      "ESM_bodyposition", "ESM_bodyposition_time",
                                                                      "ESM_activity", "ESM_activity_time",
                                                                      "ESM_smartphonelocation", "ESM_smartphonelocation_time",
                                                                      "ESM_aligned", "ESM_aligned_time"])
            # delete columns "Unnamed: 0", "0", "1" and "2" from df_sensor_user_event: all the information of these
            # columns is already contained in the JSON data
            df_sensor_user_event = df_sensor_user_event.drop(columns=['Unnamed: 0', '0', '1', '2'])


            # merge dataframes
            df_merged = pd.merge_asof(df_base_user_event, df_sensor_user_event, on='timestamp',
                                      tolerance=pd.Timedelta("100ms"))

            # add merged data to general dataframe
            df_final = df_final.append(df_merged)

    return df_final


## iterate through sensors
df_base = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/"+str(base_sensor) + "_esm_timeperiod_5 min.csv_JSONconverted.csv",
                      parse_dates=['timestamp'], infer_datetime_format=True)

#region temporary
df_base = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/accelerometer_esm_timeperiod_5 min_TimeseriesMerged.csv",
                      parse_dates=['timestamp', "1_x", "timestamp_accelerometer", "1_y"], infer_datetime_format=True)

test = df_base["1_x"]-df_base["timestamp"]
test2 = df_base["1_y"]-df_base["timestamp_accelerometer"]
test.describe()
test2.describe()

df_base = df_base.drop(columns=['ESM_timestamp_y', "ESM_location_y", "ESM_location_time_y",
                                                          "ESM_bodyposition_y", "ESM_bodyposition_time_y",
                                                          "ESM_activity_y", "ESM_activity_time_y",
                                                          "ESM_smartphonelocation_y", "ESM_smartphonelocation_time_y",
                                                          "ESM_aligned_y", "ESM_aligned_time_y",
                                "Unnamed: 0_y", "0_y", "1_y", "2_y", "3_y"])
df_base = df_base.rename(columns={"Unnamed: 0_x": "Unnamed: 0", "0_x": "0", "1_x": "1", "2_x": "2", "3_x": "3",
                                  "ESM_timestamp_x": "ESM_timestamp", "ESM_location_x": "ESM_location",
                                  "ESM_location_time_x": "ESM_location_time", "ESM_bodyposition_x": "ESM_bodyposition",
                                  "ESM_bodyposition_time_x": "ESM_bodyposition_time", "ESM_activity_x": "ESM_activity",
                                  "ESM_activity_time_x": "ESM_activity_time", "ESM_smartphonelocation_x": "ESM_smartphonelocation",
                                  "ESM_smartphonelocation_time_x": "ESM_smartphonelocation_time", "ESM_aligned_x": "ESM_aligned",
                                  "ESM_aligned_time_x": "ESM_aligned_time"})

#endregion temporary

for sensor in sensors:
    time_begin = time.time()
    print("Current sensor is: ", sensor)
    df_sensor = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/" + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.csv",
                            parse_dates=['timestamp'], infer_datetime_format=True)
    df_base = merge_unaligned_timeseries(df_base, df_tomerge=df_sensor, merge_sensor=sensor)
    df_base.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/" + str(sensor) + "_esm_timeperiod_5 min_TimeseriesMerged.csv", index=False)
    time_end = time.time()
    print("Time for sensor ", sensor, " is: ", time_end - time_begin)
# save merged data
df_base.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_timeperiod_5 min_TimeseriesMerged.csv", index=False)


#region Keras
# fix random seed for reproducibility
tf.random.set_seed(7)
