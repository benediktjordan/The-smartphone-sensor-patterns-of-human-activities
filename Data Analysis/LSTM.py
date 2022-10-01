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

#endregion

#region data preparation
## import data
base_sensor = "linear_accelerometer"
sensors = ["accelerometer", "gravity",
           "gyroscope", "magnetometer", "rotation"]

## merging function
def merge_unaligned_timeseries(df_base, df_tomerge):
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
            df_sensor_user_event['timestamp2'] = df_sensor_user_event['timestamp']

            # merge dataframes
            df_merged = pd.merge_asof(df_base_user_event, df_sensor_user_event, on='timestamp',
                                      tolerance=pd.Timedelta("99ms"))

            # add merged data to general dataframe
            df_final = df_final.append(df_merged)

    return df_final

# count NaN values in dataframe
df_sensor_user_event.isnull().sum()
df_base_user_event.dtypes

df_base = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_" + str(base_sensor) + "_esm_timeperiod_5 min.csv_JSONconverted.csv", nrows= 100000)
df_tomerge = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_" + str(sensors[0]) + "_esm_timeperiod_5 min.csv_JSONconverted.csv", nrows= 100000)

## convert timestamp to datetime
df_base['timestamp'] = pd.to_datetime(df_base['timestamp'], unit='ms')
df_tomerge['timestamp'] = pd.to_datetime(df_tomerge['timestamp'], unit='ms')

df_merged = merge_unaligned_timeseries(df_base, df_tomerge)

# test result
df_merged.isnull().sum()
df_base.isnull().sum()
df_tomerge.isnull().sum()


### Trial
path_sensor1 = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_accelerometer_esm_timeperiod_5 min.csv_JSONconverted.csv"
path_sensor2 = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_gyroscope_esm_timeperiod_5 min.csv_JSONconverted.csv"
df_sensor1 = pd.read_csv(path_sensor1, nrows=100000)
df_sensor2 = pd.read_csv(path_sensor2, nrows=100000)

## convert timestamp to datetime
df_sensor1['timestamp'] = pd.to_datetime(df_sensor1['timestamp'], unit='ms')
df_sensor2['timestamp'] = pd.to_datetime(df_sensor2['timestamp'], unit='ms')

## iterate through devices and ESM_timestamps
for user in df_sensor1['2'].unique():
    for event in df_sensor1['ESM_timestamp'].unique():
        # get data for specific user and ESM event
        df_sensor1_user_timestamp = df_sensor1[(df_sensor1['2'] == user) & (df_sensor1['ESM_timestamp'] == event)]
        df_sensor2_user_timestamp = df_sensor2[(df_sensor2['2'] == user) & (df_sensor2['ESM_timestamp'] == event)]





## check out how matching could work for different timestamps
df_sensor1_user_timestamp = df_sensor1_user_timestamp[['timestamp']]
df_sensor1_user_timestamp["data"] = 1
df_sensor1_user_timestamp["index"] = df_sensor1_user_timestamp.index
df_sensor2_user_timestamp = df_sensor2_user_timestamp[['timestamp']]
df_sensor2_user_timestamp["timestamp2"] = df_sensor2_user_timestamp["timestamp"]
df_sensor2_user_timestamp["data"] = 2
df_sensor2_user_timestamp["index"] = df_sensor2_user_timestamp.index
## concatenate dataframes
df_sensor1_user_timestamp = df_sensor1_user_timestamp.append(df_sensor2_user_timestamp)
## sort by timestamp
df_together = df_sensor1_user_timestamp.sort_values(by=['timestamp'])

## try merging with pandas
df_merged = pd.merge_asof (df_sensor1_user_timestamp, df_sensor2_user_timestamp, on='timestamp', tolerance=pd.Timedelta("99ms"))

#region Keras
# fix random seed for reproducibility
tf.random.set_seed(7)
