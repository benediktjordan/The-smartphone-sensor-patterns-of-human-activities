# description: This script is dedicated to the correct selection of label segments around events
# based on, automated or manual, change point analysis
# The label selection is based on the merged sensordata from the high-frequency sensors

# import libraries
import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd
import numpy as np

# import data
path_merged_sensorfile = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/merged/all_esm_timeperiod_5 min_TimeseriesMerged.csv"
df_merged = pd.read_csv(path_merged_sensorfile ,nrows = 100000, parse_dates=["timestamp", "ESM_timestamp"])


# function to select labels

sensors_to_use = ["accelerometer", "linear_accelerometer"]
segment_around_event = 60 # in seconds
ESM_column = "ESM_bodyposition"
activity = "lying"
n_bkps = 2 # number of change points
def select_labels_visualize(df, sensors_to_use, sensor_columns, segment_around_event, ESM_column, activity, n_bkps):
    # select sensor columns from sensor_columns dictionary based on sensors_to_use list
    sensor_columns_touse = []
    for sensor in sensors_to_use:
        sensor_columns_touse.append(sensor_columns[sensor])
    sensor_columns_touse = [item for sublist in sensor_columns_touse for item in sublist]
    sensor_columns_touse.append("timestamp")
    sensor_columns_touse.append("ESM_timestamp")
    sensor_columns_touse.append(ESM_column)

    # select sensor columns from df
    df = df[sensor_columns_touse]

    # select activity
    df = df[df[ESM_column] == activity]

    # iterate through events and select segments around events
    for event in df["ESM_timestamp"].unique():
        # select segment around event
        df_segment = df[(df["ESM_timestamp"] == event) & (df["timestamp"] >= event - pd.Timedelta(seconds=segment_around_event)) & (df["timestamp"] <= event + pd.Timedelta(seconds=segment_around_event))]

        # detect change points
        model = rpt.Dynp(model="l1")
        model.fit(df_segment)
        breaks = model.predict(n_bkps=n_bkps)

        df_segment = df_segment.drop(columns=["ESM_timestamp", "timestamp", ESM_column])
        model = rpt.Pelt(model="rbf").fit(df_segment)
        breaks = algo.predict(pen=10)

        # display

        rpt.display(df_segment, result)
        plt.show()

        # plot segment
        fig, ax = plt.subplots()
        for sensor in sensors_to_use:
            ax.plot(df_segment["timestamp"], df_segment[sensor_columns[sensor]], label=sensor)
        ax.legend()
        ax.set_title(f"Activity: {activity}, Event: {event}")
        plt.show()






    sensor_column_names.append(time_column_name)
    sensor_column_names.append("ID")

    # select columns
    df = df[sensor_columns]

    # iterate through events and apply change point analysis

    return df


# testrun


# testing perform change point analysis
# generate signal
n_samples, dim, sigma = 1000, 3, 4
n_bkps = 2  # number of breakpoints
signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

# detection
algo = rpt.Pelt(model="rbf").fit(signal)
result = algo.predict(pen=10)

# display
rpt.display(signal, bkps, result)
plt.show()