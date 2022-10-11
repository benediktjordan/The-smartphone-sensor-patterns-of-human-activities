# description: This script is dedicated to the correct selection of label segments around events
# based on, automated or manual, change point analysis
# The label selection is based on the merged sensordata from the high-frequency sensors

# import libraries
import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd
import numpy as np



# function to select labels


event = df["ESM_timestamp"].unique()[0]

# test
df = df_merged

# run vis function
# import data
path_merged_sensorfile = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/merged/all_esm_timeperiod_5 min_TimeseriesMerged.csv"
df_merged = pd.read_csv(path_merged_sensorfile, parse_dates=["timestamp", "ESM_timestamp"], nrows=600000)

df = df_merged
sensors_to_use = ["linear_accelerometer", "gyroscope", "rotation"]
segment_around_event = 120 # in seconds
ESM_column = "ESM_bodyposition"
activities = df_merged["ESM_bodyposition"].unique()
number_breakpoints = 8 # number of change points
breakpoint_method = "window-based" # method for change point detection
path_save_figures = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/label_selection/"


def select_labels_visualize(df, sensors_to_use, sensor_columns, segment_around_event, ESM_column, activity,
                            number_breakpoints, breakpoint_method, path_save_figures):
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

    # convert timestamp to unix time
    df["timestamp_unix"] = df["timestamp"].astype(np.int64)

    # set timestamp as index
    df = df.set_index("timestamp_unix")


    # iterate through events and select segments around events
    for event in df["ESM_timestamp"].unique():
        # select segment around event
        df_segment = df[(df["ESM_timestamp"] == event) & (df["timestamp"] >= event - pd.Timedelta(seconds=segment_around_event)) & (df["timestamp"] <= event + pd.Timedelta(seconds=segment_around_event))]

        # select sensor columns
        df_segment = df_segment[sensor_columns_touse[:-3]]

        # convert to numpy array
        df_segment = df_segment.to_numpy()

        # breakpoint detection
        # select breakpoint method
        if breakpoint_method == "binary segmentation":
            algo = rpt.Binseg(model="l2").fit(df_segment)
            # Changepoint detection with the Binary Segmentation search method
            model = "l2"
            algo = rpt.Binseg(model=model).fit(df_segment)
            my_bkps = algo.predict(n_bkps=number_breakpoints)  # number of change points
            # show results
            rpt.show.display(df_segment, my_bkps, figsize=(10, 6))
            plt.title('Change Point Detection: Binary Segmentation Search Method')
            plt.show()

        elif breakpoint_method == "dynamic programming":
            # Changepoint detection with dynamic programming search method
            model = "l1"
            algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(df_segment)
            my_bkps = algo.predict(n_bkps=number_breakpoints)
            rpt.show.display(df_segment, my_bkps, figsize=(10, 6))
            plt.title('Change Point Detection: Dynamic Programming Search Method')
            plt.show()

        elif breakpoint_method == "window-based":
            # Changepoint detection with window-based search method
            model = "l2"
            algo = rpt.Window(width=11, model=model).fit(df_segment)
            my_bkps = algo.predict(n_bkps=number_breakpoints)
            rpt.show.display(df_segment, my_bkps, figsize=(10, 6))
            plt.title("activity:" + str(activity) + " event: " + str(event) + 'Window-Based')
            # save figure
            plt.savefig(path_save_figures + "label_selection_" + str(activity) + "_" + str(event) + ".png")
            plt.show()

        elif breakpoint_method == "pelt":
            # Changepoint detection with the Pelt search method
            model = "rbf"
            algo = rpt.Pelt(model=model).fit(df_segment)
            result = algo.predict(pen=number_breakpoints)
            rpt.display(df_segment, result, figsize=(10, 6))
            plt.title('Change Point Detection: Pelt Search Method')
            plt.show()
    return
for activity in activities:
    select_labels_visualize(df, sensors_to_use, sensor_columns, segment_around_event, ESM_column, activity,
                            number_breakpoints, breakpoint_method, path_save_figures)
select_labels_visualize(df, sensors_to_use, sensor_columns, segment_around_event, ESM_column, activity,
                        number_breakpoints, breakpoint_method, path_save_figures)
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