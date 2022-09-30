# region import
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json
#from datetime import *


# endregion

# changed it commment 3

#region visualizations

# region visualize summary of ESM data for every question
def visualize_esm_summary(esm_all_transformed, path):
    # create summary of ESM data
    esm_summary = {
        "location": esm_all_transformed["location"].value_counts(),
        "bodyposition": esm_all_transformed["bodyposition"].value_counts(),
        "activity": esm_all_transformed["activity"].value_counts(),
        "smartphonelocation": esm_all_transformed["smartphonelocation"].value_counts(),
        "aligned": esm_all_transformed["aligned"].value_counts()
    }

    # create bar plots for every question
    for activity in esm_summary:
        plt.figure(figsize=(15, 10))
        plt.title("Number of total events for " + activity)
        sns.barplot(x=esm_summary[activity][0:10].index, y=esm_summary[activity][0:10].values, palette="Blues_d")
        plt.xticks(rotation=15)
        plt.savefig(path + "/" + activity + "_ESM activity count.png")
        plt.show()

visualize_esm_summary(esm_all_transformed, "H:/InUsage/Results/visualizations/esm")

#endregion


# region create function which visualizes the data in lineplot for sensor, participant, ESN event, ESM answer and in timeperiod
dir_databases = "H:/InUsage/Databases"
esm_all_transformed = pd.read_csv(dir_databases + "/esm_all_transformed.csv")

# only data for one participant
esm_all_transformed_p1 = esm_all_transformed[esm_all_transformed["device_id"] == "8206b501-20e7-4851-8b79-601af9e4485f"]

participant_id = "8206b501-20e7-4851-8b79-601af9e4485f"
sensor = "accelerometer"
event_time = 1656166542710.00000
esm_answer = "walking"
time_period = 60 #seconds
sensordata_path = "H:/InUsage/Databases/db_iteration1_20220628/accelerometer.csv"



def vis_event_acc(participant_ID, event_time, esm_answer, time_period, sensordata_path):
    # create df with sensor data around event_time of participant_ID
    df = pd.read_csv(sensordata_path)
    df_filtered = df[(df["1"] >= event_time - time_period  * 1000) & (df["1"] <= event_time + time_period  * 1000)]
    df_filtered = df_filtered[df_filtered["2"] == participant_id]

    # transform unix time into seconds relative to event_time
    df_filtered.sort_values(by=['1'], inplace=True)
    df_filtered["1"] = (df_filtered["1"] - event_time)/1000
    #df_filtered["1"] = (df_filtered["1"] - df_filtered["1"].iloc[0])/1000

    # transform JSON data into columns
    df_filtered = df_filtered.reset_index(drop=True)
    stdf = df_filtered["3"].apply(json.loads)
    df_transformed = pd.DataFrame(stdf.tolist())  # or stdf.apply(pd.Series)
    df_transformed = pd.concat([df_filtered, df_transformed], axis=1)

    # visualize for every one of the three axis the data in a lineplot
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    axs[0].plot(df_transformed["1"], df_transformed["double_values_0"], color="blue", label="x")
    axs[1].plot(df_transformed["1"], df_transformed["double_values_1"], color="red", label="y")
    axs[2].plot(df_transformed["1"], df_transformed["double_values_2"], color="green", label="z")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)


#test runs

## decide event time
dir_databases = "H:/InUsage/Databases"
esm_all_transformed = pd.read_csv(dir_databases + "/esm_all_transformed.csv")
esm_all_transformed.sort_values(by=['timestamp'], inplace=True)
esm_all_transformed["timestamp"] = pd.to_datetime(esm_all_transformed['timestamp'],unit='ms')


# only data for one participant
esm_all_transformed_p1 = esm_all_transformed[esm_all_transformed["device_id"] == participant_id]
esm_all_transformed_p1_sitting = esm_all_transformed[esm_all_transformed["bodyposition"] == "[\"sitting: at a table\"]"]
esm_all_transformed_p1_standing = esm_all_transformed[esm_all_transformed["bodyposition"] == "[\"standing\"]"]
esm_all_transformed_p1_lying = esm_all_transformed[esm_all_transformed["bodyposition"] == "[\"lying\"]"]


# participant: "8206b501-20e7-4851-8b79-601af9e4485f"
## walking (db_iteration1_20220624)
event_time_list = [1656166542710.00000,
                   1656219883193.00000,
                   1656312067352.00000,
                   1656327932865.00000,
                   1656338500955.00000,
                   1656347904183.00000,
                   1656397267213.00000]
event_time = 1656166542710.00000
event_time = 1656219883193.00000
event_time = 1656312067352.00000
event_time = 1656327932865.00000
event_time = 1656338500955.00000
event_time = 1656347904183.00000
event_time = 1656397267213.00000

## cycling (db_iteration1_20220628)
event_time = 1656486039637.00000
event_time = 1656625509247.00000

## sitting: at a table, smartphone in hand (db_iteration1_20220624)
event_time_list = [1656079231617.00000,
                   1656338426511.00000]

## standing, smartphone in hand (db_iteration1_20220624)
event_time_list = [1656091225973.00000,
                   1656100136895.00000,
                   1656153702404.00000,
                   1656324595078.00000,
                   1656331226610.00000,
                   1656361973954.00000]

## lying, smartphone in hand (db_iteration1_20220624)
event_time_list = [1656104411911.00000,
                   1656280341624.00000,
                   1656349255223.00000]



participant_id = "8206b501-20e7-4851-8b79-601af9e4485f"
sensor = "gyroscope"
esm_answer = "sitting: at a table; smartphone in hand"
time_period = 30#seconds
time_period_list = [30,300,600]
sensordata_path = "H:/InUsage/Databases/db_iteration1_20220624/" +sensor+".csv"

## load data
time_begin = time.time()
df = pd.read_csv(sensordata_path)
time_end = time.time()
print("read csv in " + str(time_end - time_begin) + " seconds")

def vis_event_acc_linacc(df_sensordata, participant_ID, event_time, esm_answer, time_period):

    ## filter data for time period around event time
    df_filtered = df[(df["1"] >= event_time - time_period  * 1000) & (df["1"] <= event_time + time_period  * 1000)]
    df_filtered = df_filtered[df_filtered["2"] == participant_id]

    # transform unix time into seconds relative to event_time
    df_filtered.sort_values(by=['1'], inplace=True)
    df_filtered["1"] = (df_filtered["1"] - event_time)/1000
    #df_filtered["1"] = (df_filtered["1"] - df_filtered["1"].iloc[0])/1000

    # transform JSON data into columns
    df_filtered = df_filtered.reset_index(drop=True)
    stdf = df_filtered["3"].apply(json.loads)
    df_transformed = pd.DataFrame(stdf.tolist())  # or stdf.apply(pd.Series)
    df_transformed = pd.concat([df_filtered, df_transformed], axis=1)

    # visualize for every one of the three axis the data in a lineplot

    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle("participant 1 " + "" + sensor + " " + esm_answer, fontsize=36)

    axs[0].plot(df_transformed["1"], df_transformed["double_values_0"], color="blue", label="x")
    axs[1].plot(df_transformed["1"], df_transformed["double_values_1"], color="red", label="y")
    axs[2].plot(df_transformed["1"], df_transformed["double_values_2"], color="green", label="z")

    ## set y-axis limi
    axs[0].set_ylim([-1, 1])
    axs[1].set_ylim([-1, 1])
    axs[2].set_ylim([-1, 1])

    ## Tight layout often produces nice results
    ## but requires the title to be spaced accordingly
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()

    # save figure
    fig.savefig("D:/MasterThesis @nlbb/Iteration01/Results/visualizations/bodyposition/" + participant_id + "_" + sensor + "_" + esm_answer + "_" + str(time_period) + "s_event_time_" + str(event_time) +".png")
    plt.close()

for event_time in event_time_list:
    for time_period in time_period_list:
        vis_event_acc_linacc(df, participant_id, event_time, esm_answer, time_period)




#endregion


#region create summary statistics for every sensor for different time periods around ESM events
def create_summary_statistics_for_sensors(path_sensor, sensor, time_period, dir_results):
    # create summary stats dataframe
    device_id = []
    ESM_timestamp = []
    ESM_location = []
    ESM_location_time = []
    ESM_bodyposition = []
    ESM_bodyposition_time = []
    ESM_activity = []
    ESM_activity_time = []
    ESM_smartphonelocation = []
    ESM_smartphonelocation_time = []
    ESM_aligned = []
    ESM_aligned_time = []
    double_values_0_mean = []
    double_values_0_std = []
    double_values_1_mean = []
    double_values_1_std = []
    double_values_2_mean = []
    double_values_2_std = []
    double_values_3_mean = []
    double_values_3_std = []
    double_speed_mean = []
    double_speed_std = []
    double_bearing_mean = []
    double_bearing_std = []
    double_altitude_mean = []
    double_altitude_std = []
    double_altitude_mean = []
    double_altitude_std = []
    double_longitude_mean = []
    double_longitude_std = []
    accuracy_mean = []
    accuracy_std = []

    # track missing sensordata around events
    missing_sensordata = []

    summary_stats = pd.DataFrame()

    # load sensor csv
    sensordata = pd.read_csv(path_sensor, dtype={"0": 'int64', "1": 'float64', "2": object, "3": object})

    # iterate through esm events
    esm_events = sensordata["ESM_timestamp"].unique()
    for esm_event in esm_events:
        print("esm_event: " + str(esm_event))
        # create dataframe for every esm event and time segment around event
        esm_event_df = sensordata[sensordata["ESM_timestamp"] == esm_event]
        esm_event_df = esm_event_df[(esm_event_df["1"] >= esm_event - time_period * 1000) & (esm_event_df["1"] <= esm_event + time_period * 1000)]

        if len(esm_event_df) > 0:
            # transform JSON data into columns
            esm_event_df = esm_event_df.reset_index(drop=True)
            stdf = esm_event_df["3"].apply(json.loads)
            df_transformed = pd.DataFrame(stdf.tolist())  # or stdf.apply(pd.Series)
            df_transformed = pd.concat([esm_event_df, df_transformed], axis=1)

            # create summary stats for every esm event
            summary_stats_esm_event = df_transformed.describe()

            # add esm event to summary stats lists
            device_id.append(esm_event_df["2"].iloc[0])
            ESM_timestamp.append(esm_event)
            ESM_location.append(esm_event_df["ESM_location"].iloc[0])
            ESM_location_time.append(esm_event_df["ESM_location_time"].iloc[0])
            ESM_bodyposition.append(esm_event_df["ESM_bodyposition"].iloc[0])
            ESM_bodyposition_time.append(esm_event_df["ESM_bodyposition_time"].iloc[0])
            ESM_activity.append(esm_event_df["ESM_activity"].iloc[0])
            ESM_activity_time.append(esm_event_df["ESM_activity_time"].iloc[0])
            ESM_smartphonelocation.append(esm_event_df["ESM_smartphonelocation"].iloc[0])
            ESM_smartphonelocation_time.append(esm_event_df["ESM_smartphonelocation_time"].iloc[0])
            ESM_aligned.append(esm_event_df["ESM_aligned"].iloc[0])
            ESM_aligned_time.append(esm_event_df["ESM_aligned_time"].iloc[0])

            # differentiate between sensor types
            if sensor in ["accelerometer", "magnetometer"]:
                accuracy_mean.append(summary_stats_esm_event["accuracy"]["mean"])
                accuracy_std.append(summary_stats_esm_event["accuracy"]["std"])
                double_values_0_mean.append(summary_stats_esm_event["double_values_0"]["mean"])
                double_values_0_std.append(summary_stats_esm_event["double_values_0"]["std"])
                double_values_1_mean.append(summary_stats_esm_event["double_values_1"]["mean"])
                double_values_1_std.append(summary_stats_esm_event["double_values_1"]["std"])
                double_values_2_mean.append(summary_stats_esm_event["double_values_2"]["mean"])
                double_values_2_std.append(summary_stats_esm_event["double_values_2"]["std"])

            elif sensor in ["barometer"]:
                accuracy_mean.append(summary_stats_esm_event["accuracy"]["mean"])
                accuracy_std.append(summary_stats_esm_event["accuracy"]["std"])
                double_values_0_mean.append(summary_stats_esm_event["double_values_0"]["mean"])
                double_values_0_std.append(summary_stats_esm_event["double_values_0"]["std"])

            elif sensor in ["locations"]:
                accuracy_mean.append(summary_stats_esm_event["accuracy"]["mean"])
                accuracy_std.append(summary_stats_esm_event["accuracy"]["std"])
                double_speed_mean.append(summary_stats_esm_event["double_speed"]["mean"])
                double_speed_std.append(summary_stats_esm_event["double_speed"]["std"])
                double_bearing_mean.append(summary_stats_esm_event["double_bearing"]["mean"])
                double_bearing_std.append(summary_stats_esm_event["double_bearing"]["std"])
                double_altitude_mean.append(summary_stats_esm_event["double_altitude"]["mean"])
                double_altitude_std.append(summary_stats_esm_event["double_altitude"]["std"])
                double_altitude_meanitude_mean.append(summary_stats_esm_event["double_latitude"]["mean"])
                double_altitude_std.append(summary_stats_esm_event["double_latitude"]["std"])
                double_longitude_mean.append(summary_stats_esm_event["double_longitude"]["mean"])
                double_longitude_std.append(summary_stats_esm_event["double_longitude"]["std"])

            elif sensor in ["rotations"]:
                accuracy_mean.append(summary_stats_esm_event["accuracy"]["mean"])
                accuracy_std.append(summary_stats_esm_event["accuracy"]["std"])
                double_values_0_mean.append(summary_stats_esm_event["double_values_0"]["mean"])
                double_values_0_std.append(summary_stats_esm_event["double_values_0"]["std"])
                double_values_1_mean.append(summary_stats_esm_event["double_values_1"]["mean"])
                double_values_1_std.append(summary_stats_esm_event["double_values_1"]["std"])
                double_values_2_mean.append(summary_stats_esm_event["double_values_2"]["mean"])
                double_values_2_std.append(summary_stats_esm_event["double_values_2"]["std"])
                double_values_3_mean.append(summary_stats_esm_event["double_values_3"]["mean"])
                double_values_3_std.append(summary_stats_esm_event["double_values_3"]["std"])




            print("Progress: " + str(esm_events.tolist().index(esm_event)) + "/" + str(len(esm_events)))

        else:
            missing_sensordata.append(esm_event)
            print("missing sensordata for esm event: " + str(esm_event))

    # create summary stats dataframe
    summary_stats["ESM_timestamp"] = ESM_timestamp
    summary_stats["device_id"] = device_id
    summary_stats["ESM_location"] = ESM_location
    summary_stats["ESM_location_time"] = ESM_location_time
    summary_stats["ESM_bodyposition"] = ESM_bodyposition
    summary_stats["ESM_bodyposition_time"] = ESM_bodyposition_time
    summary_stats["ESM_activity"] = ESM_activity
    summary_stats["ESM_activity_time"] = ESM_activity_time
    summary_stats["ESM_smartphonelocation"] = ESM_smartphonelocation
    summary_stats["ESM_smartphonelocation_time"] = ESM_smartphonelocation_time
    summary_stats["ESM_aligned"] = ESM_aligned
    summary_stats["ESM_aligned_time"] = ESM_aligned_time
    summary_stats["double_values_0_mean"] = double_values_0_mean
    summary_stats["double_values_0_std"] = double_values_0_std
    summary_stats["double_values_1_mean"] = double_values_1_mean
    summary_stats["double_values_1_std"] = double_values_1_std
    summary_stats["double_values_2_mean"] = double_values_2_mean
    summary_stats["double_values_2_std"] = double_values_2_std

    return summary_stats, missing_sensordata

path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_barometer_esm_timeperiod_5 min.csv"
test_df = pd.read_csv(path_sensor, dtype={"0": 'int64', "1": 'float64', "2": object, "3": object}, nrows=100)

dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/summary_stats"
sensor_list = ["locations", "rotation"]
sensor_list = ["linear_accelerometer", "gyroscope"]
#time_periods = [10,20,40]
time_periods = [5]
for sensor in sensor_list:
    for time_period in time_periods:
        #path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_" + sensor + "_esm_timeperiod_5 min.csv"
        path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/" + sensor + "_esm_timeperiod_5 min.csv"
        summary_statistics, missing_sensordata = create_summary_statistics_for_sensors(path_sensor, sensor, time_period, dir_results)
        summary_statistics.to_csv(dir_results + "/" + sensor + "_summary_stats_" + str(time_period) + ".csv", index=False)
        missing_sensordata_df = pd.DataFrame(missing_sensordata)
        missing_sensordata_df.to_csv(dir_results + "/" + sensor + "_summary_stats_missing_sensordata" + str(time_period) + ".csv", index=False)




# to check which columns to select: temporary
path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_magnetometer_esm_timeperiod_5 min.csv_JSONconverted.csv"
df = pd.read_csv(path_sensor, nrows=100)
summary = df.describe()
df["double_values_0"].value_counts()
df["double_values_1"].value_counts()
df["double_values_2"].value_counts()
df["double_values_3"].value_counts()
df["accuracy"].value_counts()

time_begin = time.time()
sensor = "accelerometer"
time_period = 10 # seconds
create_summary_statistics_for_sensors(path_sensor, sensor, time_period, dir_results)
time_end = time.time()
print("Time: " + str(time_end - time_begin))


#endregion

#region visualize summary statistics: mean (sensordata) x std (sensordata) x participant x activity for every sensor

def vis_summary_stats(path_summary_stats, sensor, time_period, activity_type, activity_name,  list_activities, dir_results):
    # load summary stats
    summary_stats = pd.read_csv(path_summary_stats)

    # create dataframe including only the relevant activities
    summary_stats = summary_stats[summary_stats[activity_type].isin(list_activities)]

    # rename device_id to participant
    summary_stats = summary_stats.rename(columns={"device_id": "participant"})
    # replace participant id with last three digits
    summary_stats["participant"] = summary_stats["participant"].str[-3:]


    # visualize four dimensional scatterplot with seaborn: mean x std x participant x activity
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))

    if activity_name == "gravity":
        # set x limits
        axs[0].set_xlim(-1, 1)
        axs[1].set_xlim(-1, 1)
        axs[2].set_xlim(-1, 1)
        # set y limits
        axs[0].set_ylim(0, 0.5)
        axs[1].set_ylim(0, 0.5)
        axs[2].set_ylim(0, 0.5)

    if activity_name == "gyroscope":
        # set x limits
        axs[0].set_xlim(-0.1, 0.1)
        axs[1].set_xlim(-0.1, 0.1)
        axs[2].set_xlim(-0.1, 0.1)
        # set y limits
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)
        axs[2].set_ylim(0, 1)

    if activity_name == "linear_accelerometer":
        # set x limits
        axs[0].set_xlim(-0.04, 0.04)
        axs[1].set_xlim(-0.04, 0.04)
        axs[2].set_xlim(-0.04, 0.04)
        # set y limits
        axs[0].set_ylim(0, 0.3)
        axs[1].set_ylim(0, 0.3)
        axs[2].set_ylim(0, 0.3)

    if activity_name == "accelerometer":
        # set x limits
        axs[0].set_xlim(-1, 1)
        axs[1].set_xlim(-1, 1)
        axs[2].set_xlim(-1, 1)
        # set y limits
        axs[0].set_ylim(0, 0.8)
        axs[1].set_ylim(0, 0.8)
        axs[2].set_ylim(0, 0.8)

    fig.suptitle(sensor + " " + activity_name + " " + str(time_period) + " seconds ", fontsize=36)
    sns.scatterplot(x="double_values_0_mean", y="double_values_0_std", hue=activity_type, style = "participant",data=summary_stats, ax=axs[0])
    sns.scatterplot(x="double_values_1_mean", y="double_values_1_std", hue=activity_type, style = "participant",data=summary_stats, ax=axs[1])
    sns.scatterplot(x="double_values_2_mean", y="double_values_2_std", hue=activity_type, style = "participant",data=summary_stats, ax=axs[2])

    #plt.show()

    return fig


# body positions
list_activities =[ "walking", "running", "cycling", "sitting: at a table", "standing", "lying"]
activity_type = "ESM_bodyposition"
activity_name = "bodyposition"
time_periods = [5, 10,20, 40] #seconds
sensors = ["accelerometer", "linear_accelerometer", "gyroscope", "gravity"]
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/visualizations/" + activity_name + "/"
for sensor in sensors:
    for time_period in time_periods:
        path_summary_stats = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/summary_stats/" + sensor + "_summary_stats_" + str(time_period) + ".csv"
        fig = vis_summary_stats(path_summary_stats, sensor, time_period, activity_type, activity_name, list_activities, dir_results)
        fig.savefig(dir_results + "/summary_stats/"+ activity_name + " _summarystats_" + sensor + "_" + str(time_period) + ".png")

# (public) transport
list_activities =["on the way: in public transport", "on the way: in train", "on the way: standing", "on the way: walking\\/cycling"]
activity_type = "ESM_location"
activity_name = "public transport"
time_periods = [5,10,20, 40] #seconds
sensors = ["accelerometer", "linear_accelerometer", "gyroscope", "gravity"]
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/visualizations/" + activity_name + "/"
for sensor in sensors:
    for time_period in time_periods:
        path_summary_stats = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/summary_stats/" + sensor + "_summary_stats_" + str(time_period) + ".csv"
        fig = vis_summary_stats(path_summary_stats, sensor, time_period, activity_type, activity_name, list_activities, dir_results)
        fig.savefig(dir_results + "/summary_stats/"+ activity_name + " _summarystats_" + sensor + "_" + str(time_period) + ".png")


# on the toilet
list_activities =["on the toilet", "working: at computer\\/laptop"]
activity_type = "ESM_activity"
activity_name = "on the toilet"
time_periods = [5, 10,20, 40] #seconds
sensors = ["accelerometer", "linear_accelerometer", "gyroscope", "gravity"]
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/visualizations/" + activity_name + "/"
for sensor in sensors:
    for time_period in time_periods:
        path_summary_stats = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/summary_stats/" + sensor + "_summary_stats_" + str(time_period) + ".csv"
        fig = vis_summary_stats(path_summary_stats, sensor, time_period, activity_type, activity_name, list_activities, dir_results)
        fig.savefig(dir_results + "summary_stats/" + activity_name + " _summarystats_" + sensor + "_" + str(time_period) + ".png")

#endregion

#region visualize time of events (ESM) x activity for every participant




path_esm_final = "/Volumes/INTENSO/In Usage new/Databases/esm_all_transformed.csv"
activity_type = "activity"
#list_activities = ["lying in bed after sleeping ", "lying in bed before sleeping "]
list_activities = ["lying in bed before sleeping", "lying in bed after sleeping"]
activity_description = "sleeping_times"
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/visualizations/sleeping_times"
def vis_time_of_events(path_esm_final, activity_type,  list_activities, activity_description, dir_results):
    # load esm final
    esm_final = pd.read_csv(path_esm_final)

    # create dataframe including only the relevant activities
    #esm_final = esm_final[esm_final[activity_type].str.contains("|".join(list_activities), na=False)]
    esm_final = esm_final[esm_final[activity_type].isin(list_activities)]

    # rename device_id to participant
    esm_final = esm_final.rename(columns={"device_id": "participant"})
    # replace participant id with last three digits
    esm_final["participant"] = esm_final["participant"].str[-3:]

    # create column containing the hour of the event
    esm_final["timestamp_hour"] = pd.to_datetime(esm_final['timestamp'], unit='ms').astype("string").str[11:16]
    #esm_final["timestamp_hour"]  = datetime.strptime(esm_final["timestamp_hour"][i], '%H:%M')
    esm_final = esm_final.reset_index(drop=True)
    esm_final["timestamp_hour_decimal"] = 0
    for i in range(0, len(esm_final)):
        esm_final["timestamp_hour_decimal"][i] = datetime.strptime(esm_final["timestamp_hour"][i], '%H:%M')
        esm_final["timestamp_hour_decimal"][i]  = (datetime.combine(date.min, esm_final["timestamp_hour_decimal"][i] .time()) - datetime.min).total_seconds() / 3600


    # visualize time of events x participant x activity
    import matplotlib.dates as mdates
    # create plot
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle("Time of events (ESM) x participant x " + activity_description, fontsize=18)
    sns.scatterplot(x="timestamp_hour_decimal", y="participant", style=activity_type, data=esm_final, ax=axs)
    axs.set_xlim(0, 24)
    axs.set_xticks(range(1, 24))
    plt.show()
    # save figure
    fig.savefig(dir_results + "/" + activity_description + "_time_of_events_" + ".png")
vis_time_of_events(path_esm_final, activity_type,  list_activities, activity_description, dir_results)
#endregion



# region get all different users & delete the ones, which I already know
esm_all_users = esm_all
for index_users, user in users.iterrows():
    esm_all_users = esm_all_users[esm_all_users["2"] != user["ID"]]
    print("Dropped user: " + str(user["Name"]))
esm_all_users["2"].unique()

# endregion


# region labels: join all aware_device files
dir_databases = "/Volumes/TOSHIBA EXT/InUsage/Databases"
sensor = "aware_device"
aware_device_all = join_sensor_files(dir_databases, sensor)

aware_device_all.to_csv("/Volumes/TOSHIBA EXT/InUsage/Databases/aware_device_all.csv")

#endregion
