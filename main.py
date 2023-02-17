#region import
import pickle
#import tensorflow_decision_forests as tfdf
import datetime
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time
import matplotlib.pyplot as plt
import json
import itertools
import string
import pytz

# computing distance
import geopy.distance

#Classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#evaluation
from sklearn.metrics import balanced_accuracy_score

#nested CV
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# import classification report
from sklearn.metrics import classification_report

#Feature Importance
import shap

#Statistical Testing
from scipy.stats import binom_test
from sklearn.model_selection import permutation_test_score

#Keras Tuner
import keras
import keras_tuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
from kerastuner.tuners import Hyperband

# visualization
import seaborn as sns
import matplotlib

# for keeping track
from tqdm import tqdm

#for GPS visualization in maps
import utm
from collections import Counter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# for clustering
import haversine
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

# for GPS feature creation
from datetime import timedelta
import pytz

try:
  from wurlitzer import sys_pipes
except:
  from colabtools.googlelog import CaptureLog as sys_pipes
#endregion

#region check versions
# #What version of Python do you have?
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

#endregion

# region general data transformation
#TODO merge ID´s if they are from the same participant for "xmin around events" and "merged timeseries" files
#TODO make sure that not same ESM_timestamps for different participants! CURRENT TASK

#region Laboratory Data
#merge sensor files for same participant (necessary for Benedikt´s data)
path_datasets = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Benedikt_old/"
sensor_list = ["accelerometer", "gyroscope", "magnetometer", "plugin_ios_activity_recognition",
               "rotation", "linear_accelerometer", "locations"]
#iterate through folders in path_datasets and merge sensor files for same participant
for sensor in sensor_list:
    print("start with sensor: " + sensor)
    counter = 1
    for folder in os.listdir(path_datasets):
        print("start with folder: " + folder)
        # check if folder ends with ".csv"
        if folder == ".DS_Store" or folder.endswith(".csv"):
            continue
        if counter == 1:
            df_sensor = pd.read_csv(path_datasets + folder + "/" + sensor + ".csv")
        if counter > 1:
            #concatenate dataframes
            df_sensor = pd.concat([df_sensor, pd.read_csv(path_datasets + folder + "/" + sensor + ".csv")])
        counter += 1

    #delete duplicates
    print("Number of duplicates before deletion: " + str(df_sensor.duplicated().sum()))
    df_sensor = df_sensor.drop_duplicates()
    print("Number of duplicates after deletion: " + str(df_sensor.duplicated().sum()))
    #save to csv
    df_sensor.to_csv("/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Benedikt_old/" + sensor + ".csv")

# add participant activity to sensordata (=label sensordata) & add ESM timestamp
length_of_esm_events = 90 # in seconds; this is the length of the ESM events in which the sensor data will be segmented
dir_dataset = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/"
dicts_esm_labels = dict()
for index, user in users_iteration02.iterrows():
    for index2, sensor in sensors_and_frequencies.iterrows():
        print("User: " + str(user["Name"]) + " - Sensor: " + str(sensor["Sensor"]))
        path_sensorfile = dir_dataset + user["Name"] + "/" + sensor["Sensor"] + ".csv"
        dict_label = pd.read_pickle(dir_dataset + user["Name"] + "/dict_label_iteration02_" + user["Name"] + ".pkl")
        if not os.path.exists(path_sensorfile): #check if sensorfile doesn't exist
            print("Sensorfile doesn't exist")
            continue

        # label the sensordata
        df_sensor = InitialTransforms_Iteration02.label_sensor_data(path_sensorfile, dict_label) #calculate labels
        if df_sensor is None or df_sensor.empty: #check if sensorfile is empty
            continue

        #analtics: compare number of unique labels with number of labels in dict_label
        df_sensor.to_csv(dir_dataset + user["Name"] + "/" + sensor["Sensor"]  + "_labeled.csv", index=False)

        #ad ESM_timestamp to sensordata (in order to make the laboratory data compatible with all functions for the naturalistic data)
        df_sensor, dict_esm_labels = InitialTransforms_Iteration02.add_esm_timestamps(df_sensor, dict_label, length_of_esm_events) #add ESM_timestamp
        #analytics: find how many ESM_timestamps are in the dataset and how many are included in dict_esm_labels
        print("Number of ESM_timestamps in dataset: " + str(len(df_sensor["ESM_timestamp"].unique())-1))
        print("Number of ESM_timestamps in dict_esm_labels: " + str(len(dict_esm_labels)))
        #print("Percentage of ESM_timestamps in dict_esm_labels: " + str(len(dict_esm_labels)/(len(df_sensor["ESM_timestamp"].unique())-1)*100))
        if int(sensor["Frequency (in Hz)"]) != 0:# analytics: find out how many labeled sensor minutes are in df_sensor and how many of those minutes are in ESM-events
            num_minutes = len(df_sensor["timestamp"]) / sensor["Frequency (in Hz)"] / 60
            print("Number of labeled sensor minutes for user " + user["Name"] + ": " + str(num_minutes) + " minutes")
            esm_events_total_length = (len(df_sensor["ESM_timestamp"].unique()) * length_of_esm_events)/60
            print("Percentage of labeled sensor minutes that are in ESM events: " + str(esm_events_total_length/num_minutes * 100) + "%")
        df_sensor.to_csv(dir_dataset + user["Name"] + "/" + sensor["Sensor"]  + "_labeled_esm_timestamps.csv", index=False)

        #merge dict_esm_labels to dicts_esm_labels
        num_ESM_timestamps_temporary = len(dicts_esm_labels)
        dicts_esm_labels.update(dict_esm_labels)
        print("Number of ESM_timestamps in this sensor file: " + str(len(dict_esm_labels)))
        print("The number of ESM_timestamps increased by: " +  str(len(dicts_esm_labels) - num_ESM_timestamps_temporary) + " events")
        print("Lenght of dicts_esm_labels after completing user" + user["Name"]  + " and sensor " + sensor["Sensor"] + ":" + str(len(dicts_esm_labels)))
        print("Sensorfile labeled and ESM_timestamps added for user " + user["Name"] + " and sensor " + sensor["Sensor"])
with open(dir_dataset + "dicts_esmtimestamp-label-mapping.pkl", "wb") as f:
    pickle.dump(dicts_esm_labels, f)

# merge sensorfiles of different users
path_datasets = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/"
for index, sensor in sensors_and_frequencies.iterrows():
    print("start with sensor: " + sensor["Sensor"])
    df_sensor = Merge_Transform.join_sensor_files(path_datasets, sensor["Sensor"], sensor_appendix = "_labeled_esm_timestamps")
    if df_sensor is None or df_sensor.empty:
        continue
    df_sensor.to_csv(path_datasets + sensor["Sensor"] + "_labeled_esm_timestamps_allparticipants.csv", index=False)

#endregion

#endregion

#region label transformation

#region label transformation for naturalistic data
# create csv and dictionary which maps users answers to activity classes
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed.csv")
## create .csv
df_esm_including_activity_classes = label_transformation.create_activity_dataframe(df_esm, human_motion, humanmotion_general, humanmotion_specific, before_after_sleep, before_after_sleep_updated,
                                                        on_the_toilet_sittingsomewhereelse, on_the_toilet, publictransport_non_motorized, publictransport,
                                                        location, smartphonelocation, aligned)

## analytics: compare computed activity classes with user answers
df_esm_including_activity_classes["label_location"].value_counts()

df_esm_including_activity_classes.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")

## create dictionary
dict_esm_including_activity_classes = label_transformation.create_activity_dictionary_from_dataframe(df_esm_including_activity_classes)

with open(
        "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl",
        'wb') as f:
    pickle.dump(dict_esm_including_activity_classes, f, pickle.HIGHEST_PROTOCOL)
#endregion

#region label transformation for laboratory data
# create df_esm csv and dictionary in which also the activity classes are included
# (based on the mappings created in "databases_iteration02.py")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/labels/"
dict_esm_labels_iteration02 = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration02/datasets/dicts_esmtimestamp-label-mapping.pkl")
df_esm_labels_iteration02 = pd.DataFrame.from_dict(dict_esm_labels_iteration02, orient='index', columns=["user activity", "device_id"])
df_esm_labels_iteration02["ESM_timestamp"] = df_esm_labels_iteration02.index
label_column_names = ["label_human motion", "label_on the toilet"]
for label_column_name in label_column_names:
    if label_column_name == "label_human motion":
        dict_mapping = human_motion  # is created in the script "databases_iteration02.py"
    elif label_column_name == "label_on the toilet":
        dict_mapping = on_the_toilet
    df_esm_including_activity_classes = label_transformation.add_activity_classes(df_esm_labels_iteration02, dict_mapping, label_column_name)

    ## analytics: compare computed activity classes with user answers
    df_esm_including_activity_classes[label_column_name].value_counts()
    ## create dictionary
    dict_esm_including_activity_classes = label_transformation.create_activity_dictionary_from_dataframe(df_esm_including_activity_classes)
    # save
    df_esm_including_activity_classes.to_csv(path_storage + "esm_transformed_including-activity-classes.csv")
    with open(path_storage + "esm_transformed_including-activity-classes_dict.pkl", 'wb') as f:
        pickle.dump(dict_esm_including_activity_classes, f, pickle.HIGHEST_PROTOCOL)

#endregion

#endregion

# region sensor transformation
#region merge all screen- and wifi-sensor data
# Note: this is needed for adding the beginning and end of sessions to all ESM events
dir_databases = "/Volumes/INTENSO/In Usage new/Databases/"
sensors = ["screen", "sensor_wifi"]
sensors = ["sensor_wifi"]
for sensor in sensors:
    df_sensors_all = Merge_Transform.join_sensor_files(dir_databases, sensor, sensor_appendix = None)
    df_sensors_all = Merge_Transform.convert_json_to_columns(df_sensors_all, sensor)
    df_sensors_all.to_csv(dir_databases + sensor +"_all.csv")
#endregion

#region merge all linear accelerometer and rotation data: per participant (since otherwise too much data)
# Note: this is needed for activity "before and after sleep" for computing static periods during nights
dir_databases = "/Volumes/INTENSO/In Usage new/Databases/"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/"
sensors = ["linear_accelerometer", "rotation"]
sensors = ["linear_accelerometer"]
for sensor in sensors:
    print("start " + sensor)
    # iterate through users
    for user_id in users["ID"]:
        print("start with user " + str(user_id))
        # check if data for user has already been merged
        if os.path.exists(path_storage + sensor + "/" + sensor + "_device-id-" + str(user_id) + "_all.pkl"):
            print("data for user " + str(user_id) + " already merged")
            continue
        df_sensors_all = Merge_Transform.join_sensor_files(dir_databases, sensor, participant_id = user_id, sensor_appendix = None)
        df_sensors_all = Merge_Transform.convert_json_to_columns(df_sensors_all, sensor)
        # save with pickle
        with open(path_storage + sensor +  "/" + sensor +"_device-id-" + str(user_id) + "_all.pkl", 'wb') as f:
            pickle.dump(df_sensors_all, f, pickle.HIGHEST_PROTOCOL)
#endregion

#region add to each ESM-event the beginning and end of the smartphone session
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
df_screen = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/screen_all.csv")
## convert all columns in which "time" is contained to datetime
for col in df_esm.columns:
    if "time" in col:
        df_esm[col] = pd.to_datetime(df_esm[col], unit="ms")
## add beginning and end of smartphone session to each ESM-event
df_esm_with_screen = Merge_Transform.add_smartphone_session_start_end_to_esm(df_esm, df_screen)
#endregion

#region add duration between beginning and end of ESM answer session to each ESM-event
## compute duration between first and last answer
df_esm_with_screen["duration_between_first_and_last_answer (s)"] = (df_esm_with_screen["smartphonelocation_time"] - df_esm_with_screen[
    "location_time"]) / np.timedelta64(1, 's')
## drop rows for which the time it took between first and last ESM answer is above 5 minutes
df_esm_with_smartphone_sessions = df_esm_with_screen[df_esm_with_screen["duration_between_first_and_last_answer (s)"] < 5*60]
df_esm_with_smartphone_sessions.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_with_smartphone_sessions.csv")

# visualize the duration of answer sessions and duration of answer sessions in plt barplots
df_esm_with_smartphone_sessions = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_with_smartphone_sessions.csv")
## visualize duration of smartphone sessions in sns barplot
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.barplot(x="label_human motion", y="duration_between_first_and_last_answer (s)", data=df_esm_with_smartphone_sessions)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
#endregion

#region visualize duration of smartphone sessions in barplots
## drop all values of smartphone_sessions that are above 2 hours (= 7200 seconds)
df_esm_with_smartphone_sessions = df_esm_with_smartphone_sessions[df_esm_with_smartphone_sessions["smartphone_session_duration (s)"] < 7200]

## generally
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.hist(df_esm_with_smartphone_sessions["smartphone_session_duration (s)"], bins=1000)
ax.set_xlabel("duration of smartphone session (s)")
ax.set_ylabel("number of smartphone sessions")
ax.set_title("duration of smartphone sessions")
# set xlim to 300 seconds
ax.set_xlim(0, 1200)
plt.show()

## for different activity classes
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.barplot(x="label_human motion", y="smartphone_session_duration (s)", data=df_esm_with_smartphone_sessions)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
#endregion



# Rules (NOT DOCUMENTED YET):
## ESM_timestamp is timestamp when "Send" button for first answer is clicked -> IMPLEMENT!!
## events are discarded if more than 5 minutes between first and last answer (location and smartphone location) (reduces number of events from 1113 to 1065)
## NOTE: there are events for which no smartphone session start or no end could be found


#region delete all sensor data which is not inside the "smartphone session" of each ESM event & drop also duplicates
## save it as an extra file
sensor_list = ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "locations", "sensor_wifi"]
sensor_list = ["sensor_wifi"]

dir_sensors = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/"
df_esm_with_smartphone_sessions = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_with_smartphone_sessions.csv")
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/"

for sensor in tqdm(sensor_list):
    print("Start with sensor: " + sensor)
    # set frequencies
    # get from sensors_and_frequencies
    frequency = int(sensors_and_frequencies[sensors_and_frequencies["Sensor"] == sensor]["Frequency (in Hz)"])
    # load sensor data
    df_sensor = pd.read_pickle(dir_sensors + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.pkl")
    # drop duplicates
    num_before_removing_duplicates = len(df_sensor) #analytics purposes
    columns_duplicate_relevant = df_sensor.columns
    columns_duplicate_relevant = columns_duplicate_relevant.drop(["Unnamed: 0", "0"])
    df_sensor = df_sensor.drop_duplicates(subset=columns_duplicate_relevant)
    print("Percentage of duplicates: " + str((num_before_removing_duplicates - len(df_sensor)) / num_before_removing_duplicates))
    # delete all sensor data which is not inside the "smartphone session" of each ESM event
    df_sensor = Merge_Transform.delete_sensor_data_outside_smartphone_session(df_sensor, df_esm_with_smartphone_sessions, frequency)
    # save it as pickle
    with open(dir_results + sensor + "_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl", 'wb') as f:
        pickle.dump(df_sensor, f, pickle.HIGHEST_PROTOCOL)
#endregion

#region TEMPORARY join all locations sensorfiles -> just double checking if my 5-min-around-event Location
# file contains everything!
dir_databases = "/Volumes/INTENSO/In Usage new/Databases/"
sensor = "locations"
df_sensors_all = Merge_Transform.join_sensor_files(dir_databases, sensor, sensor_appendix = None)
df_sensors_all.to_csv(dir_databases + "locations_all.csv")
# save as pickle
with open(dir_databases + "locations_all.pkl", 'wb') as f:
    pickle.dump(df_sensors_all, f, pickle.HIGHEST_PROTOCOL)

##analytics: find out how many events have sensordata in 90 second segments around event time
# get dataframe with all ESM timestamps
df_sensors_all = pd.read_csv(dir_databases + "locations_all.csv")
print(df_sensors_all["1"].head()) #analytics
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
# create ESM_timestamp column which is in datetime format from timestamp column
df_esm["ESM_timestamp"] = pd.to_datetime(df_esm["timestamp"], unit = "ms")
df_esm = df_esm[["ESM_timestamp", "device_id"]]

time_period = 90
df_sensors_all["timestamp"] = pd.to_datetime(df_sensors_all["1"], unit = "ms")
# iterate through ESM_timestamps in df_esm
df_results = pd.DataFrame()
for index, row in df_esm.iterrows():
    # find data in time_period around ESM_timestamp in df_sensors_all
    event_time = row["ESM_timestamp"]
    device_id = row["device_id"]

    df_temporary = df_sensors_all[(df_sensors_all['timestamp'] >= event_time - pd.Timedelta(seconds=(time_period / 2))) & (
            df_sensors_all['timestamp'] <= event_time + pd.Timedelta(seconds=(time_period / 2))) & (df_sensors_all["2"] == device_id)]
    df_temporary["ESM_timestamp"] = row["ESM_timestamp"]

    #concatenate df_results with df_temporary
    df_results = pd.concat([df_results, df_temporary])

# merge participant IDs
#df_results = merge_participantIDs(df_results, users, device_id_col = "2", include_cities = False)
print("Unique ESM_timestamp values after cutting to 90 seconds around events for locations: " + str(df_results["ESM_timestamp"].nunique()))
#double check
df_test = df_results[df_results["ESM_timestamp"] == df_results["ESM_timestamp"].iloc[0]]
# find out how many records for every unique ESM_timestamp
test= df_results.groupby("ESM_timestamp")["ESM_timestamp"].count()

#endregion


#endregion

#region general data exploration
#region Naturalistic Data
# visualize label-data distribution
path_esm = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv"
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/labels/"
threshold = 5 #only events which have at least this number of occurences are visualized
df_esm = data_exploration_labels.load_esm(path_esm)
## visualize for every answer and every activity the number of classes & count analytics
df_analytics = data_exploration_labels.visualize_esm_activities_all(dir_results,df_esm, threshold)
df_analytics.to_csv(dir_results + "analytics.csv")
    ## visualize in one plot the number of labels per activity (bar plot)
data_exploration_labels.visualize_esm_notNaN( df_esm)

# calculate number of sensor-datapoints for each event
# find out how many labels have sensor data -> ##temporary -> make later into function!
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
gps_accuracy_min = 100
time_period = 90 #in seconds; timeperiod around events which will be included
only_active_smartphone_sessions = "yes"
sensors = ["linear_accelerometer", "gyroscope", "magnetometer", "barometer", "rotation", "locations", "sensor_wifi"]
sensors = ["accelerometer", "linear_accelerometer", "gyroscope", "magnetometer", "rotation", "locations", "sensor_wifi"]
path_sensor_database = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/"
df_esm = data_exploration_sensors.volume_sensor_data_for_events(df_esm, time_period, sensors, path_sensor_database, gps_accuracy_min, only_active_smartphone_sessions)
df_esm.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/esm-events_with-number-of-sensorrecords_segment-around-events-" + str(time_period) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_gps-accuracy-min-" + str(gps_accuracy_min) + " .csv")

# calculate mean/std & max of GPS and linear accelerometer data (over different time periods)
## Note: this data is relevant for, for example, public transport -> data exploration -> visualizing mean & std of GPS data
## calculate summary stats for GPS
dir_sensors = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/"
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/"
sensors = ["GPS", "linear_accelerometer", "rotation", "magnetometer", "gyroscope"]
only_sensordata_of_active_smartphone_sessions = "yes"
gps_accuracy_min_list = [10]
#time_periods = [10,20,40, 90, 180]
time_periods = [10,20,40, 90, 180]

for sensor in sensors:
    for time_period in time_periods:
        for gps_accuracy_min in gps_accuracy_min_list:
            if sensor == "GPS":
                if only_sensordata_of_active_smartphone_sessions == "yes":
                    path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(gps_accuracy_min) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + ".csv"
                else:
                    path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(gps_accuracy_min) + ".csv"
            else:
                if only_sensordata_of_active_smartphone_sessions == "yes":
                    path_sensor = dir_sensors + sensor + "_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl"
                else:
                    path_sensor = dir_sensors + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.pkl"

            print("Started with time period: " + str(time_period))
            summary_statistics, missing_sensordata = data_exploration_sensors.create_summary_statistics_for_sensors(path_sensor, sensor, time_period, dir_results)
            missing_sensordata_df = pd.DataFrame(missing_sensordata)
            if sensor == "GPS":
                summary_statistics.to_csv(dir_results + sensor + "_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + ".csv",
                                          index=False)
                missing_sensordata_df.to_csv(dir_results + sensor + "_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + "_missing_sensordata.csv", index=False)
            else:
                summary_statistics.to_csv(dir_results + sensor + "_summary-stats_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + ".csv", index=False)
                missing_sensordata_df.to_csv(dir_results + sensor + "_summary-stats_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + "_missing_sensordata.csv", index=False)
#endregion

#region Laboratory Data
# calculate mean/std & max of GPS and linear accelerometer data (over different time periods)
## Note: this data is relevant for, for example, public transport -> data exploration -> visualizing mean & std of GPS data
## calculate summary stats for GPS
dir_sensors = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/"
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration02/data_exploration/sensors/summary_stats/"
sensors = ["linear_accelerometer", "rotation"]
gps_accuracy_min_list = [10]
label_segments = [10, 90]

for sensor in sensors:
    for label_segment in label_segments:
        for gps_accuracy_min in gps_accuracy_min_list:
            if sensor == "GPS":
                path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(gps_accuracy_min) + ".csv"
            else:
                path_sensor = dir_sensors + sensor + "_labeled_esm_timestamps_allparticipants.csv"

            print("Started with time period: " + str(label_segment))

            summary_statistics, missing_sensordata = data_exploration_sensors.create_summary_statistics_for_sensors(path_sensor, sensor, label_segment, dir_results)
            missing_sensordata_df = pd.DataFrame(missing_sensordata)
            if sensor == "GPS":
                summary_statistics.to_csv(dir_results + sensor + "_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + ".csv",
                                          index=False)
                missing_sensordata_df.to_csv(dir_results + sensor + "_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + "_missing_sensordata.csv", index=False)
            else:
                summary_statistics.to_csv(dir_results + sensor + "_summary-stats_time-period-around-event-" + str(label_segment) + ".csv", index=False)
                missing_sensordata_df.to_csv(dir_results + sensor + "_summary-stats_time-period-around-event-" + str(label_segment) + "_missing_sensordata.csv", index=False)
#endregion



#endregion

#region different activities

#region human motion
#TODO add noise removal & normalization to data preparation. Here are some ideas from the HAR review paper: Concerning the Noise Removal step, 48 CML-based articles and 12 DL-based articles make use of different noise removal techniques. Among all such techniques the most used ones are: z-normalization [75], [120], min-max [70], [127], and linear interpolation [102], [111] are the most used normalization steps, preceded by a filtering step based on the application of outlier detection [70], [117], [163], Butterworth [82], [101], [117], [123], [127], [128], [152], [155], [174], [189], median [74], [101], [117], [127], [132], [147], [155], [182], [183], high-pass [92], [96], [117], [128], [169], [173], [208], or statistical [58] filters.
#TODO create other features (using tsfresh). Here are the ones most used in the HAR literature (compare review): compare my documentation!

#region Laboratory Data

#region data exploration for human motion
#region data exploration labels
label_column_name = "label_human motion"
df_labels = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/labels/esm_transformed_including-activity-classes.csv")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/data_exploration/labels/"
esm_segment_length = 90 #in seconds; the length of the ESM_events which were created for the laboratory data in order to be able to use the functions from naturalistic data

## create barplot: human motion classes x total minutes
## delete some of the human motion classes (reason: they are only present in one participant and therefore not enough data for training LOSOCV models
classes_to_drop = ["lying (on flat surface)", "sitting: at a table (on flat surface)", "sitting: on the couch (in hand/s)"]
df_labels = df_labels[~df_labels[label_column_name].isin(classes_to_drop)]

df_labels_humanmotion= df_labels.copy()
fig = data_exploration_labels.visualize_esm_activity_minutes(df_labels_humanmotion, "label_human motion", esm_segment_length, "Volume of Human Motion Data")
fig.savefig(path_storage + "human_motion_class-minutes.png", dpi=300)

##create table: users x label classes x minutes
df_labels_humanmotion =  Merge_Transform.merge_participantIDs(df_labels_humanmotion, users_iteration02, include_cities = True)
df_label_counts = data_exploration_labels.create_table_user_classes_minutes(df_labels_humanmotion, "label_human motion", esm_segment_length)
df_label_counts.to_csv(path_storage + "human_motion_labels_users_classes_minutes.csv")

#endregion

#region data exploration sensors
# summary stats: visualize mean/std x max x human motion classes in scatterplot for GPS and accelerometer data
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/labels/esm_transformed_including-activity-classes_dict.pkl")
time_periods = [10, 90]
list_activities = ["standing (in hand/s)", "sitting: at a table (in hand/s)", "lying (in hand/s)"] #stationary classes
#list_activities = ["walking (in hand/s)", "running (in hand/s)", "cycling (in hand/s)"] # dynamic classes
list_activities = ["walking (in hand/s)", "running (in hand/s)", "cycling (in hand/s)",
                   "standing (in hand/s)", "sitting: at a table (in hand/s)", "lying (in hand/s)"] # all classes

visualize_participants = "no"
gps_accuracy_min = 10 # minimum accuracy of GPS data to be included in analysis
sensors = [["Linear Accelerometer", "linear_accelerometer"],
           ["Rotation", "rotation"],
           ["Gyroscope", "gyroscope"],
              ["Magnetometer", "magnetometer"]]
only_sensordata_of_active_smartphone_sessions = "yes"
label_column_name = ["Human Motion Classes", "label_human motion"]
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_exploration/sensors/SummaryStats_Visualizations/"

for sensor in sensors:
    for time_period in time_periods:
        if sensor[0] == "GPS":
            df_summary_stats = pd.read_csv(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/GPS_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + ".csv")
            # add label
            df_summary_stats = labeling_sensor_df(df_summary_stats, dict_label, label_column_name[1],
                                                  ESM_identifier_column="ESM_timestamp")

            # change label_column_name in df_summary_stats
            df_summary_stats = df_summary_stats.rename(columns={label_column_name[1]: label_column_name[0]})

            fig = data_exploration_sensors.vis_summary_stats(df_summary_stats, sensor[1] , label_column_name[0], list_activities,
                                    "Mean and Maximum Speed and Acceleration of Human Motion Events", visualize_participants)
            fig.savefig(path_storage + "SummaryStatsVisualization_GPS_mean_max_gps-accuracy-min-" + str(gps_accuracy_min) + "_activities-included-" + str(list_activities) + "_including-participants-" + visualize_participants + "_time-period-around-event-"
                        + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + "s_scatterplot.png", dpi=300)

        else:
            df_summary_stats = pd.read_csv(
                "/Users/benediktjordan/Documents/MTS/Iteration02/data_exploration/sensors/summary_stats/" + sensor[1] + "_summary-stats_time-period-around-event-" + str(time_period) + ".csv")
            # add label
            df_summary_stats = labeling_sensor_df(df_summary_stats, dict_label, label_column_name[1], ESM_identifier_column="ESM_timestamp")

            # change label_column_name in df_summary_stats
            df_summary_stats = df_summary_stats.rename(columns={label_column_name[1]: label_column_name[0]})

            figure_title = "Mean and Maximum of the " + sensor[0] + " of Human Motion Classes"
            fig = data_exploration_sensors.vis_summary_stats(df_summary_stats, sensor[1], label_column_name[0], list_activities, figure_title , visualize_participants)

            list_activities_forsaving = [x.replace("/", "_") for x in list_activities]
            fig.savefig(
                path_storage + "SummaryStatsVisualization_" + sensor[1] + "_std-max_including-participants-" + visualize_participants + "_time-period-around-event-"
                + str(time_period) + "s_scatterplot.png", dpi=300)




#testarea for location data
df_locations = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration.csv")
#label data
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
df_locations = labeling_sensor_df(df_locations, dict_label, "label_public transport", ESM_identifier_column="ESM_timestamp")
# delete rows with nan values in label_public transport
df_locations = df_locations.dropna(subset=["label_public transport"])

#calculate summary stats for column accuracy
df_locations["loc_accuracy"].describe()

# only retain rows with accuracy < 35
df_locations_new = df_locations[df_locations["loc_accuracy"] < 35]
#count number of unique ESM_timestamps
df_locations["ESM_timestamp"].nunique()
df_locations_new["ESM_timestamp"].nunique()

#retain only data which is in 90s time period around ESM_timestamp
time_period = 90
#convert timestamp to datetime
df_locations_new["ESM_timestamp"] = pd.to_datetime(df_locations_new["ESM_timestamp"])
df_locations_new["timestamp"] = pd.to_datetime(df_locations_new["timestamp"])
df_locations_new = df_locations_new[(df_locations_new['timestamp'] >= df_locations_new['ESM_timestamp'] - pd.Timedelta(seconds=(time_period / 2))) & (
        df_locations_new['timestamp'] <= df_locations_new['ESM_timestamp'] + pd.Timedelta(seconds=(time_period / 2)))]

df_locations_new["ESM_timestamp"].nunique()
# take only the unique ESM_timestamps and count from them the number of label_public transport
df_locations_new_unique_ESM_timestamps = df_locations_new.drop_duplicates(subset=["ESM_timestamp"])
df_locations_new_unique_ESM_timestamps["label_public transport"].value_counts()





# add labels
df_locations = labeling_sensor_df(df_locations, dict_label, "label_public transport", ESM_identifier_column="ESM_timestamp")
#delete NaN values in label_public transport
df_locations = df_locations[df_locations["label_public transport"].notna()]
#get only data from user with loc_device_id = 590f0faf-d932-4a57-998d-e3da667a91dc
df_locations_user = df_locations[df_locations["loc_device_id"] == "b7b013b7-f78c-4325-a7ab-2dfc128fba27"]

#endregion

#endregion

#region data preparation for human motion

#region merge different sensors linear accelerometer & rotation
## NOTE: in difference to "Naturalistic Data", accelerometer will not be included as it has a lot of missing values!
sensor_sets_to_merge = [["linear_accelerometer", "gyroscope", "magnetometer", "rotation"],
                        ["linear_accelerometer", "rotation"]]
timedelta = "100ms" #must be in format compatible with pandas.Timedelta
columns_to_delete = ["device_id", "ESM_timestamp", "user activity", "Unnamed: 0"]
label_column_name = "label_human motion"
path_datasets = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/"
path_intermediate_files = None
add_prefix_to_merged_columns = True
for sensor_set in sensor_sets_to_merge:
    counter = 1
    for sensor in sensor_set:
        print("start with sensor: " + sensor)
        if counter == 1:
            sensor_base = sensor
            df_base = pd.read_csv(path_datasets + sensor + "_labeled_esm_timestamps_allparticipants.csv")
            df_base = df_base.drop(columns=["timestamp", "user activity", "Unnamed: 0"]) #delete "timestamp" column and rename "timestamp_datetime" to "timestamp"
            df_base = df_base.rename(columns={"timestamp_datetime": "timestamp"})
            for col in df_base.columns: #rename columns of df_base so that they can be still identified later on
            if col != "timestamp" and col != "device_id" and col != "timestamp_merged" and col != "ESM_timestamp" and col != label_column_name:
                df_base = df_base.rename(columns={col: sensor_base[:3] + "_" + col})
            counter += 1
            continue
        if counter > 1:
            sensor_tomerge = sensor
            df_tomerge = pd.read_csv(path_datasets + sensor + "_labeled_esm_timestamps_allparticipants.csv")
            df_tomerge = df_tomerge.drop(columns=["timestamp"]) #delete "timestamp" column and rename "timestamp_datetime" to "timestamp"
            df_tomerge = df_tomerge.rename(columns={"timestamp_datetime": "timestamp"})
            df_base = Merge_and_Impute.merge(df_base, df_tomerge, sensor_tomerge, timedelta, columns_to_delete, path_intermediate_files, add_prefix_to_merged_columns)
            counter += 1
    #save to csv
    df_base.to_csv("/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/timeseries_merged/" + str(sensor_set) + "_allparticipants.csv", index=False)

    #endregion
#endregion

#region calculate distance, speed & acceleration
accuracy_thresholds = [10, 35, 50, 100] #accuracy is measured in meters
path_datasets = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/"

#load GPS data
df_locations_events = pd.read_csv(path_datasets + "locations_labeled_esm_timestamps_allparticipants.csv")
df_locations_events = df_locations_events.drop(columns=["Unnamed: 0", "timestamp_unix", "label_human motion - general"])
# rename "timestamp" to "timestamp_unix" and "timestamp_datetime" to "timestamp"
df_locations_events = df_locations_events.rename(columns={"timestamp": "timestamp_unix"})
df_locations_events = df_locations_events.rename(columns={"timestamp_datetime": "timestamp"})

# calculate distance, speed and acceleration (class build in "FeatureExtraction_GPS.py")
df_features = FeatureExtraction_GPS().calculate_distance_speed_acceleration(df_locations_events)

# drop rows where distance, speed or acceleration contain NaN (they contain NaN if it is first entry of every event)
df_features= df_features.dropna(subset=["distance (m)", "speed (km/h)", "acceleration (km/h/s)"])
# drop rows with unrealistic speed values  & GPS accuracy values
df_features = df_features[df_features["speed (km/h)"] < 300]
# create different GPS accuracy thresholds
for accuracy_threshold in accuracy_thresholds:
    df_features_final = df_features[df_features["accuracy"] < accuracy_threshold]
    df_features_final = df_features_final.reset_index(drop=True)
    print("Number of rows with accuracy < " + str(accuracy_threshold) + "m: " + str(len(df_features_final)))
    #save to csv
    df_features_final.to_csv(path_datasets + "locations_features-distance-speed-acceleration_accuracy-less-than" + str(accuracy_threshold) + ".csv")
#endregion

#region feature extraction of high-frequency features: for both sensor subsets
## COMPARE CODE IN OTHER PROJECT
### what is missing:
### - feature extraction for all highfreq: done
### - combining chunks: for LinAcc-Rot set: done
#endregion

#region feature selection of high frequency sensors: data driven approach
##NOTE: the following code regarding feature extraction and selection with tsfresh is copied from "pythonProject", as the
## code can only be deployed there (due to requirements of tsfresh); therefore this code needs to be updated regularly
#endregion

#region NOT CURRENTLY USED feature selection of high frequency sensors and GPS data: hypothesis driven approach

#testcase
df_test = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/features/highfrequency_sensors/highfrequency-sensors_timeperiod-30_tsfresh-features-extracted.pkl")
all_features = df_test.columns
# get list of features in all_features which contain "fft"
fft_features = [feature for feature in all_features if "fft" in feature]
lin_fft_features = [feature for feature in fft_features if "lin_double_values" in feature]
#convert lin_fft_features to dataframe
df_lin_fft_features = df_test[lin_fft_features]

#endregion

#endregion

#region modeling for human motion

##region Decision Forest for comparing different feature segments
# training DF
feature_segments = [30,10, 5,2,1] #in seconds; define length of segments of parameters (over which duration they have been created)
combinations_sensors = [["linear_accelerometer", "rotation"],
                        ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]
                        ]  # define which sensor combinations should be considered
combinations_sensors = [ ["linear_accelerometer", "gyroscope", "magnetometer", "rotation"]]  # define which sensor combinations should be considered
min_gps_accuracy = 35
label_column_name = "label_human motion"
n_permutations = 0 # define number of permutations; better 1000
label_segment = 90 #define how much data around each event will be considered
# if label classes should be joint -> define in label mapping
label_classes = ["lying (in hand/s)", "sitting: at a table (in hand/s)", "standing (in hand/s)",
                 "walking (in hand/s)", "running (in hand/s)", "cycling (in hand/s)" ] # which label classes should be considered
parameter_tuning = "no" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = [] # columns that should be dropped
feature_importance = "shap"
label_mapping = None

parameter_set = {
    "n_estimators": 100, #default
    "criterion": "gini", #default
    "max_depth": None, #default
    "min_samples_split": 2,#default
    "min_samples_leaf": 1,#default
    "min_weight_fraction_leaf": 0.,#default
    "max_features": None, # THIS IS CHANGED
    "max_leaf_nodes": None,#default
    "min_impurity_decrease": 0.0,#default
    "bootstrap": True,#default
    "oob_score": False,#default
    "n_jobs": None,#default
    "verbose": 0,#default
    "warm_start": False,#default
    "class_weight": "balanced", # THIS IS CHANGED
    "n_jobs": -1,#default
    "random_state": 11#default
}
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Feature Segments"])
df_analytics = pd.DataFrame(columns=["Sensor Combination", "Feature Segment", "Event Description", "Number Participants", "Number Events", "Number Samples", "Number Features"])
for combination_sensors in combinations_sensors:

    for feature_segment in feature_segments:
        # create path to data based on sensor combination and feature segment
        path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_preparation/features/label_human motion_" + str(
            combination_sensors) + "_timeperiod-" + str(feature_segment) + "_FeaturesExtracted_Selected.pkl"
        t0 = time.time()
        print("start of combination_sensors: " + str(combination_sensors) + " and parameter_segment: " + str(feature_segment))
        path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/"+ str(combination_sensors)+ "_feature-segment-" +str(feature_segment) + "s/"
        if not os.path.exists(path_storage):
            os.makedirs(path_storage)
        with open(path_dataset, "rb") as f:
            df = pickle.load(f)
        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Dataset loaded",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df = df.drop(columns=drop_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

        #drop NaN in device_id:
        #TODO: probably error that there are still NaN in device_id -> probably from Merging step -> resolve if time
        df = df.dropna(subset=["device_id"])
        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in device_id dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #drop NaN in label_column_name
        df = df.dropna(subset=[label_column_name])
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in label_column_name dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)
        df = Merge_Transform.merge_participantIDs(df, users_iteration02)  # temporary: merge participant ids

        # drop rows with NaN: will be many, as after merging of high-freq with GPS, NaN have not be dropped yet
        df = df.dropna()
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in any column dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #combine label classes if necessary
        if label_mapping != None:
            df = df.reset_index(drop=True)
            for mapping in label_mapping:
                df.loc[df["label_human motion"] == mapping[0], "label_human motion"] = mapping[1]
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Label Classes which are not included are dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        # selects only features which are calculated in the label_segment around event.
        ## Note: updated to code so that it actually only selects features for which the WHOLE FEATURE SEGMENT is in the label_segment around event
        df_decisionforest = df[(df['timestamp'] >= (df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2)))) & (df['timestamp'] <= (df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2))- pd.Timedelta(seconds=feature_segment)))]
        df_decisionforest = df_decisionforest.reset_index(drop=True)
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "data outside of label segment around events dropped",
                                            "Number Participants": len(df_decisionforest["device_id"].unique()),
                                            "Number Events": len(df_decisionforest["ESM_timestamp"].unique()),
                                            "Number Samples": len(df_decisionforest),
                                            "Number Features": len(df_decisionforest.columns) - 1}, ignore_index=True)

        print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")

        # check if df_decisionforest is empty
        if df_decisionforest.empty:
            print("df_decisionforest is empty")
            continue

        #run DF
        df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, n_permutations, path_storage, feature_importance = feature_importance, confusion_matrix_order = label_classes,  parameter_tuning = parameter_tuning, parameter_set = parameter_set)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Feature Segments"] = str(feature_segment)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        print("Finished Combination: " + str(combination_sensors) + "and timeperiod: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")

df_analytics.to_csv("/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "s_parameter_tuning-" + parameter_tuning + "_analytics.csv")
df_decisionforest_results_all.to_csv("/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")


#visualizing different ML results
label_segment = 90
parameter_tuning = "no"
df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall_final.csv")
# add dataset description depending on value in "Sensor Combination" and "Feature Segments"
df_results["Feature Set and Feature Segment"] = ""
for index, row in df_results.iterrows():
    if row["Sensor Combination"] == "['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation']":
        if row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Segment"] = "30s"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Segment"] = "10s"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Segment"] = "5s"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Segment"] = "2s"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Segment"] = "1s"
# visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(10, 5))
#visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot; use pandas.melt
sns.lineplot(x="Feature Segment", y="value", hue="variable", data=pd.melt(df_results, id_vars=["Feature Segment"], value_vars=["Balanced Accuracy", "F1"]))
plt.title("Model Performance of Different Feature Segments")
plt.xlabel("Feature Segments")
plt.ylabel("Score")
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#set y-axis limits
plt.ylim(0, 1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.show()
#save with bbox_inches="tight" to avoid cutting off x-axis labels
plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-" + parameter_tuning + "_results_overall_visualization.png",
            bbox_inches="tight", dpi= 300)



#visualizing performances of the different DF (balanced accuracy & F1 score)
label_segment = 90
parameter_tuning = "no"
df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")
# add dataset description depending on value in "Sensor Combination" and "Feature Segments"
df_results["Feature Set and Feature Segment"] = ""
for index, row in df_results.iterrows():
    if row["Sensor Combination"] == "['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']":
        if row["Feature Segments"] == 60:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (60s)"
        elif row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (30s)"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (10s)"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (5s)"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (2s)"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (1s)"
    elif row["Sensor Combination"] == "['linear_accelerometer', 'rotation']":
        if row["Feature Segments"] == 60:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (60s)"
        elif row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (30s)"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (10s)"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (5s)"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (2s)"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (1s)"
# visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(10, 5))
#visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot; use pandas.melt
sns.lineplot(x="Feature Set and Feature Segment", y="value", hue="variable", data=pd.melt(df_results, id_vars=["Feature Set and Feature Segment"], value_vars=["Balanced Accuracy", "F1"]))
plt.title("Model Performance of Different Datasets and Feature Segments")
plt.xlabel("Dataset and Feature-Segment")
plt.ylabel("Score")
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#set y-axis limits
plt.ylim(0, 1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.show()
#save with bbox_inches="tight" to avoid cutting off x-axis labels
plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall_visualization.png",
            bbox_inches="tight", dpi= 300)
#endregion

#region hyperparameter tune the best model from previous comparison step
feature_segments = [30] #in seconds; define length of segments of parameters (over which duration they have been created)
combinations_sensors = [["linear_accelerometer", "gyroscope", "magnetometer", "rotation"]]
label_classes = ["lying (in hand/s)", "sitting: at a table (in hand/s)", "standing (in hand/s)",
                 "walking (in hand/s)", "running (in hand/s)", "cycling (in hand/s)" ] # which label classes should be considered
label_mapping = None
min_gps_accuracy = 35
label_column_name = "label_human motion"
n_permutations = 0 # define number of permutations; better 1000
label_segment = 90 #define how much data around each event will be considered
parameter_tuning = "yes" # if yes: parameter tuning is performed; if False: default parameters are used
drop_cols = [] # columns that should be dropped
feature_importance = "shap"

# set CV mode
combine_participants = False
participants_test_set = [2,4,6,9]

parameter_set = {
    "n_jobs": -1,# use all cores
    "random_state": 11 # set random state for reproducibility
}

# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 100
max_depth = [2, 5, 15, None] # default None
min_samples_split = [2, 10, 30] # default 2
min_samples_leaf = [1, 5] # default 1
max_features = ["sqrt", 5, "log2", 20, None] # default "sqrt"
oob_score = [True, False] # default False;
class_weight = ["balanced", "balanced_subsample", None] # default None
criterion =[ "gini", "entropy", "log_loss"] # default "gini"
max_samples = [None, 0.5, 0.8] # default None which means that 100% of the samples are used for each tree
bootstrap = [True]

# smaller parameter tuning set in order to have less runtime
n_estimators = [100, 800]  # default 100
max_depth = [15, None, 25] # default None
min_samples_split = [2, 8] # default 2
min_samples_leaf = [1] # default 1
max_features = ["sqrt", "log2", None] # default "sqrt"
oob_score = [False, True] # default False;
class_weight = ["balanced", "balanced_subsample"] # default None
criterion =[ "gini", "entropy"] # default "gini"
max_samples = [None, 0.5] # default None which means that 100% of the samples are used for each tree
bootstrap = [True]

grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score = oob_score, class_weight = class_weight,
                            criterion = criterion, max_samples = max_samples, bootstrap = bootstrap)

# select only data which are in the label_segment around ESM_event & drop columns
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Feature Segments"])
df_analytics = pd.DataFrame(columns=["Sensor Combination", "Feature Segment", "Event Description", "Number Participants", "Number Events", "Number Samples", "Number Features"])
for combination_sensors in combinations_sensors:

    for feature_segment in feature_segments:
        # create path to data based on sensor combination and feature segment
        path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_preparation/features/label_human motion_" + str(
            combination_sensors) + "_timeperiod-" + str(feature_segment) + "_FeaturesExtracted_Selected.pkl"
        t0 = time.time()
        print("start of combination_sensors: " + str(combination_sensors) + " and parameter_segment: " + str(feature_segment))
        path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/"+ str(combination_sensors)+ "_feature-segment-" +str(feature_segment) + "s/"
        if not os.path.exists(path_storage):
            os.makedirs(path_storage)
        if path_dataset.endswith(".csv"):
            df = pd.read_csv(path_dataset)
        elif path_dataset.endswith(".pkl"):
            with open(path_dataset, "rb") as f:
                df = pickle.load(f)

        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Dataset loaded",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df = df.drop(columns=drop_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

        #drop NaN in device_id:
        #TODO: probably error that there are still NaN in device_id -> probably from Merging step -> resolve if time
        df = df.dropna(subset=["device_id"])
        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in device_id dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #drop NaN in label_column_name
        df = df.dropna(subset=[label_column_name])
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in label_column_name dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)
        df = Merge_Transform.merge_participantIDs(df, users_iteration02)  # temporary: merge participant ids

        #in case commbine_participants == "yes", merge participant IDs into either 1 (=train) or 2 (=test)
        if combine_participants == True:
            # iterate through all IDs in "device_id" column and replace with 1 or 2: if ID is in participants_test_set, replace with 2, else replace with 1
            df["device_id_traintest"] = df["device_id"].apply(lambda x: 2 if x in participants_test_set else 1)

            # update analytics
            df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                                "Feature Segment": str(feature_segment),
                                                "Event Description": "Training and Test Participants Merged",
                                                "Number Participants": len(df["device_id_traintest"].unique()),
                                                "Number Events": len(df["ESM_timestamp"].unique()),
                                                "Number Samples": len(df),
                                                "Number Features": len(df.columns) - 1}, ignore_index=True)

        # drop rows with NaN: will be many, as after merging of high-freq with GPS, NaN have not be dropped yet
        df = df.dropna()
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in any column dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #combine label classes if necessary
        if label_mapping != None:
            df = df.reset_index(drop=True)
            for mapping in label_mapping:
                df.loc[df["label_human motion"] == mapping[0], "label_human motion"] = mapping[1]
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Label Classes which are not included are dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        # selects only features which are calculated in the label_segment around event.
        ## Note: updated to code so that it actually only selects features for which the WHOLE FEATURE SEGMENT is in the label_segment around event
        df_decisionforest = df[(df['timestamp'] >= (df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2)))) & (df['timestamp'] <= (df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2))- pd.Timedelta(seconds=feature_segment)))]
        df_decisionforest = df_decisionforest.reset_index(drop=True)
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "data outside of label segment around events dropped",
                                            "Number Participants": len(df_decisionforest["device_id"].unique()),
                                            "Number Events": len(df_decisionforest["ESM_timestamp"].unique()),
                                            "Number Samples": len(df_decisionforest),
                                            "Number Features": len(df_decisionforest.columns) - 1}, ignore_index=True)

        print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")

        # check if df_decisionforest is empty
        if df_decisionforest.empty:
            print("df_decisionforest is empty")
            continue

        #run DF
        df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name,
                                                              n_permutations, path_storage, feature_importance = feature_importance,
                                                              confusion_matrix_order = label_classes,  parameter_tuning = parameter_tuning,
                                                              parameter_set = parameter_set, grid_search_space = grid_search_space, combine_participants = combine_participants)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Feature Segments"] = str(feature_segment)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        print("Finished Combination: " + str(combination_sensors) + "and timeperiod: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")

df_analytics.to_csv("/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "s_parameter_tuning-" + parameter_tuning + "_analytics.csv")
df_decisionforest_results_all.to_csv("/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")


#endregion

#region create final (=deployment) model
# Note: this trains a DF on whole dataset without any CV, parameter tuning, feature importance, or validation
# parameter initializations
feature_segment = 30 #in seconds; define length of segments of parameters (over which duration they have been created)
combination_sensors = ["linear_accelerometer", "gyroscope", "magnetometer", "rotation"]
label_classes = ["lying (in hand/s)", "sitting: at a table (in hand/s)", "standing (in hand/s)",
                 "walking (in hand/s)", "running (in hand/s)", "cycling (in hand/s)" ] # which label classes should be considered
label_column_name = "label_human motion"
label_segment = 90 #define how much data around each event will be considered
drop_cols = [] # columns that should be dropped
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/deployment_model/" + str(
    combination_sensors) + "_feature-segment-" + str(feature_segment) + "s/"
parameter_set = {"n_estimators": 800,
                    "max_depth": 15,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": None,
                    "bootstrap": True,
                    "criterion": "gini",
                 "oob_score": False,
                 "class_weight": "balanced",
                 "max_samples": None,
                 "n_jobs": -1,# use all cores
                 "random_state": 11 # set random state for reproducibility
                 }

# initial data transformations
if not os.path.exists(path_storage):
    os.makedirs(path_storage)
path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration02/human_motion/data_preparation/features/label_human motion_" + str(
    combination_sensors) + "_timeperiod-" + str(feature_segment) + "_FeaturesExtracted_Selected.pkl"
df = pd.read_pickle(path_dataset)
df = df.drop(columns=drop_cols)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
df = df.dropna(subset=["device_id"])
df = df.dropna(subset=[label_column_name])
df = Merge_Transform.merge_participantIDs(df, users_iteration02)  # temporary: merge participant ids
df = df.dropna()
df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes
df_decisionforest = df[(df['timestamp'] >= (df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment / 2)))) & (
            df['timestamp'] <= (df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment / 2)) - pd.Timedelta(
        seconds=feature_segment)))]
df_decisionforest = df_decisionforest.reset_index(drop=True)
print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
# create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
df_label_counts["total"] = df_label_counts.sum(axis=1)
df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
df_label_counts.to_csv(
    path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")

#run and save model
model = DecisionForest.DF_sklearn_deployment(df_decisionforest, label_segment, label_column_name, parameter_set, path_storage)
pickle.dump(model, open(
    path_storage + label_column_name + "_timeperiod_around_event-" + str(
        label_segment) + "_FinalDeploymentModel.sav", "wb"))

#endregion

#region OUTDATED; FIRST TRY: modeling with decision forest for human motion
### initialize parameters
parameter_segments = [30] #in seconds; define length of segments of parameters (over which duration they have been created)
#combinations_sensors = [["highfrequencysensors-all", "GPS"],
#                        ["linear_accelerometer", "rotation", "GPS"],
#                        ["linear_accelerometer", "GPS"]]  # define which sensor combinations should be considered
combinations_sensors = [["highfrequency-sensors"]]

label_column_name = "label_human motion - general"
n_permutations = 2 # define number of permutations; better 1000
label_segment = 90 #define how much data around each event will be considered
label_classes = ["standing", "sitting_attable_phoneinhand", "lying_phoneinfront_onback", "wallking", "running", "cycling"]
#label_classes = ["walking", "walking_mediumspeed", "walking_fastspeed", "running", "cycling"] # which label classes should be considered
#label_classes = ["standing", "sitting_attable_phoneinhand", "sitting_attable_phoneontable", "lying_ontheside","lying_phoneinfront_onback"]

# label_classes_all_possible =['sitting_attable_phoneinhand', 'standing', 'on_the_toilet',
#        'cycling', 'running', 'walking', 'walking_fastspeed',
#        'cycling_includingstops', 'lying_phoneoverhead',
#        'lying_phoneonbed', 'sitting_attable_phoneontable',
#        'walking_mediumspeed', 'sitting_on_the_toilet', 'lying_ontheside',
#        'walking_lowspeed', 'on the toilet', 'lying_phoneinfront_onback']
parameter_tuning = "no" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = [] # columns that should be dropped

#testarea
#get unique labels from label_column_name


# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 500
max_depth = [2, 5, 15]
min_samples_split = [2, 10, 30]
min_samples_leaf = [1, 5]
max_features = ["sqrt", 3, "log2", 20]
grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features)


# select only data which are in the label_segment around ESM_event & drop columns
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Parameter Segments"])

for combination_sensors in combinations_sensors:
    for parameter_segment in parameter_segments:
        t0 = time.time()
        print("start of combination_sensors: " + str(combination_sensors) + " and parameter_segment: " + str(parameter_segment))
        path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/data_analysis/decision_forest/label_human motion - general/sensor-"+ combination_sensors[0]+"("+ str(parameter_segment) +"seconds)/"
        if not os.path.exists(path_storage):
            os.makedirs(path_storage)
        path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/features/highfrequency_sensors/activity-"+ label_column_name +"_"+ combination_sensors[0]+ "_feature-segment-"+ str(parameter_segment) +" s_tsfresh-features-exctracted_featuresselected.pkl"
        df = pd.read_pickle(path_dataset)
        #if len(combination_sensors) == 3: #this is necessary to make the code work for the case with three merged sensors
        #    df = df.rename(columns={"timestamp_merged_x": "timestamp_merged", "label_human motion - general_x": "label_human motion - general"})
        #    df = df.drop(columns=["timestamp_merged_y", "label_human motion - general_y"])
        df = df.rename(columns={"sensor_timestamp": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
        df = df.drop(columns=drop_cols)
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        df_decisionforest = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2))) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2)))]
        print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")
        df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, n_permutations, path_storage, parameter_tuning, grid_search_space)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Parameter Segments"] = str(parameter_segment)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        print("Finished Combination: " + str(combination_sensors) + "and timeperiod: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")
df_decisionforest_results_all.to_csv("/Users/benediktjordan/Documents/MTS/Iteration02/data_analysis/decision_forest/label_human motion - general/parameter_tuning-" + parameter_tuning + "_results_overall.csv")


#endregion

#endregion

#endregion



#region Naturalistic Data

#region data exploration for human motion
#region data exploration labels
label_column_name = "label_human motion"
df_labels = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/labels/"

## create table with users x relevant ESM-answers
df_label_counts_bodyposition_smartphonelocation = df_labels.groupby("bodyposition")["smartphonelocation"].value_counts().unstack().fillna(0)
df_label_counts_bodyposition_smartphonelocation["total"] = df_label_counts_bodyposition_smartphonelocation.sum(axis=1)
df_label_counts_bodyposition_smartphonelocation.loc["total"] = df_label_counts_bodyposition_smartphonelocation.sum(axis=0)
df_label_counts_bodyposition_locations = df_labels.groupby("bodyposition")["location"].value_counts().unstack().fillna(0)
df_label_counts_bodyposition_locations["total"] = df_label_counts_bodyposition_locations.sum(axis=1)
df_label_counts_bodyposition_locations.loc["total"] = df_label_counts_bodyposition_locations.sum(axis=0)

## create barplot with all human motion classes
df_labels_humanmotion= df_labels.copy()
#df_labels_publictransport["label_public transport"] = df_labels_publictransport["label_public transport"].replace("train", "public transport")
fig = data_exploration_labels.visualize_esm_activity(df_labels_humanmotion, "label_human motion", "Human Motion - Number of Answers per Class")
fig.savefig(path_storage + "human_motion_class-counts.png")

##create table with users x label classes
df_labels_humanmotion =  Merge_Transform.merge_participantIDs(df_labels_humanmotion, users, include_cities = True)
df_label_counts = data_exploration_labels.create_table_user_activity(df_labels_humanmotion, "label_human motion")
df_label_counts.to_csv(path_storage + "human_motion_labels-overview.csv")


#endregion

#region data exploration sensors
# find out how many labels have sensor data: visualize as barplot and table
segment_around_events = 90 # timeperiod considered around events
min_sensordata_percentage = 50 #in percent; only sensordata above this threshold will be considered
gps_accuracy_min = 35 # in meters; only gps data with accuracy below this threshold was considered when counting GPS records for each event
label_column_name = "label_human motion"
df_esm_including_number_sensordata = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/esm-events_with-number-of-sensorrecords_segment-around-events-" + str(segment_around_events) + "_gps-accuracy-min-" + str(gps_accuracy_min) + " .csv")
# drop row with "nan" in ESM_timestamp
df_esm_including_number_sensordata = df_esm_including_number_sensordata[df_esm_including_number_sensordata["ESM_timestamp"].notnull()]
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
sensors_included = ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "locations"]
## visualize as table
df_esm_including_sensordata_above_threshold, df_esm_user_class_counts = data_exploration_labels.create_table_user_activity_including_sensordata(df_esm_including_number_sensordata, dict_label, label_column_name , sensors_included, segment_around_events)
print("Number of Events for which all Sensor Data is available: " + str(len(df_esm_including_sensordata_above_threshold)))
df_esm_user_class_counts.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_human motion/human_motion_user-class-counts_event-segments-" + str(segment_around_events) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".csv")
## this is necessary for the following visualiztion of sample events: from this df will get the relevant ESM timestamps
df_esm_including_sensordata_above_threshold.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_human motion/human_motion_events_with-sensor-data_event-segments-" + str(segment_around_events) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".csv")

## visualize as barplot
fig = data_exploration_labels.visualize_esm_activity(df_esm_including_sensordata_above_threshold, "label_human motion", "Human Motion - Number of Answers per Class with Sensor Data")
fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_human motion/human_motion_class-counts_event-segments-" + str(segment_around_events) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".png")


# create plots for all events visualizing linear accelerometer, magnetometer, gyroscope, rotation, and speed data
label_column_name = "label_human motion"
time_period = 90  # seconds; timeperiod around event which should be visualized
df_relevant_events = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_human motion/human_motion_events_with-sensor-data_event-segments-90_min-sensor-percentage-50-sensors-['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'locations']_min-gps-accuracy-35.csv")
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_human motion/Event_Visualizations/"
path_sensordata = "/Users/benediktjordan/Downloads/INSERT_esm_timeperiod_5 min.csv_JSONconverted.pkl"
gps_accuracy_min = 35 # meters
only_active_smartphone_sessions = "no"
path_GPS_features = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(gps_accuracy_min) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions + ".csv"
axs_limitations = "general"
label_data = True
figure_title = ""
list_sensors = [["Linear Accelerometer", "linear_accelerometer"],
                ["Magnetometer", "magnetometer"],
                ["Gyroscope", "gyroscope"],
                ["Rotation", "rotation"],
                ["GPS"]] # this sensor set is meant for the appendix
list_sensors = [["Linear Accelerometer", "linear_accelerometer"],
                ["GPS"]] #this sensor set is meant for analysing dynamic human motion
list_sensors = [["Linear Accelerometer", "linear_accelerometer"],
                ["Rotation", "rotation"]] #this sensor set is meant for analysing stationary human motion

## create list of event times for which sensor data exists: double check with GPS data according to above accuracy
event_times = []
for i in range(len(df_relevant_events)):
    event_times.append(df_relevant_events["ESM_timestamp"][i])
df_gps = pd.read_csv(path_GPS_features)
# drop events in event_times if they are not in df_gps["ESM_timestamp"]
event_times = [x for x in event_times if x in df_gps["ESM_timestamp"].tolist()]
num_events = 1
sensor_names = ""
for sensor in list_sensors:
    sensor_names += sensor[0] + "_"

for event_time in tqdm(event_times):
    time0 = time.time()
    # TEMPORARY check if figure has already been saved; if yes, skip
    #list_files = []
    #for root, dirs, files in os.walk(path_to_save):
    #    for file in files:
    #        if file.endswith(".png"):
    #            list_files.append(os.path.join(root, file))
    # check if event_time is already in any element of list_files
    #if any(str(event_time) in s for s in list_files):
    #    print("Figure already saved for event_time: " + str(event_time))
    #    num_events += 1
    #    continue

    fig, activity_name = data_exploration_sensors.vis_sensordata_around_event_severalsensors(list_sensors, event_time, time_period,
                                                                           label_column_name, path_sensordata, axs_limitations, dict_label, label_data = label_data,
                                                                                             path_GPS_features = path_GPS_features)
    # check if fig is None: if yes, continue with next event_time
    if fig is None:
        num_events += 1
        continue

    # find out if path exists
    #NOte: reason for this is that there was a problem with this case
    if activity_name == "car/bus/train/tram":
        activity_name = "car-bus-train-tram"

    # if path_to save doesnt exist, create it with mkdir
    if not os.path.exists(path_to_save + activity_name + "/"):
        os.makedirs(path_to_save + activity_name + "/")

    fig.savefig(
        path_to_save + activity_name + "/gps-accuracy-min-" + str(gps_accuracy_min) + "_" + activity_name + "_EventTime-" + str(event_time) + "_Segment-" + str(
            time_period) + "_Sensors-" + sensor_names + ".png")
    plt.close(fig)
    # print progress
    print("Finished event " + str(num_events) + " of " + str(len(event_times)) + " in " + str((time.time() - time0)/60) + " minutes.")
    num_events += 1



# visualize sensordata of specific events for documentation
event_times_and_activity = [["2022-06-23 13:08:15.327000064", "Walking (Smartphone in One Hand)", [["Linear Accelerometer", "linear_accelerometer"],
                ["GPS"]]],
                            ["2022-06-29 07:00:32.575000064", "Cycling (Smartphone in One Hand)", [["Linear Accelerometer", "linear_accelerometer"],
                ["GPS"]]],
                            ["2022-06-23 08:00:03.296999936", "Standing (Smartphone in One Hand)", [["Linear Accelerometer", "linear_accelerometer"],
                ["Rotation", "rotation"]]],
                            ["2022-06-30 09:24:22.292999936", "Sitting at a Table (Smartphone in One Hand)", [["Linear Accelerometer", "linear_accelerometer"],
                ["Rotation", "rotation"]]],
                            ["2022-06-22 17:00:03.004999936", "Lying (Smartphone in One Hand)", [["Linear Accelerometer", "linear_accelerometer"],
                ["Rotation", "rotation"]]],
                            ["2022-06-23 06:24:26.934000128", "Lying (Smartphone on a Flat Surface)", [["Linear Accelerometer", "linear_accelerometer"],
                ["Rotation", "rotation"]]]]
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_human motion/SummaryStats_Visualizations/"
for event_time_and_activity in event_times_and_activity:
    print("Start with Event " + str(event_time_and_activity[1]))
    figure_title = "Sample Event of " + event_time_and_activity[1]
    list_sensors = event_time_and_activity[2]
    event_time = event_time_and_activity[0]
    sensor_names = ""
    for sensor in list_sensors:
        sensor_names += sensor[0] + "_"
    fig, activity_name = data_exploration_sensors.vis_sensordata_around_event_severalsensors(list_sensors, event_time,
                                                                                             time_period,
                                                                                             label_column_name,
                                                                                             path_sensordata,
                                                                                             axs_limitations,
                                                                                             dict_label,
                                                                                             label_data=label_data,
                                                                                             path_GPS_features=path_GPS_features,
                                                                                             figure_title = figure_title)
    path_to_save_updated = path_to_save + activity_name + "/"
    #create path if it doesnt exist
    if not os.path.exists(path_to_save_updated):
        os.makedirs(path_to_save_updated)
    fig.savefig(
        path_to_save_updated + "gps-accuracy-min-" + str(gps_accuracy_min) + "_" + activity_name + "_EventTime-" + str(event_time) + "_Segment-" + str(
            time_period) + "_Sensors-" + sensor_names + ".png", dpi = 300)


# visualize mean/std x max x human motion classes in scatterplot for GPS and accelerometer data
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
time_periods = [10,20,40, 90, 180]
list_activities = ["standing", "sitting: at table", "lying", "walking", "cycling"]
join_classes = "yes"
visualize_participants = "no"
gps_accuracy_min = 10 # minimum accuracy of GPS data to be included in analysis
sensors = [["Linear Accelerometer", "linear_accelerometer"],
           ["GPS", "locations"],
           ["Rotation", "rotation"]]
only_sensordata_of_active_smartphone_sessions = "yes"
label_column_name = ["Human Motion Classes", "label_human motion"]
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/" + label_column_name[1] + "/SummaryStats_Visualizations/"

for sensor in sensors:
    for time_period in time_periods:
        if sensor[0] == "GPS":
            df_summary_stats = pd.read_csv(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/GPS_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + ".csv")
            # add label
            df_summary_stats = labeling_sensor_df(df_summary_stats, dict_label, label_column_name[1],
                                                  ESM_identifier_column="ESM_timestamp")

            # change label_column_name in df_summary_stats
            df_summary_stats = df_summary_stats.rename(columns={label_column_name[1]: label_column_name[0]})

            # combine human motion classes
            if join_classes == "yes":
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["standing (in one hand)", "standing (in two hands)"], "standing")
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["sitting: at a table (in one hand)", "sitting: at a table (in two hands)"], "sitting: at table")
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["lying (in one hand)", "lying (in two hands)"], "lying")
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["walking (in one hand)", "walking (in two hands)"], "walking")
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["cycling (in one hand)", "cycling (in two hands)"], "cycling")

            fig = data_exploration_sensors.vis_summary_stats(df_summary_stats, sensor[1] , label_column_name[0], list_activities,
                                    "Mean and Maximum Speed and Acceleration of Human Motion Events", visualize_participants)
            fig.savefig(path_storage + "SummaryStatsVisualization_GPS_mean_max_gps-accuracy-min-" + str(gps_accuracy_min) + "_activities-included-" + str(list_activities) + "_including-participants-" + visualize_participants + "_time-period-around-event-"
                        + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + "s_scatterplot.png", dpi=300)

        else:
            df_summary_stats = pd.read_csv(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/" + sensor[1] + "_summary-stats_time-period-around-event-" + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + ".csv")
            # add label
            df_summary_stats = labeling_sensor_df(df_summary_stats, dict_label, label_column_name[1], ESM_identifier_column="ESM_timestamp")

            # change label_column_name in df_summary_stats
            df_summary_stats = df_summary_stats.rename(columns={label_column_name[1]: label_column_name[0]})

            # combine human motion classes
            if join_classes == "yes":
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["standing (in one hand)", "standing (in two hands)"], "standing")
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["sitting: at a table (in one hand)", "sitting: at a table (in two hands)"], "sitting: at table")
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["lying (in one hand)", "lying (in two hands)"], "lying")
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["walking (in one hand)", "walking (in two hands)"], "walking")
                df_summary_stats[label_column_name[0]] = df_summary_stats[label_column_name[0]].replace(
                    ["cycling (in one hand)", "cycling (in two hands)"], "cycling")

            figure_title = "Mean and Maximum of the " + sensor[0] + " of Human Motion Events"
            fig = data_exploration_sensors.vis_summary_stats(df_summary_stats, sensor[1], label_column_name[0], list_activities, figure_title , visualize_participants)
            fig.savefig(
                path_storage + "SummaryStatsVisualization_" + sensor[1] + "_std_max_activities-included-" + str(
                    list_activities) + "_including-participants-" + visualize_participants + "_time-period-around-event-"
                + str(time_period) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + "s_scatterplot.png", dpi=300)




#testarea for location data
df_locations = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration.csv")
#label data
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
df_locations = labeling_sensor_df(df_locations, dict_label, "label_public transport", ESM_identifier_column="ESM_timestamp")
# delete rows with nan values in label_public transport
df_locations = df_locations.dropna(subset=["label_public transport"])

#calculate summary stats for column accuracy
df_locations["loc_accuracy"].describe()

# only retain rows with accuracy < 35
df_locations_new = df_locations[df_locations["loc_accuracy"] < 35]
#count number of unique ESM_timestamps
df_locations["ESM_timestamp"].nunique()
df_locations_new["ESM_timestamp"].nunique()

#retain only data which is in 90s time period around ESM_timestamp
time_period = 90
#convert timestamp to datetime
df_locations_new["ESM_timestamp"] = pd.to_datetime(df_locations_new["ESM_timestamp"])
df_locations_new["timestamp"] = pd.to_datetime(df_locations_new["timestamp"])
df_locations_new = df_locations_new[(df_locations_new['timestamp'] >= df_locations_new['ESM_timestamp'] - pd.Timedelta(seconds=(time_period / 2))) & (
        df_locations_new['timestamp'] <= df_locations_new['ESM_timestamp'] + pd.Timedelta(seconds=(time_period / 2)))]

df_locations_new["ESM_timestamp"].nunique()
# take only the unique ESM_timestamps and count from them the number of label_public transport
df_locations_new_unique_ESM_timestamps = df_locations_new.drop_duplicates(subset=["ESM_timestamp"])
df_locations_new_unique_ESM_timestamps["label_public transport"].value_counts()





# add labels
df_locations = labeling_sensor_df(df_locations, dict_label, "label_public transport", ESM_identifier_column="ESM_timestamp")
#delete NaN values in label_public transport
df_locations = df_locations[df_locations["label_public transport"].notna()]
#get only data from user with loc_device_id = 590f0faf-d932-4a57-998d-e3da667a91dc
df_locations_user = df_locations[df_locations["loc_device_id"] == "b7b013b7-f78c-4325-a7ab-2dfc128fba27"]

#endregion

#endregion

#region data preparation for human motion

# load labels
sensors_included = "all"
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_" + sensors_included + "_transformed_labeled_dict.pkl", 'rb') as f:
    dict_label = pickle.load(f)

#region merge different sensors 1. set: all high-freq. sensors; 2. set: linear accelerometer & rotation
label_column_name = "label_human motion"
sensor_sets_to_merge = [["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"],
                        ["linear_accelerometer", "rotation"]]
timedelta = "100ms" #must be in format compatible with pandas.Timedelta
columns_to_delete = ["device_id", "ESM_timestamp", "Unnamed: 0", "0", "1", "3"]
add_prefix_to_merged_columns = False # this is not necessary here since in the sensor data is already prefix contained
path_datasets = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/"

for sensor_set in sensor_sets_to_merge:
    print("Merging sensor set: " + str(sensor_set))
    counter = 1
    for sensor in sensor_set:
        print("Start with sensor: " + sensor)
        if counter == 1:
            sensor_base = sensor
            df_base = pd.read_pickle(path_datasets + sensor + "_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl")
            counter += 1
            continue
        if counter > 1:
            sensor_tomerge = sensor
            df_tomerge = pd.read_pickle(path_datasets + sensor + "_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl")
            df_base = Merge_and_Impute.merge(df_base, sensor_base, df_tomerge, sensor_tomerge, timedelta, columns_to_delete, add_prefix_to_merged_columns)
            counter += 1
    df_base.to_csv(
        "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/timeseries_merged/_esm_timeperiod_5 min_only-active-smartphone-sessions_timeseries-merged_sensors-" + str(sensor_set) + ".csv", index=False)
#endregion

#region calculate distance, speed & acceleration
# load data around events
only_sensordata_of_active_smartphone_sessions = "yes"
accuracy_thresholds = [10, 35, 50, 100] #accuracy is measured in meters

#load GPS data
if only_sensordata_of_active_smartphone_sessions == "yes":
    df_locations_events = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/locations_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl")
else:
    df_locations_events = pd.read_pickle("/Users/benediktjordan/Downloads/locations_esm_timeperiod_5 min.csv_JSONconverted.pkl")

#drop duplicates (this step should be integrated into data preparation)
df_locations_events = df_locations_events.drop(columns=["Unnamed: 0", "0"])
df_locations_events = df_locations_events.drop_duplicates()

# calculate distance, speed and acceleration (class build in "FeatureExtraction_GPS.py")
df_features = FeatureExtraction_GPS().calculate_distance_speed_acceleration(df_locations_events)

# drop rows where distance, speed or acceleration contain NaN (they contain NaN if it is first entry of every event)
df_features= df_features.dropna(subset=["distance (m)", "speed (km/h)", "acceleration (km/h/s)"])
# drop rows with unrealistic speed values  & GPS accuracy values
df_features = df_features[df_features["speed (km/h)"] < 300]
# create different GPS accuracy thresholds
for accuracy_threshold in accuracy_thresholds:
    df_features_final = df_features[df_features["loc_accuracy"] < accuracy_threshold]
    df_features_final = df_features_final.reset_index(drop=True)
    print("Number of rows with accuracy < " + str(accuracy_threshold) + "m: " + str(len(df_features_final)))
    #save to csv
    df_features_final.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(accuracy_threshold) + "_only-active-smartphone-sessions-" + only_sensordata_of_active_smartphone_sessions + ".csv")
#endregion

#region feature extraction of high-frequency features: for both sensor subsets
## COMPARE CODE IN OTHER PROJECT
### what is missing:
### - feature extraction for all highfreq: 2,5, and 10s feature segments
### - combining chunks: for LinAcc-Rot set: 1s and 2s chunks; for all highfreq: all feature segments chunks
#endregion

#region feature extraction of Speed & Acceleration features
## COMPARE CODE IN OTHER PROJECT
#endregion

#region merge high-frequency features with speed & acceleration features (or, for feature_segment == 1: merge high-frequency features with speed & acceleration)
## WHAT STILL MISSING: for set "LinAcc-Rot": everything; for set "all highfreq": everything
### NOTE: for LinAcc-Rot set, it crashed always; therefore I implemented a "chunk" approach: after data for one user is merged,
### the intermediate result is stored in "path_intermediate_files". If the function is called again, it starts only after this user
high_freq_sensor_sets = [["linear_accelerometer", "rotation"],
                         ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]]
high_freq_sensor_sets = [["linear_accelerometer", "rotation"]]
feature_segments = [60, 30, 10,5,2,1] #in seconds
feature_segments = [30] #in seconds
only_active_smartphone_sessions = "yes"
min_gps_accuracy = 35
columns_to_delete = ["device_id", "ESM_timestamp"]
path_features = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/"
path_intermediate_files = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/temp/" #this location is for the intermediate df_final files which will be saved after each user is done. If the
# script is stopped, the intermediate files can be loaded and the script can be continued with the next user

for high_freq_sensor_set in high_freq_sensor_sets:
    for feature_segment in feature_segments:
        timedelta = str(feature_segment) + "s"
        print("Start with high_freq_sensor_set: " + str(high_freq_sensor_set) + " and feature_segment: " + str(feature_segment))
        df_features_highfreq = pd.read_pickle(path_features + str(high_freq_sensor_set) + "_timeperiod-" + str(
            feature_segment) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions + "_FeaturesExtracted.pkl")

        # if feature_segment == 1, use speed & acceleration data, as there are no derived features
        if feature_segment == 1:
            df_features_speedacc = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(min_gps_accuracy) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions  + ".csv")
        else:
            df_features_speedacc = pd.read_pickle(path_features + "GPS_timeperiod-" + str(feature_segment) + "_min-gps-accuracy-" + str(min_gps_accuracy) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions  + "_FeaturesExtracted.pkl")

        df_merged = Merge_and_Impute.merge(df_features_highfreq, df_features_speedacc, "GPS", timedelta, columns_to_delete, path_intermediate_files,
                  add_prefix_to_merged_columns=False)

        #double check if merging worked
        #df_merged_timecolumns = df_merged[["timestamp", "GPS_timestamp_merged"]]

        # save with open and pickle highest protocall
        with open(path_storage + str(high_freq_sensor_set) + "-merged-to-GPSFeatures_feature-segment-" + str(feature_segment) + "_min-gps-accuracy-" + str(min_gps_accuracy) +"_only-active-smartphone-sessions-" + only_active_smartphone_sessions + "_FeaturesExtracted_Merged.pkl", 'wb') as f:
            pickle.dump(df_merged, f, pickle.HIGHEST_PROTOCOL)


#testarea DO IT!
df_features = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']-merged-to-GPSFeatures_feature-segment-10_min-gps-accuracy-35_only-active-smartphone-sessions-yes_FeaturesExtracted_Merged.pkl")# count NaN values in ESM_timestamp column
df_features["ESM_timestamp"].isnull().sum()
df_merged_timecolumns["GPS_timestamp_merged"].isnull().sum()

#endregion

#region feature selection: data-driven approach
## COMPARE CODE IN OTHER PROJECT
### What is missing:
### - feature selection for all highfreq AND GPS: all feature segments
#endregion

#region feature selection: hypothesis-driven approach
## choose the features which I need
path_features_merged = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/"

sensor_set = ["linear_accelerometer", "rotation"]
feature_segments = [60,30,10,5,2,1]
feature_segments = [60,30]
min_gps_accuracy = 35
only_active_smartphone_sessions = "yes"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/hypothesis_driven_approach/"

features_HypothesisApproach = [["linear_accelerometer", "maximum", "one feature"],
                                 ["linear_accelerometer", "minimum", "one feature"],
                                    ["linear_accelerometer", "mean", "one feature"],
                                    ["linear_accelerometer", "median", "one feature"],
                                    ["linear_accelerometer", "standard_deviation", "one feature"],
                                    ["linear_accelerometer", "root_mean_square", "one feature"],
                                    ["linear_accelerometer", "fft_coefficient__attr_\"abs\"", "several features"],
                                    ["rotation", "mean", "one feature"],
                                    ["rotation", "median", "one feature"],
                                    ["rotation", "standard_deviation", "one feature"],
                                    ["", "mean", "Speed feature"],
                                    ["", "median", "Speed feature"],
                                    ["", "maximum", "Speed feature"]]
for feature_segment in feature_segments:
    print("Start with sensor_set: " + str(sensor_set) + " and feature_segment: " + str(feature_segment))
    #load features
    df_features = pd.read_pickle(path_features_merged + str(sensor_set) + "-merged-to-GPSFeatures_feature-segment-" + str(feature_segment) + "_min-gps-accuracy-" + str(min_gps_accuracy) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions + "_FeaturesExtracted_Merged.pkl")
    # create list of names of features to keep
    features_to_keep_one = []
    for feature in features_HypothesisApproach:
        if feature[2] == "one feature":
            for axis in ["double_values_0", "double_values_1", "double_values_2"]:
                features_to_keep_one.append(feature[0][:3] + "_" + axis + "__" + feature[1])
        elif feature[2] == "Speed feature":
            features_to_keep_one.append("speed (km/h)__" + feature[1])
    features_to_keep_several = []
    for feature in features_HypothesisApproach:
        if feature[2] == "several features":
            for axis in ["double_values_0", "double_values_1", "double_values_2"]:
                features_to_keep_several.append(feature[0][:3] + "_" + axis + "__" + feature[1])
    # add "sensor_timestamp", "ESM_timestamp", "device_id" to features_to_keep
    features_to_keep_one.extend(["timestamp", "ESM_timestamp", "device_id"])
    # keep only columns from df_feature than contain either exactly one of the features from "features_to_keep_one" or contain in their name one of the features from "features_to_keep_several"
    df_features_hypothesisdriven = df_features[df_features.columns[df_features.columns.isin(features_to_keep_one) | df_features.columns.str.contains('|'.join(features_to_keep_several))]]
    df_features_hypothesisdriven.to_csv(path_storage + str(sensor_set) + "_timeperiod-" + str(feature_segment) + "_min-gps-accuracy-" + str(min_gps_accuracy) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions + "_FeaturesExtracted_FeaturesSelected-HypothesisDrivenApproach.csv", index=False)

    # double-check: get the columns of df_features_hypothesisdriven
    df_features_hypothesisdriven_columns = pd.DataFrame(df_features_hypothesisdriven.columns)

##add the feature "dominant frequency" to the list of features and compute it as the maximum of the fft_coefficient__attr_"abs" features
sampling_rate = 10 #Hz
for feature_segment in feature_segments:
    number_of_samples_per_feature = feature_segment * sampling_rate #this number tells how many records are used for one feature value to compute
    df_features = pd.read_csv(path_storage + str(sensor_set) + "_timeperiod-" + str(feature_segment) + "_min-gps-accuracy-" + str(min_gps_accuracy) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions + "_FeaturesExtracted_FeaturesSelected-HypothesisDrivenApproach.csv")
    # iterate through the three axis of the accelerometer
    for axis in ["lin_double_values_0", "lin_double_values_1", "lin_double_values_2"]:
        # get in each row the name of the column in which maximum of the fft_coefficient__attr_"abs" features is
        df_features[axis + "__fft_coefficient__attr_\"abs\"__dominant_frequency"] = df_features.loc[:, df_features.columns.str.contains(axis + "__fft_coefficient__attr_\"abs\"__")].idxmax(axis=1)
        # retain only whatever is after the last "_"
        df_features[axis + "__fft_coefficient__attr_\"abs\"__dominant_frequency"] = df_features[axis + "__fft_coefficient__attr_\"abs\"__dominant_frequency"].str.split("_").str[-1]
        # convert the index into the frequency
        df_features[axis + "__fft_coefficient__attr_\"abs\"__dominant_frequency"] = df_features[axis + "__fft_coefficient__attr_\"abs\"__dominant_frequency"].astype(float)
        df_features[axis + "__fft_coefficient__attr_\"abs\"__dominant_frequency"] = df_features[axis + "__fft_coefficient__attr_\"abs\"__dominant_frequency"] * sampling_rate / number_of_samples_per_feature

    # save the dataframe
    df_features.to_csv(path_storage + str(sensor_set) + "_timeperiod-" + str(feature_segment) + "_min-gps-accuracy-" + str(min_gps_accuracy) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions + "_FeaturesExtracted_FeaturesSelected-HypothesisDrivenApproach.csv", index=False)

## cleaning of selected features
### NOTEuse the function from the high-frequency feature selection!




# RELOCATE THIS SECTION SOMEWHERE ELSE: explore features & understand them
df_features = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/linear_accelerometer/linear_accelerometer_timeperiod-10_only-active-smartphone-sessions-yes s_FeaturesExtracted.pkl")
feature_names = pd.DataFrame(df_features.columns)


# import filtfilt
from scipy.signal import filtfilt
from scipy.signal import butter, lfilter

# Generate test signal
fs = 1000  # sample rate
t = np.linspace(0, 1, fs)  # time vector
f1 = 50  # frequency 1
f2 = 150  # frequency 2
A1 = 2  # amplitude 1
A2 = 1  # amplitude 2
sig = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)
plt.plot(t, sig)
plt.show()

# Measure dominant frequency
fft_sig = np.fft.fft(sig)
fft_freq = np.fft.fftfreq(len(sig), 1/fs)
dominant_freq = fft_freq[np.argmax(np.abs(fft_sig))]
print("Dominant frequency: {:.2f} Hz".format(dominant_freq))

# Design band-pass filter
low = dominant_freq - 5
high = dominant_freq + 5
nyquist = 0.5 * fs
low = low / nyquist
high = high / nyquist
b, a = butter(3, [low, high], btype='band')

# Apply band-pass filter to original signal
filtered_sig = filtfilt(b, a, sig)

# Measure peak-to-peak difference in filtered signal
p2p_diff = np.ptp(filtered_sig)
print("Peak-to-peak difference in filtered signal: {:.2f}".format(p2p_diff))

# Plot original and filtered signal
plt.figure(figsize=(12,6))
plt.subplot(211)
plt.plot(t, sig)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.subplot(212)
plt.plot(t, filtered_sig)
plt.title('Filtered Signal')
plt.xlabel('Time (s)')
plt.show()




## cyclical patterns
import numpy as np
import matplotlib.pyplot as plt

# Signal 1: Constant amplitude sine wave at 1 Hz
fs = 100  # sample rate
t = np.linspace(0, 1, fs)  # time vector
f1 = 1  # frequency
A1 = 1  # amplitude
sig1 = A1 * np.sin(2 * np.pi * f1 * t)

# Signal 2: Two sine waves at 1 Hz and 2 Hz with constant amplitude
f2 = 2
A2 = 1
sig2 = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

# Signal 3: Two sine waves at 1 Hz and 2 Hz with variable amplitude
A3 = 2
A4 = 0.5
sig3 = A3 * np.sin(2 * np.pi * f1 * t) + A4 * np.sin(2 * np.pi * f2 * t)

# Signal 4: Amplitude modulated sine wave at 1 Hz
f3 = 1
A5 = 1
sig4 = A5 * np.sin(2 * np.pi * f3 * t) * np.sin(2 * np.pi * 5 * t)

# Plot the signals
plt.figure(figsize=(12,6))
plt.subplot(221)
plt.plot(t, sig1)
plt.title('Signal 1')
plt.xlabel('Time (s)')
plt.subplot(222)
plt.plot(t, sig2)
plt.title('Signal 2')
plt.xlabel('Time (s)')
plt.subplot(223)
plt.plot(t, sig3)
plt.title('Signal 3')
plt.xlabel('Time (s)')
plt.subplot(224)
plt.plot(t, sig4)
plt.title('Signal 4')
plt.xlabel('Time (s)')
plt.show()

# Calculate FFT for each signal
fft1 = np.fft.fft(sig1)
fft2 = np.fft.fft(sig2)
fft3 = np.fft.fft(sig3)
fft4 = np.fft.fft(sig4)

# Plot the frequency-amplitude spectrum
plt.figure(figsize=(12,6))
plt.subplot(221)
plt.plot(np.abs(fft1))
plt.title('Signal 1')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.subplot(222)
plt.plot(np.abs(fft2))
plt.title('Signal 2')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.subplot(223)
plt.plot(np.abs(fft3))
plt.title('Signal 3')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.subplot(224)
plt.plot(np.abs(fft4))
plt.title('Signal 4')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()
#endregion


#region create different combinations of sensor features OUTDATED!!!
combinations_sensors = [["highfrequencysensors-all", "GPS"],
                        ["linear_accelerometer", "rotation", "GPS"],
                        ["linear_accelerometer", "GPS"]]
segments_highfreqfeatures = [30,10,5,2,1]
timedelta = "1000ms" # the time which merged points can be apart; has to be in a format compatible with pd.TimeDelta()
drop_cols = ['device_id', "ESM_timestamp"] # columns that should be deleted from df_tomerge
for combination in combinations_sensors:
    for segment in segments_highfreqfeatures:
        print("started with combination: " + str(combination) + " and segment: " + str(segment))
        time_0  = time.time()
        df_base = pd.read_pickle(
            "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/"+ combination[0] +"/activity-"+ label_column_name +"_sensor-"+ combination[0] +"_timeperiod-"+ str(segment) +" s_featureselection.pkl")  # load df_tomerge: this should be the df with features with higher "sampling rate"/frequency
        df_base = df_base.dropna(subset=["label_human motion - general"])  # drop NaN labels
        #check if more than two sensors need to be merged
        if len(combination) == 3:
            df_tomerge = pd.read_pickle(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/"+ combination[1] +"/activity-"+ label_column_name +"_sensor-"+ combination[1] +"_timeperiod-"+ str(segment) +" s_featureselection.pkl")
            timedelta_intermediate = str(int((segment/2)*1000)) + "ms"
            df_base = Merge_and_Impute.merge(df_base, df_tomerge, timedelta_intermediate, drop_cols)  # merge df_base and df_tomerge

        df_tomerge = pd.read_csv(
            "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration.csv")  # load df_base: this should be the df with features with less "sampling rate"/frequency; # here: load motion features which are computed for 2 seconds

        df_final = Merge_and_Impute.merge(df_base, df_tomerge, timedelta, drop_cols)  # merge df_base and df_tomerge
        # TODO integrate "delete columns with only "NaN" values" into "features selection" section
        df_final = df_final.dropna(axis=1, how='all')  # delete columns with only "NaN" values
        #get number of rows with missing values
        print("So many rows contain missing values after merging: " + str(df_final.isnull().any(axis=1).sum()))
        df_final_nan = Merge_and_Impute.impute_deleteNaN(df_final)  # delete rows which contain missing values
        # storage_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-label_human motion - general_sensor-highfrequencysensorsAll(1seconds)-and-locations(1seconds).pkl"
        storage_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-"+ label_column_name +"_sensor-"+ combination[0]+"("+ str(segment) +"seconds)-and-GPS(1seconds).pkl"
        if len(combination) == 3:
            storage_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-"+ label_column_name +"_sensor-"+ combination[0]+"("+ str(segment) +"seconds)-"+combination[1] +"("+ str(segment) +"seconds)-and-GPS(1seconds).pkl"
        df_final_nan.to_pickle(storage_path)
        print("finished with combination: " + str(combination) + " and segment: " + str(segment) + " in " + str((time.time() - time_0)/60) + " minutes")
#endregion
#endregion

#region testing label segments for human motion
#region testing the label segments
label_column_name = "label_human motion"
feature_segment = 2
sensor_set = ['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']
path_features = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/data_driven_approach/label_human motion_" +str(sensor_set) + "-merged-to-GPSFeatures_feature-segment-" + str(feature_segment) + "_min-gps-accuracy-35_only-active-smartphone-sessions-yes_FeaturesExtracted_Merged_Selected.csv"
n_permutations = 0 # define number of permutations; better 1000
label_segments = list(range(65, 121, 5)) #in seconds; defines label segments to test
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/label_segment_testing/" + str(sensor_set) + "-GPS_feature-segment-" + str(feature_segment) + "/"

# if path_storage doesnt exist, create it
if not os.path.exists(path_storage):
    os.makedirs(path_storage)

# combine label classes if necessary
label_mapping = [["lying (in one hand)", "lying (hand/s)"],
                 ["lying (in two hands)", "lying (hand/s)"],
                 ["lying (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                 ["standing (in one hand)", "standing (hand/s)"],
                 ["standing (in two hands)", "standing (hand/s)"],
                 ["standing (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                 ["sitting: at a table (in one hand)", "sitting (hand/s)"],
                 ["sitting: at a table (in two hands)", "sitting (hand/s)"],
                 ["sitting: at a table (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                 ["sitting: on the couch (in one hand)", "sitting (hand/s)"],
                 ["sitting: on the couch (in two hands)", "sitting (hand/s)"],
                 ["sitting: somewhere else (in one hand)", "sitting (hand/s)"],
                 ["sitting: somewhere else (in two hands)", "sitting (hand/s)"],
                 ["walking (in one hand)", "walking (hand/s)"],
                 ["walking (in two hands)", "walking (hand/s)"],
                 ["cycling (in one hand)", "cycling (hand/s)"],
                 ["cycling (in two hands)", "cycling (hand/s)"]
                 ]

label_classes = ["lying (hand/s)","standing (hand/s)", "sitting (hand/s)",
                 "standing, sitting, or lying (on flat surface)",
                 "walking (hand/s)"] # which label classes should be considered

parameter_tuning = "no" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = [] # columns that should be dropped
df = pd.read_csv(path_features)

parameter_set = {
    "n_estimators": 100, #default
    "criterion": "gini",#default
    "max_depth": None,#default
    "min_samples_split": 2,#default
    "min_samples_leaf": 1,#default
    "min_weight_fraction_leaf": 0.,#default
    "max_features": None, # THIS IS CHANGED
    "max_leaf_nodes": None,#default
    "min_impurity_decrease": 0.0,#default
    "bootstrap": True,#default
    "oob_score": False,#default
    "n_jobs": None,#default
    "verbose": 0,#default
    "warm_start": False,#default
    "class_weight": "balanced", # THIS IS CHANGED
    "n_jobs": -1, #CHANGED: all processors
    "random_state": 11 #default
}

#drop rows in which device_id == nan
#TODO: in relatively many rows the device_id == NaN and I have no idea why. Investigate this issue
# by looking into the creation of the hypothesis driven dataset (maybe a side effect of merging somewhere...)
# one idea: I think this is due to an incorrect code in merging features: similar problem which I had before with ESM_timestamps:
# if a row cant be merged, it is concatenated in the end, but device_id is not included (I think)

df = df.dropna(subset=["device_id"])

# drop all rows which contain at least one missing value (didn´t do that before since I thought that I could train DF with missing values)
#TODO check if I use imputation in the Cleaning step (I dont think so though!)
df = df.dropna()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
df = df.drop(columns=drop_cols)
df = Merge_Transform.merge_participantIDs(df, users) #temporary: merge participant ids

df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1", "Precision", "Recall"])
for label_segment in label_segments:
    t0 = time.time()

    # implement label transformation if necessary
    if label_mapping != None:
        df = df.reset_index(drop=True)
        for mapping in label_mapping:
            df.loc[df["label_human motion"] == mapping[0], "label_human motion"] = mapping[1]
    df = df[
        df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

    #select only timeframe of label segment around each event
    df_decisionforest = df[(df['timestamp'] >= (df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment / 2)))) & (
            df['timestamp'] <= (df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment / 2)) - pd.Timedelta(
        seconds=feature_segment)))]
    print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))

    # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
    df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
    df_label_counts["total"] = df_label_counts.sum(axis=1)
    df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
    df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")

    # check if df_decisionforest is empty
    if df_decisionforest.empty:
        print("df_decisionforest is empty for label_segment: " + str(label_segment))
        continue
    # implement DF
    df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, n_permutations, path_storage,
                                                          confusion_matrix_order=label_classes,
                                                          parameter_tuning=parameter_tuning,
                                                          parameter_set=parameter_set)
    df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
    #intermediate saving
    df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
    print("finished with label_segment: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")
df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-1-11s_parameter_tuning-"+ parameter_tuning + "df_results.csv")
#endregion

#region visualizing the accuracies for different label segments
#df_decisionforest_results_all = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/labelsegments_comparison/features-highfrequencysensors-all-1s-gps-1s/label_human motion - general_timeperiod_around_event-29s_parameter_tuning-no_results.csv")
paths_label_segment_comparisons = [["Feature Segment: 2 Seconds", "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/label_segment_testing/['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']-GPS_feature-segment-2/label_human motion_timeperiod_around_event-5-75s_parameter_tuning-nodf_results.csv"],
                                   ["Feature Segment: 10 Seconds", "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/label_segment_testing/['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']-GPS_feature-segment-10s/label_human motion_timeperiod_around_event-15-120s_parameter_tuning-no_results.csv"],
                                   ["Feature Segment: 30 Seconds", "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/label_segment_testing/['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']-GPS_feature-segment-30/label_human motion_timeperiod_around_event-35-120s_parameter_tuning-no_results.csv"]]

#create lineplot for Seconds around Eventx x Balanaced Accuracy with Seaborn for different feature segments (2,10,30) all in same plot
fig, ax = plt.subplots(figsize=(10, 5))
# create colour palette as before
pal = sns.color_palette("bright")
hex_colors = [matplotlib.colors.rgb2hex(color) for color in pal]
## remove similar colors from the palette
hex_colors = [color for color in hex_colors if
              color != '#023eff' and color != '#e8000b' and color != '#f14cc1' and color != '#1ac938']
hex_colors = hex_colors[:len(paths_label_segment_comparisons)]
color_counter = 0
for path in paths_label_segment_comparisons:
    df = pd.read_csv(path[1])
    hex_color = hex_colors[color_counter]
    color_counter += 1
    sns.lineplot(x="Seconds around Event", y="Balanced Accuracy", data=df, label=path[0], palette=hex_color)
ax.set(xlabel="Seconds Around Event", ylabel="Model Performance (Balanced Accuracy)")
#sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
#sns.set(style="whitegrid")
plt.tight_layout()
#include title
plt.title("Label Segment Comparison for Different Feature Segments")
plt.legend()
plt.show()
fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/label_segment_testing/label_segment_for_different_feature_segments_comparison", dpi=600, bbox_inches='tight')
#endregion
#endregion

#region modeling for human motion

##region Decision Forest for comparing different datasets (2 sensor combinations x 6 feature segments)
#TODO: run DF comparison with only high-frequency sensor data (without GPS): should be a lot more data (check additionally
# where the data, where only high-frequency was available, was actually deleted before (make sure it wasnt imputed); and compare the results
# to the DF runs which included GPS data

#region training DF
feature_segments = [60, 30,10, 5,2,1] #in seconds; define length of segments of parameters (over which duration they have been created)
feature_segments = [2,1] #in seconds; define length of segments of parameters (over which duration they have been created)

combinations_sensors = [["linear_accelerometer", "rotation"],
                        ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]
                        ]  # define which sensor combinations should be considered

combinations_sensors = [
                        ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]
                        ]  # define which sensor combinations should be considered
min_gps_accuracy = 35
only_active_smartphone_sessions = "yes"
label_column_name = "label_human motion"
n_permutations = 0 # define number of permutations; better 1000
label_segment = 45 #define how much data around each event will be considered
# if label classes should be joint -> define in label mapping
label_mapping = [["lying (in one hand)", "lying (hand/s)"],
                            ["lying (in two hands)", "lying (hand/s)"],
                            ["lying (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                         ["standing (in one hand)", "standing (hand/s)"],
                            ["standing (in two hands)", "standing (hand/s)"],
                               ["standing (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                            ["sitting: at a table (in one hand)", "sitting (hand/s)"],
                               ["sitting: at a table (in two hands)", "sitting (hand/s)"],
                            ["sitting: at a table (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                         ["sitting: on the couch (in one hand)", "sitting (hand/s)"],
                            ["sitting: on the couch (in two hands)", "sitting (hand/s)"],
                         ["sitting: somewhere else (in one hand)", "sitting (hand/s)"],
                            ["sitting: somewhere else (in two hands)", "sitting (hand/s)"],
                         ["walking (in one hand)", "walking (hand/s)"],
                            ["walking (in two hands)", "walking (hand/s)"],
                         ["cycling (in one hand)", "cycling (hand/s)"],
                            ["cycling (in two hands)", "cycling (hand/s)"]
                         ]
label_classes = ["lying (hand/s)", "sitting (hand/s)", "standing (hand/s)",
                 "standing, sitting, or lying (on flat surface)",
                 "walking (hand/s)" ] # which label classes should be considered
parameter_tuning = "no" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = ["Unnamed: 0"] # columns that should be dropped
feature_importance = "shap"

parameter_set = {
    "n_estimators": 100, #default
    "criterion": "gini", #default
    "max_depth": None, #default
    "min_samples_split": 2,#default
    "min_samples_leaf": 1,#default
    "min_weight_fraction_leaf": 0.,#default
    "max_features": None, # THIS IS CHANGED
    "max_leaf_nodes": None,#default
    "min_impurity_decrease": 0.0,#default
    "bootstrap": True,#default
    "oob_score": False,#default
    "n_jobs": None,#default
    "verbose": 0,#default
    "warm_start": False,#default
    "class_weight": "balanced", # THIS IS CHANGED
    "n_jobs": -1,#default
    "random_state": 11#default
}

# select only data which are in the label_segment around ESM_event & drop columns
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Feature Segments"])
df_analytics = pd.DataFrame(columns=["Sensor Combination", "Feature Segment", "Event Description", "Number Participants", "Number Events", "Number Samples", "Number Features"])
for combination_sensors in combinations_sensors:
    for feature_segment in feature_segments:
        # since no dataset yet for combination_sensors = and feature_segments = 1
        if combination_sensors == ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"] and feature_segment == 1:
            print("no dataset yet")
            continue

        # create path to data based on sensor combination and feature segment
        if combination_sensors == ["linear_accelerometer", "rotation"]:
            path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/hypothesis_driven_approach/" + str(
                combination_sensors) + "_timeperiod-" + str(feature_segment) + "_min-gps-accuracy-" + str(
                min_gps_accuracy) + "_only-active-smartphone-sessions-" + str(
                only_active_smartphone_sessions) + "_FeaturesExtracted_FeaturesSelected-HypothesisDrivenApproach_Cleaned.csv"

        elif combination_sensors == ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]:
            path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/data_driven_approach/label_human motion_" + str(
                combination_sensors) + "-merged-to-GPSFeatures_feature-segment-" + str(feature_segment) + "_min-gps-accuracy-" + str(
                min_gps_accuracy) + "_only-active-smartphone-sessions-" + str(
                only_active_smartphone_sessions) + "_FeaturesExtracted_Merged_Selected.csv"

        t0 = time.time()
        print("start of combination_sensors: " + str(combination_sensors) + " and parameter_segment: " + str(feature_segment))
        path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/"+ str(combination_sensors)+"-GPS_feature-segment-"+ str(feature_segment) +"s/"
        if not os.path.exists(path_storage):
            os.makedirs(path_storage)
        if path_dataset.endswith(".csv"):
            df = pd.read_csv(path_dataset)
        elif path_dataset.endswith(".pkl"):
            with open(path_dataset, "rb") as f:
                df = pickle.load(f)


        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Dataset loaded",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df = df.drop(columns=drop_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

        #drop NaN in device_id:
        #TODO: probably error that there are still NaN in device_id -> probably from Merging step -> resolve if time
        df = df.dropna(subset=["device_id"])
        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in device_id dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #drop NaN in label_column_name
        df = df.dropna(subset=[label_column_name])
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in label_column_name dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)
        df = Merge_Transform.merge_participantIDs(df, users)  # temporary: merge participant ids

        # drop rows with NaN: will be many, as after merging of high-freq with GPS, NaN have not be dropped yet
        df = df.dropna()
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in any column dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #combine label classes if necessary
        if label_mapping != None:
            df = df.reset_index(drop=True)
            for mapping in label_mapping:
                df.loc[df["label_human motion"] == mapping[0], "label_human motion"] = mapping[1]
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Label Classes which are not included are dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        # selects only features which are calculated in the label_segment around event.
        ## Note: updated to code so that it actually only selects features for which the WHOLE FEATURE SEGMENT is in the label_segment around event
        df_decisionforest = df[(df['timestamp'] >= (df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2)))) & (df['timestamp'] <= (df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2))- pd.Timedelta(seconds=feature_segment)))]
        df_decisionforest = df_decisionforest.reset_index(drop=True)
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "data outside of label segment around events dropped",
                                            "Number Participants": len(df_decisionforest["device_id"].unique()),
                                            "Number Events": len(df_decisionforest["ESM_timestamp"].unique()),
                                            "Number Samples": len(df_decisionforest),
                                            "Number Features": len(df_decisionforest.columns) - 1}, ignore_index=True)

        print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")

        # check if df_decisionforest is empty
        if df_decisionforest.empty:
            print("df_decisionforest is empty")
            continue

        #run DF
        df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, n_permutations, path_storage, feature_importance = feature_importance, confusion_matrix_order = label_classes,  parameter_tuning = parameter_tuning, parameter_set = parameter_set)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Feature Segments"] = str(feature_segment)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        print("Finished Combination: " + str(combination_sensors) + "and timeperiod: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")

df_analytics.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "s_parameter_tuning-" + parameter_tuning + "_analytics.csv")
df_decisionforest_results_all.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")

#endregion

#region visualize results
# visualizing performances for 2 label segments x 11 feature segment-dataset combinations for balanced accuracy
label_segments = [45, 90]
parameter_tuning = "no"
fig, ax = plt.subplots(figsize=(10, 5))
for label_segment in label_segments:
    df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")
    # add dataset description depending on value in "Sensor Combination" and "Feature Segments"
    df_results["Feature Set and Feature Segment"] = ""
    for index, row in df_results.iterrows():
        if row["Sensor Combination"] == "['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']":
            if row["Feature Segments"] == 60:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (60s)"
            elif row["Feature Segments"] == 30:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (30s)"
            elif row["Feature Segments"] == 10:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (10s)"
            elif row["Feature Segments"] == 5:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (5s)"
            elif row["Feature Segments"] == 2:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (2s)"
            elif row["Feature Segments"] == 1:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (1s)"
        elif row["Sensor Combination"] == "['linear_accelerometer', 'rotation']":
            if row["Feature Segments"] == 60:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (60s)"
            elif row["Feature Segments"] == 30:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (30s)"
            elif row["Feature Segments"] == 10:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (10s)"
            elif row["Feature Segments"] == 5:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (5s)"
            elif row["Feature Segments"] == 2:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (2s)"
            elif row["Feature Segments"] == 1:
                df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (1s)"
    # reorder the rows: first hypothesis-driven, then data-driven; and with those, ascending feature segments
    df_results = df_results.sort_values(by=["Feature Segments"])
    df_results = df_results.sort_values(by=["Sensor Combination"], ascending=False)
    label = str(label_segment) + "s Label Segment"
    if label_segment == 90:
        label = "95s Label Segment"
    # visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
    #sns.set_style("whitegrid")
    #sns.set_context("paper", font_scale=1.5)
    #sns.set_palette("colorblind")
    #create sns lineplot
    sns.lineplot(x="Feature Set and Feature Segment", y="Balanced Accuracy", data=df_results, label=label, ax=ax)
plt.title("Model Performance of Different Datasets, Feature Segments, and Label Segments")
plt.xlabel("Dataset and Feature-Segment")
plt.ylabel("Balanced Accuracy")
plt.legend()
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#set y-axis limits
plt.ylim(0, 1)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.show()
#save with bbox_inches="tight" to avoid cutting off x-axis labels
fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segments) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall_visualization.png",
            bbox_inches="tight", dpi= 600)




# visualizing performances of the different DF (balanced accuracy & F1 score)
label_segment = 90
parameter_tuning = "no"
df_results = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")
# add dataset description depending on value in "Sensor Combination" and "Feature Segments"
df_results["Feature Set and Feature Segment"] = ""
for index, row in df_results.iterrows():
    if row["Sensor Combination"] == "['linear_accelerometer', 'gyroscope', 'magnetometer', 'rotation', 'accelerometer']":
        if row["Feature Segments"] == 60:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (60s)"
        elif row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (30s)"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (10s)"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (5s)"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (2s)"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Data-Driven (1s)"
    elif row["Sensor Combination"] == "['linear_accelerometer', 'rotation']":
        if row["Feature Segments"] == 60:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (60s)"
        elif row["Feature Segments"] == 30:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (30s)"
        elif row["Feature Segments"] == 10:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (10s)"
        elif row["Feature Segments"] == 5:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (5s)"
        elif row["Feature Segments"] == 2:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (2s)"
        elif row["Feature Segments"] == 1:
            df_results.loc[index, "Feature Set and Feature Segment"] = "Hypothesis-Driven (1s)"
# visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(10, 5))
#visualize Balanced Accuracy and F1 Score for "Feature Set and Feature Segment" as two lines in a SNS lineplot; use pandas.melt
sns.lineplot(x="Feature Set and Feature Segment", y="value", hue="variable", data=pd.melt(df_results, id_vars=["Feature Set and Feature Segment"], value_vars=["Balanced Accuracy", "F1"]))
plt.title("Model Performance of Different Datasets and Feature Segments")
plt.xlabel("Dataset and Feature-Segment")
plt.ylabel("Score")
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#set y-axis limits
plt.ylim(0, 1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.show()
#save with bbox_inches="tight" to avoid cutting off x-axis labels
plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall_visualization.png",
            bbox_inches="tight", dpi= 300)
#endregion
#endregion

#region hyperparameter tune the best model from previous comparison step
feature_segments = [10] #in seconds; define length of segments of parameters (over which duration they have been created)
combinations_sensors = [["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]]
min_gps_accuracy = 35
only_active_smartphone_sessions = "yes"
label_column_name = "label_human motion"
n_permutations = 0 # define number of permutations; better 1000
label_segment = 45 #define how much data around each event will be considered
# if label classes should be joint -> define in label mapping
label_mapping = [["lying (in one hand)", "lying (hand/s)"],
                            ["lying (in two hands)", "lying (hand/s)"],
                            ["lying (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                         ["standing (in one hand)", "standing (hand/s)"],
                            ["standing (in two hands)", "standing (hand/s)"],
                               ["standing (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                            ["sitting: at a table (in one hand)", "sitting (hand/s)"],
                               ["sitting: at a table (in two hands)", "sitting (hand/s)"],
                            ["sitting: at a table (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                         ["sitting: on the couch (in one hand)", "sitting (hand/s)"],
                            ["sitting: on the couch (in two hands)", "sitting (hand/s)"],
                         ["sitting: somewhere else (in one hand)", "sitting (hand/s)"],
                            ["sitting: somewhere else (in two hands)", "sitting (hand/s)"],
                         ["walking (in one hand)", "walking (hand/s)"],
                            ["walking (in two hands)", "walking (hand/s)"],
                         ["cycling (in one hand)", "cycling (hand/s)"],
                            ["cycling (in two hands)", "cycling (hand/s)"]
                         ]
label_classes = ["lying (hand/s)", "sitting (hand/s)", "standing (hand/s)",
                 "standing, sitting, or lying (on flat surface)",
                 "walking (hand/s)" ] # which label classes should be considered
parameter_tuning = "yes" # if yes: parameter tuning is performed; if False: default parameters are used
drop_cols = ["Unnamed: 0"] # columns that should be dropped
feature_importance = "shap"

# set CV mode
combine_participants = True
participants_test_set = [2,4,6,9] #this test set was used for (feature segment = 30s and label segment = 90s) and
# (feature segment = 10s and label segment =45s) , as these participants all include most classes while containing
# around 20-30% of all data in both cases


parameter_set = {
    "n_jobs": 4,# use all cores
    "random_state": 11 # set random state for reproducibility
}

# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 100
max_depth = [2, 5, 15, None] # default None
min_samples_split = [2, 10, 30] # default 2
min_samples_leaf = [1, 5] # default 1
max_features = ["sqrt", 5, "log2", 20, None] # default "sqrt"
oob_score = [True, False] # default False;
class_weight = ["balanced", "balanced_subsample", None] # default None
criterion =[ "gini", "entropy", "log_loss"] # default "gini"
max_samples = [None, 0.5, 0.8] # default None which means that 100% of the samples are used for each tree
bootstrap = [True]

# smaller parameter tuning set in order to have less runtime
n_estimators = [500]  # default 100
max_depth = [15] # default None
min_samples_split = [30] # default 2
min_samples_leaf = [5] # default 1
max_features = [None] # default "sqrt"
oob_score = [True] # default False;
class_weight = ["balanced"] # default None
criterion =[ "gini"] # default "gini"
max_samples = [None] # default None which means that 100% of the samples are used for each tree
bootstrap = [True]

# smaller parameter tuning set in order to have less runtime
n_estimators = [100, 500]  # default 100
max_depth = [15, None] # default None
min_samples_split = [2, 30] # default 2
min_samples_leaf = [1, 3] # default 1
max_features = ["sqrt", None] # default "sqrt"
oob_score = [False, True] # default False;
class_weight = ["balanced", "balanced_subsample"] # default None
criterion =[ "gini", "entropy"] # default "gini"
max_samples = [None, 0.8] # default None which means that 100% of the samples are used for each tree
bootstrap = [True]

grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score = oob_score, class_weight = class_weight,
                            criterion = criterion, max_samples = max_samples, bootstrap = bootstrap)

# select only data which are in the label_segment around ESM_event & drop columns
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Feature Segments"])
df_analytics = pd.DataFrame(columns=["Sensor Combination", "Feature Segment", "Event Description", "Number Participants", "Number Events", "Number Samples", "Number Features"])
for combination_sensors in combinations_sensors:

    for feature_segment in feature_segments:
        # since no dataset yet for combination_sensors = and feature_segments = 1
        if combination_sensors == ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"] and feature_segment == 1:
            print("no dataset yet")
            continue

        # create path to data based on sensor combination and feature segment
        if combination_sensors == ["linear_accelerometer", "rotation"]:
            path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/hypothesis_driven_approach/" + str(
                combination_sensors) + "_timeperiod-" + str(feature_segment) + "_min-gps-accuracy-" + str(
                min_gps_accuracy) + "_only-active-smartphone-sessions-" + str(
                only_active_smartphone_sessions) + "_FeaturesExtracted_FeaturesSelected-HypothesisDrivenApproach_Cleaned.csv"

        elif combination_sensors == ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]:
            path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/data_driven_approach/label_human motion_" + str(
                combination_sensors) + "-merged-to-GPSFeatures_feature-segment-" + str(feature_segment) + "_min-gps-accuracy-" + str(
                min_gps_accuracy) + "_only-active-smartphone-sessions-" + str(
                only_active_smartphone_sessions) + "_FeaturesExtracted_Merged_Selected.csv"

        t0 = time.time()
        print("start of combination_sensors: " + str(combination_sensors) + " and parameter_segment: " + str(feature_segment))
        path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/"+ str(combination_sensors)+"-GPS_feature-segment-"+ str(feature_segment) +"s/"
        if not os.path.exists(path_storage):
            os.makedirs(path_storage)
        if path_dataset.endswith(".csv"):
            df = pd.read_csv(path_dataset)
        elif path_dataset.endswith(".pkl"):
            with open(path_dataset, "rb") as f:
                df = pickle.load(f)

        # if parameter_tuning == "yes": save the grid_search_space
        if parameter_tuning == "yes":
            # save with pickle
            with open(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning +"_grid_search_space.pkl", "wb") as f:
                pickle.dump(grid_search_space, f)

        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Dataset loaded",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df = df.drop(columns=drop_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

        #drop NaN in device_id:
        #TODO: probably error that there are still NaN in device_id -> probably from Merging step -> resolve if time
        df = df.dropna(subset=["device_id"])
        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in device_id dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #drop NaN in label_column_name
        df = df.dropna(subset=[label_column_name])
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in label_column_name dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)
        df = Merge_Transform.merge_participantIDs(df, users)  # temporary: merge participant ids

        #in case commbine_participants == "yes", merge participant IDs into either 1 (=train) or 2 (=test)
        if combine_participants == True:
            # iterate through all IDs in "device_id" column and replace with 1 or 2: if ID is in participants_test_set, replace with 2, else replace with 1
            df["device_id_traintest"] = df["device_id"].apply(lambda x: 2 if x in participants_test_set else 1)

            # update analytics
            df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                                "Feature Segment": str(feature_segment),
                                                "Event Description": "Training and Test Participants Merged",
                                                "Number Participants": len(df["device_id_traintest"].unique()),
                                                "Number Events": len(df["ESM_timestamp"].unique()),
                                                "Number Samples": len(df),
                                                "Number Features": len(df.columns) - 1}, ignore_index=True)

        # drop rows with NaN: will be many, as after merging of high-freq with GPS, NaN have not be dropped yet
        df = df.dropna()
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in any column dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #combine label classes if necessary
        if label_mapping != None:
            df = df.reset_index(drop=True)
            for mapping in label_mapping:
                df.loc[df["label_human motion"] == mapping[0], "label_human motion"] = mapping[1]
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Label Classes which are not included are dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        # selects only features which are calculated in the label_segment around event.
        ## Note: updated to code so that it actually only selects features for which the WHOLE FEATURE SEGMENT is in the label_segment around event
        df_decisionforest = df[(df['timestamp'] >= (df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2)))) & (df['timestamp'] <= (df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2))- pd.Timedelta(seconds=feature_segment)))]
        df_decisionforest = df_decisionforest.reset_index(drop=True)
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "data outside of label segment around events dropped",
                                            "Number Participants": len(df_decisionforest["device_id"].unique()),
                                            "Number Events": len(df_decisionforest["ESM_timestamp"].unique()),
                                            "Number Samples": len(df_decisionforest),
                                            "Number Features": len(df_decisionforest.columns) - 1}, ignore_index=True)

        print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")

        # check if df_decisionforest is empty
        if df_decisionforest.empty:
            print("df_decisionforest is empty")
            continue

        #run DF
        df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name,
                                                              n_permutations, path_storage, feature_importance = feature_importance,
                                                              confusion_matrix_order = label_classes,  parameter_tuning = parameter_tuning,
                                                              parameter_set = parameter_set, grid_search_space = grid_search_space, combine_participants = combine_participants)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Feature Segments"] = str(feature_segment)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        print("Finished Combination: " + str(combination_sensors) + "and timeperiod: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")


#df_analytics.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "s_parameter_tuning-" + parameter_tuning + "_analytics.csv")
#df_decisionforest_results_all.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")


#endregion

#region best model training: train the best model (based on best dataset and best parameters determined before)
#region hyperparameter tune the best model from previous comparison step
feature_segments = [30] #in seconds; define length of segments of parameters (over which duration they have been created)
combinations_sensors = [["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]]
min_gps_accuracy = 35
only_active_smartphone_sessions = "yes"
label_column_name = "label_human motion"
n_permutations = 50 # define number of permutations; better 1000
label_segment = 90 #define how much data around each event will be considered
# if label classes should be joint -> define in label mapping
label_mapping = [["lying (in one hand)", "lying (hand/s)"],
                            ["lying (in two hands)", "lying (hand/s)"],
                            ["lying (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                         ["standing (in one hand)", "standing (hand/s)"],
                            ["standing (in two hands)", "standing (hand/s)"],
                               ["standing (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                            ["sitting: at a table (in one hand)", "sitting (hand/s)"],
                               ["sitting: at a table (in two hands)", "sitting (hand/s)"],
                            ["sitting: at a table (on flat surface)", "standing, sitting, or lying (on flat surface)"],
                         ["sitting: on the couch (in one hand)", "sitting (hand/s)"],
                            ["sitting: on the couch (in two hands)", "sitting (hand/s)"],
                         ["sitting: somewhere else (in one hand)", "sitting (hand/s)"],
                            ["sitting: somewhere else (in two hands)", "sitting (hand/s)"],
                         ["walking (in one hand)", "walking (hand/s)"],
                            ["walking (in two hands)", "walking (hand/s)"],
                         ["cycling (in one hand)", "cycling (hand/s)"],
                            ["cycling (in two hands)", "cycling (hand/s)"]
                         ]
label_classes = ["lying (hand/s)", "sitting (hand/s)", "standing (hand/s)",
                 "standing, sitting, or lying (on flat surface)",
                 "walking (hand/s)" ] # which label classes should be considered
parameter_tuning = "no" # if yes: parameter tuning is performed; if False: default parameters are used
drop_cols = ["Unnamed: 0"] # columns that should be dropped
feature_importance = "shap"

# set CV mode
combine_participants = False
participants_test_set = [2,4,6,9] #this test set was used for (feature segment = 30s and label segment = 90s) and
# (feature segment = 10s and label segment =45s) , as these participants all include most classes while containing
# around 20-30% of all data in both cases

# use parameter set which performed best in hyperparameter tuning
parameter_set = {
    "n_estimators": 500,
    "criterion": "gini",
    "max_depth": 15,
    "min_samples_split": 30,
    "min_samples_leaf": 5,
    "max_features": None,
    "oob_score": True,
    "n_jobs": -1,
    "class_weight": "balanced",
    "random_state": 11
}

grid_search_space = None

# select only data which are in the label_segment around ESM_event & drop columns
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Feature Segments"])
df_analytics = pd.DataFrame(columns=["Sensor Combination", "Feature Segment", "Event Description", "Number Participants", "Number Events", "Number Samples", "Number Features"])
for combination_sensors in combinations_sensors:

    for feature_segment in feature_segments:
        # since no dataset yet for combination_sensors = and feature_segments = 1
        if combination_sensors == ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"] and feature_segment == 1:
            print("no dataset yet")
            continue

        # create path to data based on sensor combination and feature segment
        if combination_sensors == ["linear_accelerometer", "rotation"]:
            path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/hypothesis_driven_approach/" + str(
                combination_sensors) + "_timeperiod-" + str(feature_segment) + "_min-gps-accuracy-" + str(
                min_gps_accuracy) + "_only-active-smartphone-sessions-" + str(
                only_active_smartphone_sessions) + "_FeaturesExtracted_FeaturesSelected-HypothesisDrivenApproach_Cleaned.csv"

        elif combination_sensors == ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]:
            path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/data_driven_approach/label_human motion_" + str(
                combination_sensors) + "-merged-to-GPSFeatures_feature-segment-" + str(feature_segment) + "_min-gps-accuracy-" + str(
                min_gps_accuracy) + "_only-active-smartphone-sessions-" + str(
                only_active_smartphone_sessions) + "_FeaturesExtracted_Merged_Selected.csv"

        t0 = time.time()
        print("start of combination_sensors: " + str(combination_sensors) + " and parameter_segment: " + str(feature_segment))
        path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/best_performing_models/"+ str(combination_sensors)+"-GPS_feature-segment-"+ str(feature_segment) +"s/"
        if not os.path.exists(path_storage):
            os.makedirs(path_storage)
        if path_dataset.endswith(".csv"):
            df = pd.read_csv(path_dataset)
        elif path_dataset.endswith(".pkl"):
            with open(path_dataset, "rb") as f:
                df = pickle.load(f)

        # if parameter_tuning == "yes": save the grid_search_space
        if parameter_tuning == "yes":
            # save with pickle
            with open(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning +"_grid_search_space.pkl", "wb") as f:
                pickle.dump(grid_search_space, f)
        else:
            #save the parameter_set
            with open(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning +"_parameter_set.pkl", "wb") as f:
                pickle.dump(parameter_set, f)

        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Dataset loaded",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        df = df.drop(columns=drop_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])

        #drop NaN in device_id:
        #TODO: probably error that there are still NaN in device_id -> probably from Merging step -> resolve if time
        df = df.dropna(subset=["device_id"])
        #update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in device_id dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #drop NaN in label_column_name
        df = df.dropna(subset=[label_column_name])
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in label_column_name dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)
        df = Merge_Transform.merge_participantIDs(df, users)  # temporary: merge participant ids

        #in case commbine_participants == "yes", merge participant IDs into either 1 (=train) or 2 (=test)
        if combine_participants == True:
            # iterate through all IDs in "device_id" column and replace with 1 or 2: if ID is in participants_test_set, replace with 2, else replace with 1
            df["device_id_traintest"] = df["device_id"].apply(lambda x: 2 if x in participants_test_set else 1)

            # update analytics
            df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                                "Feature Segment": str(feature_segment),
                                                "Event Description": "Training and Test Participants Merged",
                                                "Number Participants": len(df["device_id_traintest"].unique()),
                                                "Number Events": len(df["ESM_timestamp"].unique()),
                                                "Number Samples": len(df),
                                                "Number Features": len(df.columns) - 1}, ignore_index=True)

        # drop rows with NaN: will be many, as after merging of high-freq with GPS, NaN have not be dropped yet
        df = df.dropna()
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "NaN in any column dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        #combine label classes if necessary
        if label_mapping != None:
            df = df.reset_index(drop=True)
            for mapping in label_mapping:
                df.loc[df["label_human motion"] == mapping[0], "label_human motion"] = mapping[1]
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "Label Classes which are not included are dropped",
                                            "Number Participants": len(df["device_id"].unique()),
                                            "Number Events": len(df["ESM_timestamp"].unique()),
                                            "Number Samples": len(df),
                                            "Number Features": len(df.columns) - 1}, ignore_index=True)

        # selects only features which are calculated in the label_segment around event.
        ## Note: updated to code so that it actually only selects features for which the WHOLE FEATURE SEGMENT is in the label_segment around event
        df_decisionforest = df[(df['timestamp'] >= (df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2)))) & (df['timestamp'] <= (df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2))- pd.Timedelta(seconds=feature_segment)))]
        df_decisionforest = df_decisionforest.reset_index(drop=True)
        # update analytics
        df_analytics = df_analytics.append({"Sensor Combination": str(combination_sensors),
                                            "Feature Segment": str(feature_segment),
                                            "Event Description": "data outside of label segment around events dropped",
                                            "Number Participants": len(df_decisionforest["device_id"].unique()),
                                            "Number Events": len(df_decisionforest["ESM_timestamp"].unique()),
                                            "Number Samples": len(df_decisionforest),
                                            "Number Features": len(df_decisionforest.columns) - 1}, ignore_index=True)

        print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")

        # check if df_decisionforest is empty
        if df_decisionforest.empty:
            print("df_decisionforest is empty")
            continue

        #run DF
        df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name,
                                                              n_permutations, path_storage, feature_importance = feature_importance,
                                                              confusion_matrix_order = label_classes,  parameter_tuning = parameter_tuning,
                                                              parameter_set = parameter_set, grid_search_space = grid_search_space, combine_participants = combine_participants)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Feature Segments"] = str(feature_segment)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        print("Finished Combination: " + str(combination_sensors) + "and timeperiod: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")


#df_analytics.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "s_parameter_tuning-" + parameter_tuning + "_analytics.csv")
#df_decisionforest_results_all.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")


#endregion
#endregion




##region Decision Forest OLD
### initialize parameters
parameter_segments = [10, 5,2,1] #in seconds; define length of segments of parameters (over which duration they have been created)
combinations_sensors = [["highfrequencysensors-all", "GPS"],
                        ["linear_accelerometer", "rotation", "GPS"],
                        ["linear_accelerometer", "GPS"]]  # define which sensor combinations should be considered
combinations_sensors = [["linear_accelerometer", "rotation", "GPS"]]

label_column_name = "label_human motion - general"
n_permutations = 100 # define number of permutations; better 1000
label_segment = 11 #define how much data around each event will be considered
label_classes = ["standing", "lying", "sitting", "walking", "cycling"] # which label classes should be considered
parameter_tuning = "yes" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = ["Unnamed: 0", "1", "3", "loc_accuracy", "loc_provider", "loc_device_id", "timestamp_merged"] # columns that should be dropped

# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 100
max_depth = [2, 5, 15] # default None
min_samples_split = [2, 10, 30] # default 2
min_samples_leaf = [1, 5] # default 1
max_features = ["sqrt", 5, "log2", 20, None] # default "sqrt"
oob_score = [True, False] # default False;
class_weight = ["balanced", "balanced_subsample", None] # default None
criterion =
max_samples = [None, 0.5, 0.8] # default None which means that 100% of the samples are used for each tree

grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features)


# select only data which are in the label_segment around ESM_event & drop columns
df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1",
                                             "Precision", "Recall", "Sensor Combination", "Parameter Segments"])
for combination_sensors in combinations_sensors:
    for parameter_segment in parameter_segments:
        t0 = time.time()
        print("start of combination_sensors: " + str(combination_sensors) + " and parameter_segment: " + str(parameter_segment))
        path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/sensor-"+ combination_sensors[0]+"("+ str(parameter_segment) +"seconds)-and-GPS(1seconds)/"
        if len(combination_sensors) == 3:
            path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/sensor-"+ combination_sensors[0]+"("+ str(parameter_segment) +"seconds)-"+combination_sensors[1] +"("+ str(parameter_segment) +"seconds)-and-GPS(1seconds)/"
        if not os.path.exists(path_storage):
            os.makedirs(path_storage)
        path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-"+ label_column_name +"_sensor-"+ combination_sensors[0]+"("+ str(parameter_segment) +"seconds)-and-GPS(1seconds).pkl"
        if len(combination_sensors) == 3:
            path_dataset = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-"+ label_column_name +"_sensor-"+ combination_sensors[0]+"("+ str(parameter_segment) +"seconds)-"+combination_sensors[1] +"("+ str(parameter_segment) +"seconds)-and-GPS(1seconds).pkl"
        df = pd.read_pickle(path_dataset)
        if len(combination_sensors) == 3: #this is necessary to make the code work for the case with three merged sensors
            df = df.rename(columns={"timestamp_merged_x": "timestamp_merged", "label_human motion - general_x": "label_human motion - general"})
            df = df.drop(columns=["timestamp_merged_y", "label_human motion - general_y"])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
        df = df.drop(columns=drop_cols)
        df = Merge_Transform.merge_participantIDs(df, users)  # temporary: merge participant ids
        df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

        # selects only features which are calculated in the label_segment around event.
        ## Note: updated to code so that it actually only selects features for which the WHOLE FEATURE SEGMENT is in the label_segment around event
        df_decisionforest = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2))) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2)))]
        print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
        # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
        df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts["total"] = df_label_counts.sum(axis=1)
        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
        df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")
        df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, n_permutations, path_storage, parameter_tuning, grid_search_space)
        df_decisionforest_results["Sensor Combination"] = str(combination_sensors)
        df_decisionforest_results["Parameter Segments"] = str(parameter_segment)

        df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
        df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(
            label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
        print("Finished Combination: " + str(combination_sensors) + "and timeperiod: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")
df_decisionforest_results_all.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/parameter_tuning-" + parameter_tuning + "_results_overall.csv")

#endregion

#region LSTM

#endregion

#endregion

#endregion
#endregion


#region public transport
## what can be improved?
#TODO add also motion features
#TODO train LSTM on GPS data (and maybe motion data)
#TODO add further control-group labels

#region data exploration for public transport

#region data exploration labels
label_column_name = "label_public transport"
df_labels = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/labels/"

## create table with users x relevant ESM-answers
df_label_counts = df_labels.groupby("bodyposition")["location"].value_counts().unstack().fillna(0)
df_label_counts["total"] = df_label_counts.sum(axis=1)
df_label_counts.loc["total"] = df_label_counts.sum(axis=0)

## create barplot with label classes: stationary, walking, running, cycling, public transport, car
df_labels_publictransport = df_labels.copy()
df_labels_publictransport["label_public transport"] = df_labels_publictransport["label_public transport"].replace("train", "public transport")
df_labels_publictransport = df_labels_publictransport[~df_labels_publictransport["label_public transport"].isin(["car/bus/train/tram"])]
fig = data_exploration_labels.visualize_esm_activity(df_labels_publictransport, "label_public transport", "Transportation Mode - Number of Answers per Class")
fig.savefig(path_storage + "public_transport_publictransport-and-train-combined_car-bus-train-deleted.png")

##create table with users x label classes
df_labels_publictransport =  Merge_Transform.merge_participantIDs(df_labels_publictransport, users, include_cities = True)
df_label_counts = data_exploration_labels.create_table_user_activity(df_labels_publictransport, "label_public transport")
df_label_counts.to_csv(path_storage + "public_transport_publictransport-and-train-combined_car-bus-train-deleted_participant-labels-overview.csv")

#endregion

#region data exploration sensors

# find out how many labels have sensor data -> ##temporary -> make later into function!
## NOTE: this subsection was adapted using the new function; in the documentation, still older results are shown (i.e. without
## applying a "minimum_sensordata_percentage" threshold; maybe adapt the documentation!
segment_around_events = 90 # timeperiod considered around events
min_sensordata_percentage = 0.0001 #in percent; only sensordata above this threshold will be considered
gps_accuracy_min = 35 # in meters; only gps data with accuracy below this threshold was considered when counting GPS records for each event
label_column_name = "label_public transport"
df_esm_including_number_sensordata = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/esm-events_with-number-of-sensorrecords_segment-around-events-" + str(segment_around_events) + "_gps-accuracy-min-" + str(gps_accuracy_min) + " .csv")
# drop row with "nan" in ESM_timestamp
df_esm_including_number_sensordata = df_esm_including_number_sensordata[df_esm_including_number_sensordata["ESM_timestamp"].notnull()]
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
sensors_included = ["linear_accelerometer", "locations"]
df_esm_including_sensordata_above_threshold, df_esm_user_class_counts = data_exploration_labels.create_table_user_activity_including_sensordata(df_esm_including_number_sensordata, dict_label, label_column_name , sensors_included, segment_around_events)

#combine public transport and train and #delete car/bus/train/tram columns in df_esm_user_class_counts
if "train" in df_esm_user_class_counts.columns:
    df_esm_user_class_counts["public transport"] = df_esm_user_class_counts["public transport"] + df_esm_user_class_counts["train"]
    df_esm_user_class_counts = df_esm_user_class_counts.drop(columns=["train", "car/bus/train/tram", "total"])
    df_esm_user_class_counts = df_esm_user_class_counts.drop(index="total")
    # sum axis 1 without colum "User ID"
    df_esm_user_class_counts["total"] = df_esm_user_class_counts.iloc[:,1:].sum(axis=1)
    df_esm_user_class_counts.loc["total"] = df_esm_user_class_counts.sum(axis=0)
    df_esm_including_sensordata_above_threshold["label_public transport"] = df_esm_including_sensordata_above_threshold["label_public transport"].replace("train", "public transport")
    df_esm_including_sensordata_above_threshold = df_esm_including_sensordata_above_threshold[~df_esm_including_sensordata_above_threshold["label_public transport"].isin(["car/bus/train/tram"])]
else:
    df_esm_user_class_counts = df_esm_user_class_counts.drop(columns=["car/bus/train/tram", "total"])
    df_esm_user_class_counts = df_esm_user_class_counts.drop(index="total")
    df_esm_user_class_counts["total"] = df_esm_user_class_counts.sum(axis=1)
    df_esm_user_class_counts.loc["total"] = df_esm_user_class_counts.sum(axis=0)
    df_esm_including_sensordata_above_threshold = df_esm_including_sensordata_above_threshold[~df_esm_including_sensordata_above_threshold["label_public transport"].isin(["car/bus/train/tram"])]
print("Number of Events for which all Sensor Data is available: " + str(len(df_esm_including_sensordata_above_threshold)))
#df_esm_user_class_counts.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_human motion/human_motion_user-class-counts_event-segments-" + str(segment_around_events) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".csv")

path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/labels/"
df_esm_user_class_counts.to_csv(path_storage + "public_transport_publictransport-and-train-combined_car-bus-train-deleted_participant-labels-overview-for-which-linacc-and-gps-available_and-after-noise-cleaning.csv")


# create plots for all events visualizing linear accelerometer & GPS features (speed & acceleration)
label_column_name = "label_public transport"
time_period = 600  # seconds; timeperiod around event which should be visualized
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_public transport/"
path_sensordata = "/Users/benediktjordan/Downloads/INSERT_esm_timeperiod_5 min.csv_JSONconverted.pkl"
axs_limitations = "general"
label_data = True
gps_accuracy_min = 10 # meters
figure_title = ""
list_sensors = [["Linear Accelerometer", "linear_accelerometer"],
                ["GPS", ""]]
list_sensors = [["GPS", ""]]
event_times = []
for key in dict_label.keys():
    event_times.append(key)
num_events = 1
sensor_names = ""
for sensor in list_sensors:
    sensor_names += sensor[0] + "_"
path_GPS_features = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(gps_accuracy_min) + ".csv"

for event_time in event_times:
    time0 = time.time()
    # TEMPORARY check if figure has already been saved; if yes, skip
    #list_files = []
    #for root, dirs, files in os.walk(path_to_save):
    #    for file in files:
    #        if file.endswith(".png"):
    #            list_files.append(os.path.join(root, file))
    # check if event_time is already in any element of list_files
    #if any(str(event_time) in s for s in list_files):
    #    print("Figure already saved for event_time: " + str(event_time))
    #    num_events += 1
    #    continue

    fig, activity_name = data_exploration_sensors.vis_sensordata_around_event_severalsensors(list_sensors, event_time, time_period,
                                                                           label_column_name, path_sensordata, axs_limitations, label_data = label_data,
                                                                                             path_GPS_features = path_GPS_features)
    # check if fig is None: if yes, continue with next event_time
    if fig is None:
        num_events += 1
        continue

    # find out if path exists
    #NOte: reason for this is that there was a problem with this case
    if activity_name == "car/bus/train/tram":
        activity_name = "car-bus-train-tram"

    # if path_to save doesnt exist, create it with mkdir
    if not os.path.exists(path_to_save + activity_name + "/"):
        os.makedirs(path_to_save + activity_name + "/")

    fig.savefig(
        path_to_save + activity_name + "/gps-accuracy-min-" + str(gps_accuracy_min) + "_" + activity_name + "_EventTime-" + str(event_time) + "_Segment-" + str(
            time_period) + "_Sensors-" + sensor_names + ".png")
    plt.close(fig)
    # print progress
    print("Finished event " + str(num_events) + " of " + str(len(event_times)) + " in " + str((time.time() - time0)/60) + " minutes.")
    num_events += 1

# visualize mean/std x max x public transport class in scatterplot for GPS and accelerometer data
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
time_periods = [10,20,40, 90, 180]
list_activities = ["stationary", "walking", "public transport", "car"]
visualize_participants = "no"
gps_accuracy_min = 35 # minimum accuracy of GPS data to be included in analysis
sensors = ["linear_accelerometer", "GPS"]
sensors = ["GPS"]

for sensor in sensors:
    for time_period in time_periods:
        if sensor == "GPS":
            df_summary_stats = pd.read_csv(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/GPS_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + ".csv")
            # add label public transport
            df_summary_stats = labeling_sensor_df(df_summary_stats, dict_label, "label_public transport",
                                                  ESM_identifier_column="ESM_timestamp")
            fig = data_exploration_sensors.vis_summary_stats(df_summary_stats, "GPS", "label_public transport", list_activities,
                                    "Figure Title", visualize_participants)
            fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_public transport/GPS_mean_max_gps-accuracy-min-" + str(gps_accuracy_min) + "_activities-included-" + str(list_activities) + "_including-participants-" + visualize_participants + "_time-period-around-event-"
                        + str(time_period) + "s_scatterplot.png")
        else:
            df_summary_stats = pd.read_csv(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/" + sensor + "_summary-stats_time-period-around-event-" + str(time_period) + ".csv")
            # add label public transport
            df_summary_stats = labeling_sensor_df(df_summary_stats, dict_label, "label_public transport",
                                                  ESM_identifier_column="ESM_timestamp")
            fig = data_exploration_sensors.vis_summary_stats(df_summary_stats, "GPS", "label_public transport", list_activities,"Figure Title", visualize_participants)
            fig.savefig(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_public transport/" + sensor + "_std_max_activities-included-" + str(
                    list_activities) + "_including-participants-" + visualize_participants + "_time-period-around-event-"
                + str(time_period) + "s_scatterplot.png")


# UPDATED APPROACH 2: visualize mean x max x public transport class in scatterplot for GPS data BUT THIS TIME COMBINE ALL PUBLIC TRANSPORT CLASSES
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
time_periods = [10,20,40, 90, 180]
sensors = ["linear_accelerometer", "GPS"]
list_activities = ["stationary", "walking", "public transport", "car", "running", "cycling"]
visualize_participants = "no"
gps_accuracy_min = 35 # minimum accuracy of GPS data to be included in analysis
for sensor in sensors:
    for time_period in time_periods:
        if sensor == "GPS":
            df_summary_stats = pd.read_csv(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/GPS_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + ".csv")
            # add label public transport
            df_summary_stats = labeling_sensor_df(df_summary_stats, dict_label, "label_public transport",
                                                  ESM_identifier_column="ESM_timestamp")
            # combine all public transport classes
            df_summary_stats["label_public transport"] = df_summary_stats["label_public transport"].replace(
                ["train", "car/bus/train/tram"], "public transport")
            fig = data_exploration_sensors.vis_summary_stats(df_summary_stats, "GPS", "label_public transport", list_activities,
                                    "Mean and Speed of Transportation Mode Events", visualize_participants)
            fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_public transport/GPS_mean_max_gps-accuracy-min-" + str(gps_accuracy_min) + "_activities-included-" + str(list_activities) + "_including-participants-" + visualize_participants + "_time-period-around-event-"
                        + str(time_period) + "s_scatterplot.png")
        else:
            df_summary_stats = pd.read_csv(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/" + sensor + "_summary-stats_time-period-around-event-" + str(time_period) + ".csv")
            # add label public transport
            df_summary_stats = labeling_sensor_df(df_summary_stats, dict_label, "label_public transport", ESM_identifier_column="ESM_timestamp")
            # combine all public transport classes
            df_summary_stats["label_public transport"] = df_summary_stats["label_public transport"].replace(
                ["train", "car/bus/train/tram"], "public transport")
            fig = data_exploration_sensors.vis_summary_stats(df_summary_stats, sensor, "label_public transport", list_activities,"Mean and Maxima of the Linear Accelerometer of Transportation Mode Events", visualize_participants)
            fig.savefig(
                "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_public transport/" + sensor + "_std_max_activities-included-" + str(
                    list_activities) + "_including-participants-" + visualize_participants + "_time-period-around-event-"
                + str(time_period) + "s_scatterplot.png")

# get unique ESM_timestamp values
df_summary_stats["ESM_timestamp"].nunique()




#testarea for location data
df_locations = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration.csv")
#label data
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
df_locations = labeling_sensor_df(df_locations, dict_label, "label_public transport", ESM_identifier_column="ESM_timestamp")
# delete rows with nan values in label_public transport
df_locations = df_locations.dropna(subset=["label_public transport"])

#calculate summary stats for column accuracy
df_locations["loc_accuracy"].describe()

# only retain rows with accuracy < 35
df_locations_new = df_locations[df_locations["loc_accuracy"] < 35]
#count number of unique ESM_timestamps
df_locations["ESM_timestamp"].nunique()
df_locations_new["ESM_timestamp"].nunique()

#retain only data which is in 90s time period around ESM_timestamp
time_period = 90
#convert timestamp to datetime
df_locations_new["ESM_timestamp"] = pd.to_datetime(df_locations_new["ESM_timestamp"])
df_locations_new["timestamp"] = pd.to_datetime(df_locations_new["timestamp"])
df_locations_new = df_locations_new[(df_locations_new['timestamp'] >= df_locations_new['ESM_timestamp'] - pd.Timedelta(seconds=(time_period / 2))) & (
        df_locations_new['timestamp'] <= df_locations_new['ESM_timestamp'] + pd.Timedelta(seconds=(time_period / 2)))]

df_locations_new["ESM_timestamp"].nunique()
# take only the unique ESM_timestamps and count from them the number of label_public transport
df_locations_new_unique_ESM_timestamps = df_locations_new.drop_duplicates(subset=["ESM_timestamp"])
df_locations_new_unique_ESM_timestamps["label_public transport"].value_counts()





# add labels
df_locations = labeling_sensor_df(df_locations, dict_label, "label_public transport", ESM_identifier_column="ESM_timestamp")
#delete NaN values in label_public transport
df_locations = df_locations[df_locations["label_public transport"].notna()]
#get only data from user with loc_device_id = 590f0faf-d932-4a57-998d-e3da667a91dc
df_locations_user = df_locations[df_locations["loc_device_id"] == "b7b013b7-f78c-4325-a7ab-2dfc128fba27"]

#endregion


#region modeling transportation mode
#get overview over public transport labels
## create df from dict_label with location_labels
df_labels = pd.DataFrame.from_dict(dict_label, orient='index')
df_label_counts = df_labels.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
df_label_counts["total"] = df_label_counts.sum(axis=1)
df_label_counts.loc["total"] = df_label_counts.sum(axis=0)




## load data & some preprocessing
#timedelta = "1000ms" # the time which merged points can be apart; has to be in a format compatible with pd.TimeDelta()
#drop_cols = ['device_id', 'label_human motion - general', "ESM_timestamp", "label_human motion - general"] # columns that should be deleted from df_tomerge
df_gps_features = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/locations-aroundevents_features-distance-speed-acceleration.csv")
df_gps_features = Merge_Transform.merge_participantIDs(df_gps_features, users)#temporary: merge participant ids
df_gps_features = labeling_sensor_df(df_gps_features, dict_label, label_column_name = "label_public transport" , ESM_identifier_column = "ESM_timestamp")
df_gps_features = df_gps_features.dropna(subset=[label_column_name]) #drop NaN labels

#df_tomerge = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-label_human motion - general_highfrequencysensors-all_timeperiod-1 s_featureselection.pkl") # load df_tomerge: this should be the df with features with higher "sampling rate"/frequency
#df_final = Merge_and_Impute.merge(df_base, df_tomerge, timedelta, drop_cols) # merge df_base and df_tomerge
##TODO integrate "delete columns with only "NaN" values" into "features selection" section
df_gps_features = df_gps_features.dropna(axis=1, how='all') # delete columns with only "NaN" values
#print("So many rows contain missing values after merging: " + str(df_final.isnull().values.any()))
#df_final_nan = Merge_and_Impute.impute_deleteNaN(df_final) # delete rows which contain missing values
#df_final_nan.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-label_human motion - general_sensor-highfrequencysensorsAll(1seconds)-and-locations(1seconds).pkl")

## train Decision Forest
### initialize parameters
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_public transport/GPS_features/"
n_permutations = 2 # define number of permutations; better 1000
label_segments = [2, 5, 10, 20, 60] #in seconds; define length of segments for label
label_classes = ["public transport", "train", "walking/cycling"] # which label classes should be considered
parameter_tuning = "no" # if True: parameter tuning is performed; if False: best parameters are used
drop_cols = ["Unnamed: 0", "1", "2" ,  "3", "loc_accuracy", "loc_provider", "loc_double_altitude"] # columns that should be dropped

# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 500
max_depth = [2, 5, 15]
min_samples_split = [2, 10, 30]
min_samples_leaf = [1, 5]
max_features = ["sqrt", 3, "log2", 20]
grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features)

df = df_gps_features.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
df = df.drop(columns=drop_cols)
df = df[df[label_column_name].isin(label_classes)] # drop rows which don´t contain labels which are in label_classes
df = df.rename(columns={"loc_device_id": "device_id"})

# select only data which are in the label_segment around ESM_event & drop columns
for label_segment in label_segments:
    df_decisionforest = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2))) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2)))]
    print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
    # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
    df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
    df_label_counts["total"] = df_label_counts.sum(axis=1)
    df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
    df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")
    df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, grid_search_space, n_permutations, path_storage, parameter_tuning)
    df_decisionforest_results.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-"+ parameter_tuning + "s_df_results.csv")


#endregion

#endregion

#endregion

#region locations
## What to improve (current accuracy: 54% (with 33% random accuracy))
#TODO merge participant IDs and run model again
#TODO change classification of frequent labels: "other sleep place" is often actually in reality "home". How to deal with this?
#TODO look in detail into incorrect predictions: maybe need to delete "other frequent locations" label? Other improvements?
#TODO maybe cluster the labeled location data since there are several different locations for i.e. label "home" -> maybe some of them are mistakes?
#TODO use DBSCAN instead of KMeans to cluster locations (think first: why did Lennart propose this?)

#region data exploration for locations
#region data exploration labels
#region create table with users x relevant ESM-answers (without sensor data)
label_column_name = "label_location"
df_labels = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_exploration/labels/"
## merge participant IDs
df_labels = Merge_Transform.merge_participantIDs(df_labels, users, device_id_col = None, include_cities = True)
df_label_counts_participants_locations = data_exploration_labels.create_table_user_classes_eventcount(df_labels, label_column_name)
df_label_counts_participants_locations.to_csv(path_storage + "label_counts_participants_locations.csv")
#endregion

#region find out how many labels have sensor data: visualize as table and barplot
segment_around_events = 90 # timeperiod considered around events
min_sensordata_percentage = 50 #in percent; only sensordata above this threshold will be considered
gps_accuracy_min = 100 # in meters; only gps data with accuracy below this threshold was considered when counting GPS records for each event
label_column_name = "label_location"
only_active_smartphone_sessions = "yes"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_exploration/labels/"
df_esm_including_number_sensordata = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/esm-events_with-number-of-sensorrecords_segment-around-events-" + str(segment_around_events) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_gps-accuracy-min-" + str(gps_accuracy_min) + " .csv")
# drop row with "nan" in ESM_timestamp
df_esm_including_number_sensordata = df_esm_including_number_sensordata[df_esm_including_number_sensordata["ESM_timestamp"].notnull()]
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
sensors_included = ["locations", "sensor_wifi" ]
#sensors_included = ["locations" ]
#sensors_included = ["sensor_wifi" ]

## visualize as table
df_esm_including_sensordata_above_threshold, df_esm_user_class_counts = data_exploration_labels.create_table_user_activity_including_sensordata(df_esm_including_number_sensordata, dict_label, label_column_name , sensors_included, segment_around_events)
print("Number of Events for which all Sensor Data is available: " + str(len(df_esm_including_sensordata_above_threshold)))
df_esm_user_class_counts.to_csv(path_storage + "location_user-class-counts_event-segments-" + str(segment_around_events) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".csv")
## this is necessary for the following visualiztion of sample events: from this df will get the relevant ESM timestamps
df_esm_including_sensordata_above_threshold.to_csv(path_storage + "location_events_with-sensor-data_event-segments-" + str(segment_around_events) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".csv")

## visualize as barplot
fig = data_exploration_labels.visualize_esm_activity(df_esm_including_sensordata_above_threshold, "label_location", "Frequent Locations - Number of Answers per Class with Sensor Data")
fig.savefig(path_storage + "human_motion_class-counts_event-segments-" + str(segment_around_events) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".png")
#endregion
#endregion

#region data exploration sensors
#region visualize GPS: generic maps with locations for each participant
## based on all location data
df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/locations_all_JSON-transformed_merged-participantIDs.csv")
path_storage = '/Users/benediktjordan/Documents/MTS/Iteration01/location/data_exploration/sensors/GPS/'
for participant in df["loc_device_id"].unique():
    t0 = time.time()
    print("started with participant: " + str(participant))
    figure_title = "GPS Data of User " + str(participant)
    df_participant = df[df["loc_device_id"] == participant]
    fig = GPS_visualization.gps_utm_genericmap(df_participant, figure_title, colour_based_on_labels = "no")
    fig.savefig(path_storage + "GPSData_all_user_" + str(participant) + ".png", dpi=600, bbox_inches = "tight")
    print("finished with participant: " + str(participant) + " in " + str((time.time() - t0)/60) + " minutes")

#region based on 1 record per event & coloured regarding to location classes
label_column_name = "label_location"
colour_based_on_labels = "yes"
min_gps_accuracy = 100 # only use GPS points with accuracy below this value
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/locations_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl")
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
path_storage = '/Users/benediktjordan/Documents/MTS/Iteration01/location/data_exploration/sensors/GPS/'
df_locations = df_locations[df_locations["loc_accuracy"] < min_gps_accuracy] # delete all records with accuracy > min_gps_accuracy
df_locations = labeling_sensor_df(df_locations, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp") #add labels to df
df_locations = df_locations.dropna(subset=[label_column_name]) # drop NaN values in label column
df_locations = Merge_Transform.merge_participantIDs(df_locations, users, device_id_col = None, include_cities = False)# merge participant IDs
### keep for each event the GPS record with a timestamp closest to the ESM timestamp (first one after ESM timestamp)
df_locations = GPS_computations.first_sensor_record_after_event(df_locations)

df_locations = df_locations.rename(columns={label_column_name: "Location Classes"}) # rename label_column into "Location Classes" (necessary for visualization)
for participant in df_locations["device_id"].unique():
    t0 = time.time()
    print("started with participant: " + str(participant))
    figure_title = "GPS Data of Events for User " + str(participant)
    df_participant = df_locations[df_locations["device_id"] == participant]
    fig = GPS_visualization.gps_utm_genericmap(df_participant, figure_title, "Location Classes", colour_based_on_labels)
    fig.savefig(path_storage +  "GPSData_OnlyEvents_only-active-smartphon-seessions-yes" + "_min-gps-acc-" + str(min_gps_accuracy) +"_user_" + str(participant) + ".png", dpi=600, bbox_inches = "tight")
    print("finished with participant: " + str(participant) + " in " + str((time.time() - t0)/60) + " minutes")
#endregion
#endregion

# region visualize WiFi
#region all WiFi data: histograms of different WiFi BSSID for each participant
df_wifi = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/sensor_wifi_all.csv")
y_label = "Minutes Connected to This WiFi Network"
df_wifi = df_wifi.dropna(subset=["sen_wifi_bssid"])#drop nan values in sen_ssid and sen_bssid column
df_wifi = Merge_Transform.merge_participantIDs(df_wifi, users, device_id_col = "sen_wifi_device_id", include_cities = False)# merge participant IDs
## create anonymized BSSID column: replaece each BSSID with a letter combination
df_wifi["Anonymized WiFi BSSID"] = np.nan
counter = 1
BSSID_RandomBSSID_mapping = {}
for bssid in df_wifi["sen_wifi_bssid"].unique():
    print("started with BSSID " + str(counter) + " of " + str(len(df_wifi["sen_wifi_bssid"].unique())))
    #create random letter combination of length 6
    random_letters = ''.join(random.choice(string.ascii_uppercase) for i in range(6))
    #make sure that the letter combination is not already used
    while random_letters in df_wifi["Anonymized WiFi BSSID"].unique():
        random_letters = ''.join(random.choice(string.ascii_uppercase) for i in range(6))
    #replace BSSID with letter combination
    df_wifi.loc[df_wifi["sen_wifi_bssid"] == bssid, "Anonymized WiFi BSSID"] = random_letters
    #add mapping to dictionary
    BSSID_RandomBSSID_mapping[bssid] = random_letters
    counter += 1
df_wifi.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/WiFi_Anonymization/sensor_wifi_all_BSSIDNaNDropped_IDsMerged_anonymized.csv")
# save BSSID_RandomBSSID_mapping as pkl
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/WiFi_Anonymization/BSSID_RandomBSSID_mapping.pkl", 'wb') as f:
    pickle.dump(BSSID_RandomBSSID_mapping, f)

##create histogram for each participant
df_wifi = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/WiFi_Anonymization/sensor_wifi_all_BSSIDNaNDropped_IDsMerged_anonymized.csv")
conversion_factor = 1/60 # with this the counts will be multiplied: should be 1/(sampling frequency per minute)
number_bars_to_plot = 70 # maximum number of bars/bins in barplot
for participant in df_wifi["sen_wifi_device_id"].unique():
    print("started with participant " + str(participant))
    figure_title = "WiFi BSSID Distribution of User " + str(participant)
    df_participant = df_wifi[df_wifi["sen_wifi_device_id"] == participant]
    fig = data_exploration_sensors.vis_barplot(df_participant, "Anonymized WiFi BSSID", figure_title, y_label, number_bars_to_plot, conversion_factor = conversion_factor)
    fig.savefig(path_storage + "WiFiBSSID_histogram_only-active-smartphon-seessions-yes_user_" + str(participant) + ".png", dpi=600, bbox_inches='tight')
#endregion

#region xmin around event WiFi data: scatterplots with WiFi BSSID x location classes for each participant
label_column_name = "label_location"
point_size_based_on_point_occurence = "yes"
df_wifi = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/sensor_wifi_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl")
BSSID_RandomBSSID_mapping = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/WiFi_Anonymization/BSSID_RandomBSSID_mapping.pkl")

dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
path_storage = '/Users/benediktjordan/Documents/MTS/Iteration01/location/data_exploration/sensors/WiFi/'
df_wifi = labeling_sensor_df(df_wifi, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp") #add labels to df
df_wifi = df_wifi.dropna(subset=[label_column_name]) #drop nan in label column
df_wifi = Merge_Transform.merge_participantIDs(df_wifi, users, device_id_col = None, include_cities = False)# merge participant IDs

print("NaN values in sen_ssid column: " + str(df_wifi["sen_ssid"].isna().sum()))## how many NaN values in sen_ssid column?
df_wifi = df_wifi.dropna(subset=["sen_ssid"])#drop nan values in sen_ssid and sen_bssid column
# create anonymized BSSID column: replaece each BSSID with THE SAME COMBINATION OF LETTERS used for anonymizing the whole WiFi dataset
df_wifi["Anonymized WiFi BSSID"] = np.nan
for bssid in df_wifi["sen_bssid"].unique():
    if bssid in BSSID_RandomBSSID_mapping.keys():
        df_wifi.loc[df_wifi["sen_bssid"] == bssid, "Anonymized WiFi BSSID"] = BSSID_RandomBSSID_mapping[bssid]
    else:
        print("BSSID " + str(bssid) + " not found in mapping dictionary")
        #create random letter combination of length 6
        random_letters = ''.join(random.choice(string.ascii_uppercase) for i in range(6))
        #make sure that the letter combination is not already used
        while random_letters in df_wifi["Anonymized WiFi BSSID"].unique():
            random_letters = ''.join(random.choice(string.ascii_uppercase) for i in range(6))
        #replace BSSID with letter combination
        df_wifi.loc[df_wifi["sen_bssid"] == bssid, "Anonymized WiFi BSSID"] = random_letters

df_wifi = df_wifi.rename(columns={label_column_name: "Location Classes"})# rename "label_location" into "Location Classes"
x_column = "Location Classes"
y_column = "Anonymized WiFi BSSID"

### keep for each event the record with a timestamp closest to the ESM timestamp (first one after ESM timestamp)
df_wifi = df_wifi[df_wifi["timestamp"] >= df_wifi["ESM_timestamp"]] ## delete, for every event, all records with a timestamp earlier than the ESM timestamp
df_wifi = df_wifi.sort_values(by=['ESM_timestamp', 'timestamp']) ## now keep only one record per event, the one with the timestamp closest to the ESM timestamp
df_wifi = df_wifi.drop_duplicates(subset=['ESM_timestamp'], keep='first')
for participant in df_wifi["device_id"].unique():
    print("started with participant " + str(participant))
    figure_title = "Location Classes x WiFi BSSID Counts for User " + str(participant)
    df_participant = df_wifi[df_wifi["device_id"] == participant]
    fig = data_exploration_sensors.vis_scatterplot(df_participant, x_column, y_column, figure_title, point_size_based_on_point_occurence)
    fig.savefig(path_storage + "WiFiBSSIDxLocationClasses_only-active-smartphon-seessions-yes_user_" + str(participant) + ".png", dpi=600, bbox_inches = 'tight')
#endregion

#endregion

#endregion

#endregion

#region data preparation for locations
#region delete accuracies less than 100m,  missing values in GPS data, and duplicates: for entire GPS dataset and event GPS dataset
dataset_paths = [["GPS-all", "/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/locations_all_JSON-transformed_merged-participantIDs.csv"],
                    ["GPS-events_only-active-smartphone-sessions-yes", "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/locations_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl"]]
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/remove_GPSaccuracy/"
min_gps_accuracy = 100
for path in dataset_paths:
    print("started with " + path[0] + " dataset")
    if path[1].endswith(".pkl"):
        df_locations = pd.read_pickle(path[1])
    else:
        df_locations = pd.read_csv(path[1])

    # delete duplicates in timestamp, device_id, latitude, longitude
    print("number of rows before dropping duplicates: " + str(len(df_locations)))
    # change column name "loc_timestamp" into "timestamp" in case it exists
    if "loc_timestamp" in df_locations.columns:
        df_locations = df_locations.rename(columns={"loc_timestamp": "timestamp"})
    df_locations = df_locations.drop_duplicates(subset=["timestamp", "loc_device_id", "loc_double_latitude", "loc_double_longitude"])
    print("number of rows after dropping duplicates: " + str(len(df_locations)))

    # drop rows which have less accuracy than threshold
    print("number of rows before deleting accuracies: " + str(len(df_locations)))
    df_locations, df_analytics = GPS_computations.GPS_delete_accuracy(df_locations, min_gps_accuracy)
    print("number of rows after deleting accuracies: " + str(len(df_locations)))

    df_locations.to_pickle(path_storage + path[0] + "_min_accuracy_" + str(min_gps_accuracy) + ".pkl")
    df_analytics.to_csv(path_storage + path[0] + "_min_accuracy_" + str(min_gps_accuracy) + "_analytics.csv")

#endregion

#region compute places for each participant
#region compute places
distance_threshold = 0.1 # in km
min_gps_accuracy = 100
df_locations_all = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/remove_GPSaccuracy/GPS-all_min_accuracy_" + str(min_gps_accuracy) + ".pkl")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/"
#TODO check where "ESM_timestamp" column was introduced into df_locations_all and remove it again

# reset index: important for later merging the clustered data in again
df_locations_all = df_locations_all.reset_index(drop=True)

#delete unnecessary columns
drop_cols = ["ESM_timestamp", "Unnamed: 0.1", "Unnamed: 0", "0", "1", "3", "loc_label", "loc_provider", "loc_double_speed", "loc_double_bearing"]
df_locations_all = df_locations_all.drop(drop_cols, axis = 1)

#merge participant IDs
df_locations_all = Merge_Transform.merge_participantIDs(df_locations_all, users, device_id_col = None, include_cities = False)# merge participant IDs

#testarea: check how many records exist per participant
for participant in df_locations_all["device_id"].unique():
    df_participant = df_locations_all[df_locations_all["device_id"] == participant]
    print("participant " + str(participant) + " has " + str(len(df_participant)) + " records")

#testarea: check how many unique GPS locations exist per participant
for participant in df_locations_all["device_id"].unique():
    df_participant = df_locations_all[df_locations_all["device_id"] == participant]
    # delete duplicates in GPS location
    df_participant = df_participant.drop_duplicates(subset=["latitude", "longitude"], keep="first")
    print("participant " + str(participant) + " has " + str(len(df_participant)) + " GPS locations")

# iterate through participants and compute places
df_analytics_all = pd.DataFrame()
for participant in tqdm(df_locations_all["device_id"].unique()):
    print("start with participant " + str(participant))

    #check if this participant has been already computed
    if os.path.exists(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df-locations-including-clusters_participant-" +
                                 str(participant) + ".pkl"):
        print("participant " + str(participant) + " already computed")
        df_locations_all = pd.read_pickle(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df-locations-including-clusters_participant-" +
                                 str(participant) + ".pkl")
        df_analytics_all = pd.read_csv(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df_analytics_participant-" +
                        str(participant) +".csv")
        continue

    df_participant = df_locations_all[df_locations_all["device_id"] == participant]

    # check if participant has more than 40000 unique GPS locations: if so, split dataframe (otherwise computationally too expensive)
    df_participant_unique = df_participant.drop_duplicates(subset=["latitude", "longitude"], keep="first")
    if len(df_participant_unique) > 40000:
        print("participant " + str(participant) + " has more than 40000 unique GPS locations")
        # iterate through chunks of 40000 GPS locations through df_participant
        for i in range(0, len(df_participant_unique), 40000):
            print("start with chunk " + str(i) + " of participant " + str(participant))
            # get ID of last value in chunk
            df_participant_unique_chunk = df_participant_unique.iloc[i:i+40000]
            # set first value
            if i == 0:
                # take as first value the first value in df_participant
                first_value = df_participant.index[0]
            else:
                first_value = last_value-1

            # set last value: if it is the last iteration, take the last value in df_participant
            if i >= (len(df_participant_unique)-40000):
                last_value = df_participant.index[-1]
            else:
                last_value = df_participant_unique_chunk.index[-1]

            #create df_participant_chunk from df_participant
            df_participant_chunk = df_participant.loc[first_value:last_value]
            df_label, df_analytics, = GPS_clustering.agg_clustering_computing_centroids(df_participant_chunk, distance_threshold)
            # add chunk identifier column to df_label and df_analytics
            df_label["chunk"] = i
            df_analytics["chunk"] = i

            # merge df_label into df_locations_all
            df_locations_all.loc[df_label.index, "cluster_label"] = df_label["cluster_label"]
            df_locations_all.loc[df_label.index, "cluster_latitude_mean"] = df_label["cluster_latitude_mean"]
            df_locations_all.loc[df_label.index, "cluster_longitude_mean"] = df_label["cluster_longitude_mean"]
            df_locations_all.loc[df_label.index, "chunk"] = df_label["chunk"]

            # concatenate df_analytics
            df_analytics_all = pd.concat([df_analytics_all, df_analytics])

            #double check
            print("Number of NaN values in cluster_label: " + str(df_label["cluster_label"].isna().sum()))


    else:
        df_label, df_analytics, = GPS_clustering.agg_clustering_computing_centroids(df_participant, distance_threshold)

        # merge df_label into df_locations_all
        df_locations_all.loc[df_label.index, "cluster_label"] = df_label["cluster_label"]
        df_locations_all.loc[df_label.index, "cluster_latitude_mean"] = df_label["cluster_latitude_mean"]
        df_locations_all.loc[df_label.index, "cluster_longitude_mean"] = df_label["cluster_longitude_mean"]

        #concatenate df_analytics
        df_analytics_all = pd.concat([df_analytics_all, df_analytics])

    #save intermediate files
    df_analytics_all.to_csv(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df_analytics_participant-" +
                        str(participant) +".csv")
    # save df_locations_all as pickle
    df_locations_all.to_pickle(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df-locations-including-clusters_participant-" +
                                 str(participant) + ".pkl")
df_locations_all.to_pickle(path_storage + "GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df-locations-including-clusters.pkl")
df_analytics_all.to_csv(path_storage + "GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df_analytics.csv")

# analytics section: how many clusters are there for each participant?
df_analytics = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df_analytics.csv")
## delete records with anything else than NaN or 0.0 in chunk column
df_analytics = df_analytics[df_analytics["chunk"].isna() | (df_analytics["chunk"] == 0.0)]
for participant in df_analytics["participant"].unique():
    print("Number of clusters for participant " + str(participant) + ": " + str(len(df_analytics[df_analytics["participant"] == participant]["cluster_label"].unique())))
## get the average number of clusters per participant
print("Average number of clusters per participant: " + str(df_analytics["cluster_label"].nunique()/len(df_analytics["participant"].unique())))

## get for every participant how many clusters contain 90% of all "number_points" values
for participant in df_analytics["participant"].unique():
    df_participant = df_analytics[df_analytics["participant"] == participant]
    # sort df_participant by number_points
    df_participant = df_participant.sort_values(by="number_points", ascending=False)
    # get the 90% of the sum of all number_points
    sum_90_percent = df_participant["number_points"].sum()*0.9
    # get the number of clusters which contain 90% of all number_points
    sum_90_percent_clusters = 0
    for i in range(len(df_participant)):
        sum_90_percent_clusters += df_participant.iloc[i]["number_points"]
        if sum_90_percent_clusters >= sum_90_percent:
            print("Participant " + str(participant) + " has " + str(i+1) + " clusters which contain 90% of all number_points.")
            break
mean = np.mean([29, 25, 79, 10, 33, 29])
print(mean)

## get for every participant, how many clusters contain more than 600 "number_points" values
get_mean = []
for participant in df_analytics["participant"].unique():
    df_participant = df_analytics[df_analytics["participant"] == participant]
    print("Participant " + str(participant) + " has " + str(len(df_participant[df_participant["number_points"] > 10800])) + " clusters for which the participant staid at leas three hours.")
    get_mean.append(len(df_participant[df_participant["number_points"] > 10800]))
print("Mean, median, min, max and standard deviation of clusters which contain more than three hours: " + str(np.mean(get_mean)) + ", " + str(np.median(get_mean)) + ", " + str(np.min(get_mean)) + ", " + str(np.max(get_mean)) + ", " + str(np.std(get_mean)))
#endregion

#region only keep 30 most frequent clusters and add column "number of clusters" and "number of total records"
# only keep clusters which are among 30 most frequent clusters (for every participant and every chunk)
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters.pkl")
df_analytics = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df_analytics.csv")
# create column "number of records for participant"
df_locations["records_total_for_participant"] = np.nan
for participant in df_locations["device_id"].unique():
    print("start with participant " + str(participant) + "...")
    df_participant = df_locations[df_locations["device_id"] == participant]
    # get number of records for participant
    number_of_records_for_participant = len(df_participant)
    df_locations.loc[df_participant.index, "records_total_for_participant"] = number_of_records_for_participant
    # check if participant has any values apart from NaN in chunk
    ## these are participants for which the clusters have been computed in chunks
    ## for them: the most frequent 30 clusters PER CHUNK are computed
    if df_participant["chunk"].isna().sum() != len(df_participant):
        # get list of chunks
        chunks = df_participant["chunk"].unique()
        # iterate through chunks and delete records not in 30 most frequent clusters
        for chunk in chunks:
            print("start with chunk " + str(chunk) + " of participant " + str(participant))
            # get df_participant_chunk
            df_participant_chunk = df_participant[df_participant["chunk"] == chunk]
            # get 30 most frequent clusters
            df_participant_chunk_most_frequent_clusters = df_participant_chunk["cluster_label"].value_counts().head(30)
            # get list of records not in most frequent clusters
            df_participant_chunk_not_in_most_frequent_clusters = df_participant_chunk[~df_participant_chunk["cluster_label"].isin(df_participant_chunk_most_frequent_clusters.index)]
            # delete records not in most frequent clusters in df_locations
            df_locations.drop(df_participant_chunk_not_in_most_frequent_clusters.index, inplace=True)
    else:
        # get 30 most frequent clusters
        df_participant_most_frequent_clusters = df_participant["cluster_label"].value_counts().head(30)
        # get list of records not in most frequent clusters
        df_participant_not_in_most_frequent_clusters = df_participant[~df_participant["cluster_label"].isin(df_participant_most_frequent_clusters.index)]
        # delete records not in most frequent clusters in df_locations
        df_locations.drop(df_participant_not_in_most_frequent_clusters.index, inplace=True)
df_locations.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters.pkl")
#endregion

#testarea: why are there NaN vlaues in cluster_label?
## how many values for participant
len(df_locations_all[df_locations_all["device_id"] == participant])
df_test2 = df_locations_all[df_locations_all["device_id"] == participant]
# check how many NaN values in cluster_labels
df_test2["cluster_label"].isna().sum()
## check in fixErrors if there are any NaN values after computing clusters for participant 3


#region merge clusters for participants which have been computed in chunks
# solve problem for places which have been computed in chunks
## Description: for every participant for whom the places were computed in chunks:
### 1. for every chunk: for every cluster in this chunk: the distances between the cluster_means and all cluster means of clusers in other chunks are computed.
### 2. if the distance is smaller than 100 meter: the distance between all points in this cluster and the close cluster are computed
### 3, if the distance between the "farthes away" points is less than 150 meters: the cluster is merged into the close cluster
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters.pkl")
participant_list = df_locations.groupby("device_id").filter(lambda x: len(x["chunk"].unique()) > 1)["device_id"].unique()
df_locations_merged, df_analytics  = GPS_clustering.merge_clusters_of_chunks(df_locations, participant_list)
df_locations_merged.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged.pkl")
df_analytics.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_analytics.csv")
#endregion

#region create new cluster labels since due to merging there might be clusters with same name but in different chunks
df_merged = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged.pkl")
df_merged.dropna(subset=["cluster_label"], inplace=True)
df_merged.rename(columns={"cluster_label": "cluster_label_old"}, inplace=True)
df_merged["cluster_label"] = np.nan
for participant in df_merged["device_id"].unique():
    print("start with participant " + str(participant) + "...")
    cluster_label_counter = 0
    df_participant = df_merged[df_merged["device_id"] == participant]
    for cluster_label_old in df_participant["cluster_label_old"].unique():
        # check if there are more than one unique value in chunk
        df_participant_cluster = df_participant[df_participant["cluster_label_old"] == cluster_label_old]
        if len(df_participant_cluster["chunk"].unique()) > 1:
            for chunk in df_participant_cluster["chunk"].unique():
                # set cluster_label
                df_merged.loc[df_participant_cluster[df_participant_cluster["chunk"] == chunk].index, "cluster_label"] = cluster_label_counter
                cluster_label_counter += 1
        else:
            #set cluster_label
            df_merged.loc[df_participant_cluster.index, "cluster_label"] = cluster_label_counter
            cluster_label_counter += 1
#save df_merged
df_merged.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_NewClusterLabels.pkl")
#endregion

# region merge clusters for which the centroids are less than 50m apart check distances between different clusters
df_merged = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_NewClusterLabels.pkl")
df_merged["cluster_label"] = df_merged["cluster_label"].astype(int)
# check, for each participant, how many clusters are closter than 50 meters to each other
for participant in df_merged["device_id"].unique():
    print("There are so many clusters for participant " + str(participant) + ": " + str(len(df_merged[df_merged["device_id"] == participant]["cluster_label"].unique())))
    # create distance matrix for all clusters
    df_participant = df_merged[df_merged["device_id"] == participant]
    # get df of cluster_labels and for each cluster_label the first value of cluster_longitude_mean and cluster_latitude_mean
    df_cluster_labels = df_participant[["cluster_label", "cluster_longitude_mean", "cluster_latitude_mean"]].drop_duplicates()
    # sort by cluster_label
    df_cluster_labels.sort_values(by=["cluster_label"], inplace=True)
    # compute distance matrix using cdist and haversine
    X = np.array(list(zip(df_cluster_labels["cluster_longitude_mean"], df_cluster_labels["cluster_latitude_mean"])))
    distance_matrix = cdist(X, X, metric=haversine.haversine)

    #merge the clusters which are closer than 50 meters to each other
    for cluster_label in df_cluster_labels["cluster_label"].unique():
        print("start with cluster " + str(cluster_label) + "...")
        # get all clusters which are closer than 50 meters to the cluster (be aware of indexing
        close_clusters = np.where(distance_matrix[int(cluster_label)] < 0.05)[0]

        # merge all clusters which are closer than 50 meters to the cluster for the same participant
        for close_cluster in close_clusters:
            if close_cluster != cluster_label:
                df_merged.loc[df_merged[(df_merged["cluster_label"] == close_cluster) & (df_merged["device_id"] == participant)].index, "cluster_label"] = cluster_label
                # recompute also cluster_longitude_mean and cluster_latitude_mean
                df_merged.loc[df_merged[(df_merged["cluster_label"] == cluster_label) & (df_merged["device_id"] == participant)].index, "cluster_longitude_mean"] = df_merged[(df_merged["cluster_label"] == cluster_label) & (df_merged["device_id"] == participant)]["longitude"].mean()
                df_merged.loc[df_merged[(df_merged["cluster_label"] == cluster_label) & (df_merged["device_id"] == participant)].index, "cluster_latitude_mean"] = df_merged[(df_merged["cluster_label"] == cluster_label) & (df_merged["device_id"] == participant)]["latitude"].mean()
                print("cluster " + str(close_cluster) + " merged with cluster " + str(cluster_label))

    # repeat the merging:
    df_participant = df_merged[df_merged["device_id"] == participant]
    df_cluster_labels = df_participant[["cluster_label", "cluster_longitude_mean", "cluster_latitude_mean"]].drop_duplicates()
    X = np.array(list(zip(df_cluster_labels["cluster_longitude_mean"], df_cluster_labels["cluster_latitude_mean"])))
    distance_matrix = cdist(X, X, metric=haversine.haversine)
    for cluster_label in df_cluster_labels["cluster_label"].unique():
        print("start with cluster " + str(cluster_label) + "...")
        cluster_label_matrix_index = np.where(df_cluster_labels["cluster_label"] == cluster_label)[0][0]
        # get all clusters which are closer than 50 meters to the cluster (be aware of indexing
        close_clusters_indices = np.where(distance_matrix[int(cluster_label_matrix_index)] < 0.05)[0]

        # merge all clusters which are closer than 50 meters to the cluster for the same participant
        for close_cluster in close_clusters_indices:
            # compute what the close_cluster is
            close_cluster_name = int(df_cluster_labels.iloc[close_cluster]["cluster_label"])
            if close_cluster_name != cluster_label:
                df_merged.loc[df_merged[(df_merged["cluster_label"] == close_cluster_name) & (df_merged["device_id"] == participant)].index, "cluster_label"] = cluster_label
                # recompute also cluster_longitude_mean and cluster_latitude_mean
                df_merged.loc[df_merged[(df_merged["cluster_label"] == cluster_label) & (df_merged["device_id"] == participant)].index, "cluster_longitude_mean"] = df_merged[(df_merged["cluster_label"] == cluster_label) & (df_merged["device_id"] == participant)]["longitude"].mean()
                df_merged.loc[df_merged[(df_merged["cluster_label"] == cluster_label) & (df_merged["device_id"] == participant)].index, "cluster_latitude_mean"] = df_merged[(df_merged["cluster_label"] == cluster_label) & (df_merged["device_id"] == participant)]["latitude"].mean()
                print("cluster " + str(close_cluster_name) + " merged with cluster " + str(cluster_label))

    print("There are so many clusters for participant " + str(participant) + " after merging: " + str(len(df_merged[df_merged["device_id"] == participant]["cluster_label"].unique())))
#save df_merged
df_merged.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_NewClusterLabels_CloseClustersMerged.pkl")
#endregion



#endregion

#region compute features for places

#region calculate features for GPS data: this is necessary to compute features for places
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_NewClusterLabels_CloseClustersMerged.pkl")
df_results = pd.DataFrame()

## add local timezone column
df_locations = Merge_Transform.add_local_timestamp_column(df_locations, users)

#TODO the assumption of one timezone for every participant is not entirely correct: for one participant there ws sometimes a different timezone used (Amsterdam instead of London)
#TODO: you can check with this code:
## get timezones for participants
#df_timezones = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/timezone_esm_timeperiod_5 min.csv_JSONconverted.pkl")
#df_timezones = Merge_Transform.merge_participantIDs(df_timezones, users, device_id_col = None, include_cities = False)# merge participant IDs
#df_timezones = df_timezones[["device_id", "tim_timezone"]].drop_duplicates(subset=["device_id", "tim_timezone"], keep="first")

## calculate features
df_results = pd.DataFrame()
for participant in df_locations["device_id"].unique():
    df_participant = df_locations[df_locations["device_id"] == participant]
    print("start with participant " + str(participant) + " (" + str(len(df_participant)) + " rows)")

    # delete rows which have same timestamp, device_id, latitude, and longitude
    df_participant = df_participant.drop_duplicates(subset=["loc_timestamp", "device_id", "latitude", "longitude"], keep="first")

    # convert unix time to datetime (loc_timestamp)
    df_participant["loc_timestamp"] = pd.to_datetime(df_participant["loc_timestamp"], unit="ms")
    df_participant = df_participant.rename(columns={"loc_timestamp": "timestamp"})

    ## calculate number of days for which GPS data is available
    df_participant["days_with_data_count"] = FeatureExtraction_GPS.gps_features_count_days_with_enough_data(df_participant, min_gps_points = 60)

    ## create weekday column
    df_participant["weekday"] = df_participant["timestamp_local"].dt.weekday
    ## create weekend vs. workday column
    df_participant["weekend_weekday"] = df_participant["weekday"].apply(lambda x: "weekend" if x in [5,6] else "weekday")
    ## create hour_of_day column
    df_participant["hour_of_day"] = df_participant["timestamp_local"].dt.hour
    ## create time_of_day column
    df_participant["time_of_day"] = df_participant["hour_of_day"].apply(lambda x: "morning" if x in [6,7,8,9,10,11] else "afternoon" if x in [12,13,14,15,16,17] else "evening" if x in [18,19,20,21,22,23] else "night" if x in [0,1,2,3,4,5] else "unknown")

    # create stays and compute stays features
    df = df_participant.copy()
    freq = 1 # in Hz; frequency of the sensor data
    duration_threshold= 60 #minimum duration of a stay in seconds
    min_records_fraction=0.5 #minimum fraction of records which have to exist in a chunk to be considered as a stay
    df_participant = FeatureExtraction_GPS.compute_stays_and_stays_features(df_participant, freq, duration_threshold, min_records_fraction)

   # convert cluster_label to int
    df_participant["cluster_label"] = df_participant["cluster_label"].astype(int)

    print("finished with participant " + str(participant) + " (" + str(len(df_participant)) + " rows)")
    # concatenate into df_results
    df_results = pd.concat([df_results, df_participant])
#TODO find out why I didn´t drop duplicates in any of the steps above but only here; and why are there so many duplicates?
df_results.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_features/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_NewClusterLabels_CloseClustersMerged_withGPSFeatures.pkl")


#endregion

#region calculate features for places
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_features/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_NewClusterLabels_CloseClustersMerged_withGPSFeatures.pkl")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features_new/"
frequency = 1 # in Hz; frequency of the sensor data
timeslot_lists = [["hour_of_day", [0,1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,15,16,17,18, 19,20,21,22,23]],
                    ["time_of_day", ["morning", "afternoon", "evening", "night"]],
                    ["weekday", [0,1,2,3,4,5,6]],
                    ["weekend_weekday", ["weekend", "weekday"]]]

### create columns for timeslots
columns = []
for timeslot_list in timeslot_lists:
    timeslot_type = timeslot_list[0]
    timslots = timeslot_list[1]
    for timeslot in timslots:
        columns.append(timeslot_type + "_" + str(timeslot) + "_arrive_percentage")
        columns.append(timeslot_type + "_" + str(timeslot) + "_arrive_percentage_trusted")
        columns.append(timeslot_type + "_" + str(timeslot) + "_leave_percentage")
        columns.append(timeslot_type + "_" + str(timeslot) + "_leave_percentage_trusted")
        columns.append(timeslot_type + "_" + str(timeslot) + "_intersecting_percentage")
        columns.append(timeslot_type + "_" + str(timeslot) + "_intersecting_percentage_trusted")
        columns.append(timeslot_type + "_" + str(timeslot) + "_time_fraction")
        columns.append(timeslot_type + "_" + str(timeslot) + "_time_fraction_trusted")
        columns.append(timeslot_type + "_" + str(timeslot) + "_time_per_day")
        columns.append(timeslot_type + "_" + str(timeslot) + "_time_per_day_trusted")
further_columns = ["device_id", "place", "longitude", "latitude", "total_records_in_cluster", "total_records_fraction", "total_records_of_biggest_clusters_fraction", "visits_per_day", "visits_per_day_trusted",
                                    "visits_fraction", "visits_fraction_trusted", "time_per_day", "time_per_day_trusted",
                                    "stay_duration_mean", "stay_duration_mean_trusted", "stay_duration_max", "stay_duration_max_trusted",
                                    "fraction_of_time_spent_at_place", "fraction_of_time_spent_at_place_trusted"]
columns_list = further_columns + columns


### only keep 15 biggest clusters
for participant in df_locations["device_id"].unique():
    df_participant = df_locations[df_locations["device_id"] == participant]
    print("Participant " + str(participant) + " has " + str(len(df_participant["cluster_label"].unique())) + " clusters")
    # get the index of the all clusters except the 15 biggest
    index_to_drop = df_participant["cluster_label"].value_counts().index[15:]
    # drop all rows with these clusters for this participant
    df_locations = df_locations.drop(df_locations[(df_locations["device_id"] == participant) & (df_locations["cluster_label"].isin(index_to_drop))].index)
    print("Participant " + str(participant) + " has " + str(len(df_locations[df_locations["device_id"] == participant]["cluster_label"].unique())) + " clusters left")

### compute features
df_results = pd.DataFrame(columns= columns_list)
for participant in tqdm(df_locations["device_id"].unique()):
    print("start with participant " + str(participant))
    #check if features have been computed already
    if os.path.exists(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withPlacesFeatures_participant-" + str(participant) + ".pkl"):
        df_results = pd.read_pickle(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withPlacesFeatures_participant-" + str(participant) + ".pkl")
        print("features for participant " + str(participant) + " already computed")
        continue

    # compute features
    df_participant = df_locations[df_locations["device_id"] == participant]
    cluster_counter = 1
    for place in df_participant["cluster_label"].unique():
        print("computing features for place (" + str(cluster_counter) + "/" + str(len(df_participant["cluster_label"].unique())) + ")")
        cluster_counter += 1

        line_number = len(df_results)
        df_results.loc[line_number] = np.nan

        # add device_id, place, longitude, and latitude
        df_results.loc[line_number]["device_id"] = participant
        df_results.loc[line_number]["place"] = place
        df_results.loc[line_number]["longitude"] = df_participant[df_participant["cluster_label"] == place]["cluster_longitude_mean"].iloc[0]
        df_results.loc[line_number]["latitude"] = df_participant[df_participant["cluster_label"] == place]["cluster_latitude_mean"].iloc[0]

        # add total records in the cluster
        df_results.loc[line_number]["total_records_in_cluster"] = len(df_participant[df_participant["cluster_label"] == place])

        # add fraction of total records which are at this place: number of total records for each participant is in column records_total_for_participant
        df_results.loc[line_number]["total_records_fraction"] = len(df_participant[df_participant["cluster_label"] == place]) / df_participant["records_total_for_participant"].iloc[0]

        # add fraction of total records of 30 biggest clusters which are at this place
        df_results.loc[line_number]["total_records_of_biggest_clusters_fraction"] = len(df_participant[df_participant["cluster_label"] == place]) / len(df_participant)

        # compute visits_per_day for place
        df_results.loc[line_number]["visits_per_day"], df_results.loc[line_number]["visits_per_day_trusted"] = FeatureExtraction_GPS.gps_features_places_visits_per_day(df_participant, participant, place)

        # compute visits_fraction for place
        df_results.loc[line_number]["visits_fraction"], df_results.loc[line_number]["visits_fraction_trusted"] = FeatureExtraction_GPS.gps_features_places_visits_fraction(df_participant, participant, place)

        # compute time_per_day for place
        df_results.loc[line_number]["time_per_day"], df_results.loc[line_number]["time_per_day_trusted"] = FeatureExtraction_GPS.gps_features_places_time_per_day(df_participant, participant, place)

        # stay_duration_max and mean: normal and trusted
        df_results.loc[line_number]["stay_duration_max"] = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration"].max()
        df_results.loc[line_number]["stay_duration_max_trusted"]  = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration_trusted"].max()
        df_results.loc[line_number]["stay_duration_mean"]  = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration"].mean()
        df_results.loc[line_number]["stay_duration_mean_trusted"]  = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration_trusted"].mean()

        # fraction of time spent at this place (of all time spent at all places)
        df_results.loc[line_number]["fraction_of_time_spent_at_place"] = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration"].sum() / df_participant[(df_participant["device_id"] == participant)]["stay_duration"].sum()

        # trusted fraction of time spent at this place (of all time spent at all places)
        df_results.loc[line_number]["fraction_of_time_spent_at_place_trusted"] = \
        df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)][
            "stay_duration"].sum() / df_participant[(df_participant["device_id"] == participant)]["stay_duration_trusted"].sum()

        # calculate features for different timeslots
        timeslot_counter = 1
        # create number of timeslots as all elements withing the second level of the list
        number_of_timeslots = sum([len(timeslot_list[1]) for timeslot_list in timeslot_lists])
        for timeslot_list in timeslot_lists:
            timeslot_type = timeslot_list[0]
            for timeslot in timeslot_list[1]:
                print("computing features for timeslot " + str(timeslot_counter) + "/" + str(number_of_timeslots))
                timeslot_counter += 1

                # compute arrive, leave, and intersecting percentage for place
                df_results.loc[line_number][timeslot_type + "_" + str(timeslot) + "_arrive_percentage"], df_results.loc[line_number][
                    timeslot_type + "_" + str(timeslot)  + "_arrive_percentage_trusted"], \
                    df_results.loc[line_number][timeslot_type + "_" + str(timeslot)  + "_leave_percentage"], df_results.loc[line_number][
                    timeslot_type + "_" + str(timeslot)  + "_leave_percentage_trusted"], \
                    df_results.loc[line_number][timeslot_type + "_" + str(timeslot)  + "_intersecting_percentage"], df_results.loc[line_number][
                    timeslot_type + "_" + str(timeslot)  + "_intersecting_percentage_trusted"] = FeatureExtraction_GPS.compute_arrive_leave_intersecting_percentage(df_participant,
                                                                                                              participant,
                                                                                                              place,
                                                                                                              timeslot,
                                                                                                              timeslot_type)
                # compute time fraction and time per day
                df_results.loc[line_number][timeslot_type + "_" + str(timeslot) + "_time_fraction"], \
                    df_results.loc[line_number][timeslot_type + "_" + str(timeslot) + "_time_fraction_trusted"], \
                    df_results.loc[line_number][timeslot_type + "_" + str(timeslot) + "_time_per_day"], df_results.loc[line_number][
                    timeslot_type + "_" + str(timeslot) + "_time_per_day_trusted"] = FeatureExtraction_GPS.compute_time_fraction_time_per_day(df_participant, participant, place, timeslot, timeslot_type, frequency)
    #save intermediate file
    df_results.to_pickle(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withPlacesFeatures_participant-" + str(participant) + ".pkl")
df_results.to_pickle(path_storage + "GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest15Clusters_ClustersFromChunksMerged_NewClusterLabels_CloseClustersMerged_withGPSFeatures_PlacesFeatures.pkl")

# delete all places which don´t have at least 1 stay
print("number of places before deleting places with no stay: " + str(len(df_results)))
df_results = df_results[df_results["stay_duration_mean"] > 0]
print("number of places after deleting places with no stay: " + str(len(df_results)))

# calculate features which put other place-features in a relationship: rank ascent and rank descent
## delete "device_id" and "place" from column_list
columns_list.remove("device_id")
columns_list.remove("place")
df_results = FeatureExtraction_GPS.gps_features_places_rank_ascent_descent(df_results, columns_list)
#TODO double check the intermediate results with analytics dataframe or print functions

df_results.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features/places_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withPlacesFeatures_DeletedPlacesWithoutStays.pkl")

#testarea
df_test
#endregion

#endregion

#region labeling of places
#region get GPS locations for home and office for each participant
min_gps_accuracy = 100
only_active_smartphone_sessions = "yes"
distance_threshold = 0.1 #in km; maximum distance between most distant points in any cluster
df_locations_events = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/remove_GPSaccuracy/GPS-events_only-active-smartphone-sessions-"+ only_active_smartphone_sessions +"_min_accuracy_" + str(min_gps_accuracy) +".pkl")
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
label_column_name = "label_location"
classes = ["at home", "in the office"]
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/map_classes_to_locations_and_wifi/"

# for every event: only keep first GPS record afterwards
df_locations_events = GPS_computations.first_sensor_record_after_event(df_locations_events)

#label sensor data & drop NaNs in label column
df_locations_events = labeling_sensor_df(df_locations_events, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp")
df_locations_events = Merge_Transform.merge_participantIDs(df_locations_events, users, device_id_col = None, include_cities = False)# merge participant IDs
df_locations_events = df_locations_events.dropna(subset=[label_column_name])

# for each participant and class: cluster GPS points and take dominant cluster as location
df_analytics_all = pd.DataFrame(columns=["participant", "class", "cluster_label", "number_points", "latitude_mean", "longitude_mean"])
for participant in df_locations_events["device_id"].unique():
    print("Started with participant " + str(participant) + "")
    df_participant = df_locations_events[df_locations_events["device_id"] == participant]
    for label in classes:
        df_label = df_participant[df_participant[label_column_name] == label]
        # if df_label is empty, continue with next label
        if df_label.empty:
            print("no label " + label + " for participant " + str(participant) + "")
            continue

        # if only one value in df_label, no clustering needed
        if len(df_label) == 1:
            df_locations_events.loc[df_label.index, "cluster_label"] = 0
            df_locations_events.loc[df_label.index, "cluster_latitude_mean"] = df_label["latitude"].iloc[0]
            df_locations_events.loc[df_label.index, "cluster_longitude_mean"] = df_label["longitude"].iloc[0]
            # add label to df_analytics
            df_analytics = pd.DataFrame({"participant": [participant], "class": [label], "cluster_label": [0], "number_points": [1],
                                            "latitude_mean": [df_label["latitude"].iloc[0]], "longitude_mean": [df_label["longitude"].iloc[0]]})

        # cluster GPS locations
        else:
            df_label, df_analytics, dominant_cluster_latitude, dominant_cluster_longitude = agg_clustering_computing_centroids(df_label, distance_threshold)

            # replace records in df_locations_events with the records in df_label
            df_locations_events.loc[df_label.index, "cluster_label"] = df_label["cluster_label"]
            df_locations_events.loc[df_label.index, "cluster_latitude_mean"] = df_label["cluster_latitude_mean"]
            df_locations_events.loc[df_label.index, "cluster_longitude_mean"] = df_label["cluster_longitude_mean"]

            # add label to df_analytics
            df_analytics["class"] = label

        #concatenate df_analytics to df_analytics_all
        df_analytics_all = pd.concat([df_analytics_all, df_analytics])



        print(agg_clustering.labels_)
df_analytics_all["participant"] = df_analytics_all["participant"].astype(int)
df_analytics_all.to_csv(path_storage + "GPS_only_active_smartphone_sessions" + only_active_smartphone_sessions + "_min-GPS-accuracy-" + str(min_gps_accuracy) +
          "_df_analytics.csv")
df_locations_events.to_csv(path_storage + "GPS_only_active_smartphone_sessions" + only_active_smartphone_sessions + "_min-GPS-accuracy-" + str(min_gps_accuracy) +
            "_df-locations-including-clusters.csv")

# create dictionary which maps classes to dominant cluster location
df_analytics_all = pd.read_csv(path_storage + "GPS_only_active_smartphone_sessions" + only_active_smartphone_sessions + "_min-GPS-accuracy-" + str(min_gps_accuracy) +
          "_df_analytics.csv")

dict_class_locations = {}
for participant in df_analytics_all["participant"].unique():
    dict_class_locations[participant] = {}
    for label in df_analytics_all["class"].unique():
        df_participant_label = df_analytics_all[(df_analytics_all["participant"] == participant) & (df_analytics_all["class"] == label)]

        #check if two cluster_label exist in df_participant_label
        if (len(df_participant_label) > 1) & (label == "at home") :
            # order the dataframe by cluster size
            df_participant_label = df_participant_label.sort_values(by="number_points", ascending=False)
            dict_class_locations[participant]["at home"] = {"latitude": df_participant_label["latitude_mean"].iloc[0],
                                                        "longitude": df_participant_label["longitude_mean"].iloc[0],
                                                            "number_points": df_participant_label["number_points"].iloc[0]}
            dict_class_locations[participant]["at home 2"] = {"latitude": df_participant_label["latitude_mean"].iloc[1],
                                                        "longitude": df_participant_label["longitude_mean"].iloc[1],
                                                              "number_points": df_participant_label["number_points"].iloc[1]}
            continue

        if df_participant_label.empty:
            continue
        dict_class_locations[participant][label] = {"latitude": df_participant_label["latitude_mean"].iloc[0],
                                                    "longitude": df_participant_label["longitude_mean"].iloc[0],
                                                    "number_points": df_participant_label["number_points"].iloc[0]}
with open(path_storage + "GPS_only_active_smartphone_sessions" + only_active_smartphone_sessions + "_min-GPS-accuracy-" + str(min_gps_accuracy) +
          "_dict_mapping_class_locations.pkl", "wb") as f:
    pickle.dump(dict_class_locations, f)


#endregion

#region label places (based on the mapping of GPS locations to home and office)
min_gps_accuracy = 100
number_biggest_clusters = 15
dict_mapping = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/map_classes_to_locations_and_wifi/GPS_only_active_smartphone_sessionsyes_min-GPS-accuracy-100_dict_mapping_class_locations.pkl")

#temp load example file; replace by final file!
df_places_features = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features/INTERMEDIATE_GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withPlacesFeatures_participant-8.pkl")

# add label to every place based on the mapping: if distance between place and home/office is smaller than 100m, then label place as home/office, otherwise label as "other"
df_places_features["label"] = np.nan
df_places_features["label_distance_to_label"] = np.nan
for index, row in df_places_features.iterrows():
    # get mapping for participant
    dict_mapping_participant = dict_mapping[row["device_id"]]
    # iterate through mappings and check if distance is smaller than 100m
    for key, value in dict_mapping_participant.items():
        distance = haversine.haversine((row["latitude"], row["longitude"]), (value["latitude"], value["longitude"]))
        if distance < 0.1:
            # check if at that index there is already a label
            if not pd.isnull(df_places_features.loc[index, "label"]):
                print("label already exists for participant " + str(row["device_id"]) + " and place " + str(row["place"]))
            df_places_features.loc[index, "label"] = key
            df_places_features.loc[index, "label_distance_to_label"] = distance
        else:
            df_places_features.loc[index, "label"] = "other"
# save
df_places_features.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features/GPS_min-GPS-accuracy-" + str(min_gps_accuracy)  + "_df-locations-including-clusters_OnlyBiggest" + str(number_biggest_clusters) + "Clusters_ClustersFromChunksMerged_withPlacesFeatures_labeled.csv")

#endregion
#endregion
#endregion

#region modeling for locations
min_gps_accuracy = 100
number_biggest_clusters = 15
df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features/GPS_min-GPS-accuracy-" + str(min_gps_accuracy)  + "_df-locations-including-clusters_OnlyBiggest" + str(number_biggest_clusters) + "Clusters_ClustersFromChunksMerged_withPlacesFeatures_labeled.csv")

#testarea
## keep only coluns "label", "label_distance_to_label", "latitude", "longitude", "total_records_in_cluster"
df = df[["device_id", "place", "longitude", "latitude", "label", "label_distance_to_label", "total_records_in_cluster"]]

label_segment = 0 # actually this is redundant here, but needed by DF as it was important in human motion classification
label_classes = ["at home", "at home 2", "office", "other" ] # which label classes should be considered
label_mapping = None
label_column_name = "label"
n_permutations = 10 # define number of permutations; better 1000
parameter_tuning = "yes" # if yes: parameter tuning is performed; if False: default parameters are used
drop_cols = ["label_distance_to_label", "latitude", "longitude", "total_records_in_cluster"] # columns that should be dropped
feature_importance = "shap"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/modeling/decision_forest/"

#replace any NaN values with 0 in df: the NaN values are just wrongly labelled as NaN in the feature createion process
df = df.fillna(0)

# set CV mode
combine_participants = False
#participants_test_set = [2,4,6,9]

parameter_set = {
    "n_jobs": -1,# use all cores
    "random_state": 11 # set random state for reproducibility
}

parameter_set = {
    "n_estimators": 100, #default
    "criterion": "gini", #default
    "max_depth": None, #default
    "min_samples_split": 2,#default
    "min_samples_leaf": 1,#default
    "min_weight_fraction_leaf": 0.,#default
    "max_features": None, # THIS IS CHANGED
    "max_leaf_nodes": None,#default
    "min_impurity_decrease": 0.0,#default
    "bootstrap": True,#default
    "oob_score": False,#default
    "n_jobs": None,#default
    "verbose": 0,#default
    "warm_start": False,#default
    "class_weight": "balanced", # THIS IS CHANGED
    "n_jobs": -1,#default
    "random_state": 11#default
}

# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 100
max_depth = [2, 5, 15, None] # default None
min_samples_split = [2, 10, 30] # default 2
min_samples_leaf = [1, 5] # default 1
max_features = ["sqrt", 5, "log2", 20, None] # default "sqrt"
oob_score = [True, False] # default False;
class_weight = ["balanced", "balanced_subsample", None] # default None
criterion =[ "gini", "entropy", "log_loss"] # default "gini"
max_samples = [None, 0.5, 0.8] # default None which means that 100% of the samples are used for each tree
bootstrap = [True]

# smaller parameter tuning set in order to have less runtime
n_estimators = [100, 800]  # default 100
max_depth = [15, None, 25] # default None
min_samples_split = [2, 8] # default 2
min_samples_leaf = [1] # default 1
max_features = ["sqrt", "log2", None] # default "sqrt"
oob_score = [False, True] # default False;
class_weight = ["balanced", "balanced_subsample"] # default None
criterion =[ "gini", "entropy"] # default "gini"
max_samples = [None, 0.5] # default None which means that 100% of the samples are used for each tree
bootstrap = [True]

grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score = oob_score, class_weight = class_weight,
                            criterion = criterion, max_samples = max_samples, bootstrap = bootstrap)

df = df.drop(columns=drop_cols)
#in case commbine_participants == "yes", merge participant IDs into either 1 (=train) or 2 (=test)
if combine_participants == True:
    # iterate through all IDs in "device_id" column and replace with 1 or 2: if ID is in participants_test_set, replace with 2, else replace with 1
    df["device_id_traintest"] = df["device_id"].apply(lambda x: 2 if x in participants_test_set else 1)

#combine label classes if necessary
if label_mapping != None:
    df = df.reset_index(drop=True)
    for mapping in label_mapping:
        df.loc[df["label_human motion"] == mapping[0], "label_human motion"] = mapping[1]
df = df[df[label_column_name].isin(label_classes)]  # drop rows which don´t contain labels which are in label_classes

#run DF
df_decisionforest_results = DecisionForest.DF_sklearn(df, label_segment, label_column_name,
                                                      n_permutations, path_storage, feature_importance = feature_importance,
                                                      confusion_matrix_order = label_classes,  parameter_tuning = parameter_tuning,
                                                      parameter_set = parameter_set, grid_search_space = grid_search_space, combine_participants = combine_participants)

df_analytics.to_csv("/iktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "s_parameter_tuning-" + parameter_tuning + "_analytics.csv")
df_decisionforest_results.to_csv("/diktjordan/Documents/MTS/Iteration02/human_motion/data_analysis/decision_forest/timeperiod_around_event-" + str(label_segment) + "-GPS_parameter_tuning-" + parameter_tuning + "_results_overall.csv")




#endregion

#endregion


#region location stuff: outdated
input_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all_sorted.csv"

# temporary: merge participant ids
df = pd.read_csv(input_path)
df = merge_participantIDs(df, users)
df.to_csv(input_path+ "_merged-participantIDs.csv")

# compute frequent locations for each user during day and night (on all location data) (based on GPS_find_frequent_clusters class)
input_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all_sorted.csv_merged-participantIDs.csv"
output_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/"
range_n_clusters = [2, 3, 4, 5, 6] #which number of clusters are tested by the algorithm
## calculate in chunks (since computational complexity doesn´t allow to calculate all participants at once) for day and night
for i in [[9,18],[0,6]]:
    starthour = i[0]
    endhour = i[1]

    #calculate chunks
    chunk_counter = 1
    for df_locations_alltime in pd.read_csv(input_path, chunksize=500000):
        time_start = time.time()
        df_summary = GPS_find_frequent_locations.cluster_for_timeperiod(df_locations_alltime, starthour, endhour, range_n_clusters, output_path)
        df_summary.to_csv(os.path.join(output_path, "hours-{}-{}_freqquent_locations_summary_chunk-{}.csv".format(starthour, endhour, chunk_counter),), index=False)
        chunk_counter += 1
        time_end = time.time()
        print("Time needed for chunk: " + str((time_end - time_start)/60) + " minutes")
        print("Done with chunk " + str(chunk_counter) + " chunks.")
    if starthour == 9:
        chunk_counter_day = chunk_counter
    else:
        chunk_counter_night = chunk_counter
## merge all chunks
for i in [[9,18],[0,6]]:
    starthour = i[0]
    endhour = i[1]
    if starthour == 9:
        chunk_counter = chunk_counter_day
    else:
        chunk_counter = chunk_counter_night

    for i in range(1, chunk_counter + 1):
        df_chunk = pd.read_csv(
            output_path + "hours-" + str(starthour) + "-" + str(endhour) + "_freqquent_locations_summary_chunk-" + str(i) + ".csv")
        if i == 1:
            df_merged = df_chunk
        else:
            df_merged = pd.concat([df_merged, df_chunk], ignore_index=True)
    df_merged.to_csv(output_path + "hours-" + str(starthour) + "-" + str(endhour) + "_freqquent_locations_summary_chunksmerged.csv", index=False)

## merge locations which are close to each other (for day and night)
threshold_distance = 50 #in meters
for i in [[9,18],[0,6]]:
    starthour = i[0]
    endhour = i[1]
    df = pd.read_csv(output_path + "hours-" + str(starthour) + "-" + str(endhour) + "_freqquent_locations_summary_chunksmerged.csv")
    df_merged = GPS_find_frequent_locations.merge_close_locations(df, threshold_distance)
    df_merged.to_csv(output_path + "hours-" + str(starthour) + "-" + str(endhour) + "_freqquent_locations_summary_chunksmerged_locationsmerged.csv", index=False)
## delete locations which are not frequent enough (for day and night)
## if %values of cluster entries < (number_of_entries/number_clusters+1) then delete
for i in [[9,18],[0,6]]:
    starthour = i[0]
    endhour = i[1]
    df = pd.read_csv(output_path + "hours-" + str(starthour) + "-" + str(endhour) + "_freqquent_locations_summary_chunksmerged_locationsmerged.csv")
    df_merged = GPS_find_frequent_locations.delete_locations_not_frequent_enough(df)
    df_merged.to_csv(output_path + "hours-" + str(starthour) + "-" + str(endhour) + "_freqquent_locations_summary_chunksmerged_locationsmerged_unfrequentlocationsdeleted.csv", index=False)

# classify the frequent locations as home, work-place, other frequent place
df_frequentlocations_day = pd.read_csv(output_path + "hours-9-18_freqquent_locations_summary_chunksmerged_locationsmerged_unfrequentlocationsdeleted.csv")
df_frequentlocations_night = pd.read_csv(output_path + "hours-0-6_freqquent_locations_summary_chunksmerged_locationsmerged_unfrequentlocationsdeleted.csv")
#temporary: merge participant ids
df_frequentlocations_day = Merge_Transform.merge_participantIDs(df_frequentlocations_day, users, device_id_col="participant")
df_frequentlocations_night = Merge_Transform.merge_participantIDs(df_frequentlocations_night, users, device_id_col="participant")
threshold_distance = 50 #in meters
locations_classifications = GPS_find_frequent_locations.classify_locations(df_frequentlocations_day, df_frequentlocations_night, threshold_distance)
with open(output_path + "locations_classifications.pkl", 'wb') as f:
    pickle.dump(locations_classifications, f) #save locations classifications with pickle


# load (with pickle) & label location data with ESM location data (on xmin around events location data)
locations_classifications = pickle.load(open(output_path + "locations_classifications.pkl", 'rb'))
input_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/locations_esm_timeperiod_5 min.csv_JSONconverted.pkl"
label_column_name = "label_location"
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
df_locations_xmin = pd.read_pickle(input_path)
df_locations_xmin = Merge_Transform.merge_participantIDs(df_locations_xmin, users) #temporary: merge participant ids
df_locations_xmin = labeling_sensor_df(df_locations_xmin, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp")

# get only one location for every event
df_locations = GPS_computations.get_locations_for_labels(df_locations_xmin, label_column_name)

# merge label categories to home, work, other frequent locations
map_locations = {"at another workplace": "other frequent place",
                 "at a friends place": "other locations",
                 "in restaurant": "other locations",
                    "on the way": "other locations",
                    "with friends outside": "other locations",
                 "at home": "home",
                    "in the office": "work"}
df_locations["label"] = df_locations["label"].replace(map_locations)

# delete all rows with label "other locations"
df_locations = df_locations[df_locations["label"] != "other locations"]

# create table with participants x labels
df_label_counts = df_locations.groupby("participant")["label"].value_counts().unstack().fillna(0)
df_label_counts["total"] = df_label_counts.sum(axis=1)
df_label_counts.loc["total"] = df_label_counts.sum(axis=0)

# predict for every location if it is home, work-place or other frequent place
df_locations = GPS_computations.classify_locations(df_locations, locations_classifications)#

# compute accuracy of location classification
df_locations["correct_classification"] = df_locations["label"] == df_locations["location_classification"]
df_locations["correct_classification"].value_counts()
#print("Accuracy is " + str(df_locations["correct_classification"].value_counts()[1]/df_locations["correct_classification"].value_counts().sum()))
acc = accuracy_score(df_locations["label"], df_locations["location_classification"])
acc_balanced = balanced_accuracy_score(df_locations["label"], df_locations["location_classification"])
# visualize confusion matrix
# import balanced_accuracy_score

acc_balanced, absolute_or_relative_values, title, label_mapping, save_path
visualizations.confusion_matrix(df_locations["location_classification"], df_locations["label"], acc_balanced, "relative", "Confusion Matrix Location Classification", None, output_path + "confusion_matrix_location_classification.png")
#endregion

#region before and after sleep
## How to improve the model (currently I have 54% balanced accuracy (with 4 classes))?
#TODO train seperate models for "before sleep" and "after sleep" (in "after sleep", add "time since last sleep period" as feature)
#TODO get better data: right now I am using a feature set which has been created for "human motion" label (in features
# selection). This doesnt contain some events (if there was no human motion label). Try creating a very limited
# feature set which I can also create for the "before sleep" data
#TODO train personalized models (think how to deal with very limited data)
#TODO train LSTM model
#TODO add feature: add battery charging events as feature


#region data exploration for sleep
#region data exploration labels
#how many labels do we have?
df_labels = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
print("Number of labels in sleep activity: " + str(df_labels["label_sleep"].value_counts().sum()))
# print number of labels per participant using groupby
test = df_labels.groupby("device_id")["label_sleep"].value_counts().unstack().fillna(0)

#region find out how many labels have sensor data: visualize as table and barplot
segment_around_events = 90 # timeperiod considered around events
min_sensordata_percentage = 50 #in percent; only sensordata above this threshold will be considered
gps_accuracy_min = 100 # in meters; only gps data with accuracy below this threshold was considered when counting GPS records for each event
label_column_name = "label_sleep"
only_active_smartphone_sessions = "yes"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/sleep/data_exploration/labels/"
df_esm_including_number_sensordata = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/esm-events_with-number-of-sensorrecords_segment-around-events-" + str(segment_around_events) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_gps-accuracy-min-" + str(gps_accuracy_min) + " .csv")
# drop row with "nan" in ESM_timestamp
df_esm_including_number_sensordata = df_esm_including_number_sensordata[df_esm_including_number_sensordata["ESM_timestamp"].notnull()]
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
sensors_included = ["accelerometer", "gyroscope", "linear_accelerometer", "locations", "magnetometer", "rotation" ]
#sensors_included = ["accelerometer", "gyroscope", "linear_accelerometer", "magnetometer", "rotation" ]

## visualize as table
df_esm_including_sensordata_above_threshold, df_esm_user_class_counts = data_exploration_labels.create_table_user_activity_including_sensordata(df_esm_including_number_sensordata, dict_label, label_column_name , sensors_included, segment_around_events)
print("Number of Events for which all Sensor Data is available: " + str(len(df_esm_including_sensordata_above_threshold)))
df_esm_user_class_counts.to_csv(path_storage + "location_user-class-counts_event-segments-" + str(segment_around_events) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".csv")
## this is necessary for the following visualiztion of sample events: from this df will get the relevant ESM timestamps
df_esm_including_sensordata_above_threshold.to_csv(path_storage + "location_events_with-sensor-data_event-segments-" + str(segment_around_events) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".csv")

## visualize as barplot
fig = data_exploration_labels.visualize_esm_activity(df_esm_including_sensordata_above_threshold, "label_sleep", "Frequent Locations - Number of Answers per Class with Sensor Data")
fig.savefig(path_storage + "human_motion_class-counts_event-segments-" + str(segment_around_events) + "_only-active-smartphone-sessions-" + str(only_active_smartphone_sessions) + "_min-sensor-percentage-" + str(min_sensordata_percentage) + "-sensors-" + str(sensors_included) + "_min-gps-accuracy-" + str(gps_accuracy_min) + ".png")
 #endregion

#endregion

#region data exploration sensors
# region visualize time of events x before/after sleep for every participant
label_column_name = "label_sleep"
# list_activities = ["lying in bed after sleeping ", "lying in bed before sleeping "]
list_activities = ["lying in bed before sleeping", "lying in bed after sleeping"]
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/sleep/data_exploration/sensors/time/"
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
# merge participant IDs
df_esm = Merge_Transform.merge_participantIDs(df_esm, users, device_id_col = None, include_cities = False)

legend_label = "Classes"
fig_title = "Times of Before/After Sleeping Events"

fig = vis_time_of_events(df_esm, label_column_name, list_activities, fig_title, legend_label)
#save figure
fig.savefig(path_storage + "time_of_events_" + label_column_name + ".png", dpi=600, bbox_inches='tight')


def vis_time_of_events(df_esm, label_column_name, list_activities, fig_title, legend_label):

    # create dataframe including only the relevant activities
    # esm_final = esm_final[esm_final[activity_type].str.contains("|".join(list_activities), na=False)]
    df_esm = df_esm[df_esm[label_column_name].isin(list_activities)]

    # create local timestamp (IMPORTANT that timestamp has still unix format for this step!)
    df_esm = Merge_Transform.add_local_timestamp_column(df_esm, users)

    # convert timestamp into datetime using ms
    df_esm["timestamp"] = pd.to_datetime(df_esm["timestamp"], unit='ms')

    # create column containing the hour of the event

    df_esm["timestamp_hour"] = pd.to_datetime(df_esm['timestamp_local'], unit='ms').astype("string").str[11:16]
    # esm_final["timestamp_hour"]  = datetime.strptime(esm_final["timestamp_hour"][i], '%H:%M')
    df_esm = df_esm.reset_index(drop=True)
    df_esm["Hour of Day"] = 0
    for i in range(0, len(df_esm)):
        df_esm["Hour of Day"][i] = datetime.datetime.strptime(df_esm["timestamp_hour"][i], '%H:%M')
        df_esm["Hour of Day"][i] = (datetime.datetime.combine(datetime.date.min, df_esm["Hour of Day"][
            i].time()) - datetime.datetime.min).total_seconds() / 3600

    # visualize time of events x participant x activity
    ## rename device_id to Participant ID for plot
    df_esm = df_esm.rename(columns={"device_id": "Participant ID"})
    import matplotlib.dates as mdates
    # create plot
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle(fig_title, fontsize=18)
    sns.scatterplot(x="Hour of Day", y="Participant ID", hue=label_column_name, data=df_esm, ax=axs)
    axs.set_xlim(0, 24)
    axs.set_xticks(range(1, 24))
    # label legend
    axs.legend(title=legend_label)
    plt.tight_layout()
    plt.show()
    # return figure
    return fig


# endregion

#endregion
#endregion

#region data preparation for sleep


#endregion




#region OUTDATED: first approach which resulted in 60% accuracy (4 sleep classes)
## task: train DF on GPS-features, frequent locations, human motion model output and time of day
##set parameters
path_locations_classifications = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/locations_classifications.pkl"
path_human_motion_model = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/features-motion-1s-gps-1s/" + "label_human motion - general_timeperiod_around_event-60_parameter_tuning-no_test_proband-17_model.sav"
path_features_for_human_motion_model = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/" + "activity-label_human motion - general_sensor-highfrequencysensorsAll(1seconds)-and-locations(1seconds).pkl"
label_column_name = "label_before and after sleep"

##load data
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
df_labels = pd.DataFrame.from_dict(dict_label, orient="index")
df_motion_and_gps_features = pd.read_pickle(path_features_for_human_motion_model)#load motion features which are used for model predictions
locations_classifications = pd.read_pickle(path_locations_classifications)

df_motion_and_gps_features = df_motion_and_gps_features.drop(columns=["device_id"])#temporary: drop "device_id" column
df_motion_and_gps_features = Merge_Transform.merge_participantIDs(df_motion_and_gps_features, users) #temporary: merge participant ids
df_motion_and_gps_features = labeling_sensor_df(df_motion_and_gps_features, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp") # add labels to df_motion_and_gps_features
df_motion_and_gps_features = df_motion_and_gps_features[df_motion_and_gps_features[label_column_name].notna()] # delete all rows with NaN label

# add human motion model output
human_motion_model = pd.read_pickle(path_human_motion_model)
drop_cols = ["Unnamed: 0", "1", "3", "loc_accuracy", "loc_provider", "loc_device_id", "timestamp_merged",
             "label_before and after sleep", "ESM_timestamp", "device_id", "timestamp",
             "label_human motion - general"] # drop columns which have not been used for training
X_human_motion_model = df_motion_and_gps_features.drop(drop_cols, axis=1).copy()
# add model predictions to GPS features as new column
df_motion_and_gps_features["human motion model predictions"] = human_motion_model.predict(X_human_motion_model)

df_motion_and_gps_features = GPS_computations.classify_locations(df_motion_and_gps_features, locations_classifications) # add frequent locations to GPS features
df_motion_and_gps_features["time of day"] = df_motion_and_gps_features["timestamp"].dt.hour # add time of day
df_motion_and_gps_features["weekday"] = df_motion_and_gps_features["timestamp"].dt.weekday # add weekday

# get overview over labels
df_label_counts = df_labels.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
df_label_counts["total"] = df_label_counts.sum(axis=1)
df_label_counts.loc["total"] = df_label_counts.sum(axis=0)

# train decision forest
### initialize parameters
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_before and after sleep/GPSFeatures, time, locations, human motion/"
n_permutations = 2 # define number of permutations; better 1000
label_segments = [2, 5, 10, 20, 60] #in seconds; define length of segments for label
label_classes = ["lying in bed before sleeping", "lying in bed after sleeping", "lying in bed at other times",
                 "lying on the couch"] # which label classes should be considered
parameter_tuning = "no" # if True: parameter tuning is performed; if False: best parameters are used
feature_columns = ["ESM_timestamp", "loc_device_id", "timestamp", "label_before and after sleep", "label_human motion - general",
                   "speed (km/h)", "acceleration (m/s^2)", "human motion model predictions", "location_classification",
                   "distance_to_location_classification (m)", "time of day", "weekday"] # which columns should be used as features (besides label and timestamp)
# NOTE: I cheated a bit in the feature columns since the "label_human motion - general" is not a feature but a label!

# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 500
max_depth = [2, 5, 15]
min_samples_split = [2, 10, 30]
min_samples_leaf = [1, 5]
max_features = ["sqrt", 3, "log2", 20]
grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features)

df = df_motion_and_gps_features.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
df = df[feature_columns] # keep only the feature columns
df = df[df[label_column_name].isin(label_classes)] # drop rows which don´t contain labels which are in label_classes
df = df.rename(columns={"loc_device_id": "device_id"})
# convert all columns which contain strings and are not the label_column_name to category type and then to numeric type
for column in df.columns:
    if df[column].dtype == "object" and column != label_column_name:
        df[column] = df[column].astype("category")
        df[column] = df[column].cat.codes

# select only data which are in the label_segment around ESM_event & drop columns
for label_segment in label_segments:
    df_decisionforest = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2))) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2)))]
    print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
    # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
    df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
    df_label_counts["total"] = df_label_counts.sum(axis=1)
    df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
    df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")
    df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, grid_search_space, n_permutations, path_storage, parameter_tuning)
    df_decisionforest_results.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-"+ parameter_tuning + "s_df_results.csv")

#endregion
#endregion

#region bathroom-times
#TODO train model when the feature creation & selection for rotation is done
## How to improve the model?
#TODO train LSTM to detect the "walking to bathroom" session
#TODO create classifier from human motion model predictions over a few seconds to identify "walking to bathroom" session and use that as input

##set parameters
path_features = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/rotation/rotation_timeperiod-30_featureselection.pkl"
label_column_name = "label_on the toilet"

##load data
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
df_labels = pd.DataFrame.from_dict(dict_label, orient="index")
df_features = pd.read_pickle(path_features)#load rotation features which are used for model predictions

# get overview over labels
df_label_counts = df_labels.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
df_label_counts["total"] = df_label_counts.sum(axis=1)
df_label_counts.loc["total"] = df_label_counts.sum(axis=0)

# some data transformations
df_features = Merge_Transform.merge_participantIDs(df_features, users) #temporary: merge participant ids
df_features = df_features.dropna(subset=["label_on the toilet"]) #drop rows which don´t contain labels

# train decision forest
## initialize parameters
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_on the toilet/features_rotation/"
n_permutations = 2 # define number of permutations; better 1000
label_segments = [60, 20, 10, 5, 2] #in seconds; define length of segments for label
label_classes = ["on the toilet", "sitting (not on the toilet)"] # which label classes should be considered
parameter_tuning = "no" # must be a string: so "yes" or "no"
drop_cols = [] # columns that should be dropped

# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 500
max_depth = [2, 5, 15]
min_samples_split = [2, 10, 30]
min_samples_leaf = [1, 5]
max_features = ["sqrt", 3, "log2", 20]
grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features)

df = df_features.copy()
df.columns = [column.replace("sensor_timestamp", "timestamp") for column in df.columns] #rename columns: if "sensor_timestamp" is in column name, replace it with "timestamp"
df.columns = [column.replace("loc_device_id", "device_id") for column in df.columns]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
df = df.drop(columns=drop_cols)
df = df[df[label_column_name].isin(label_classes)] # drop rows which don´t contain labels which are in label_classes

# select only data which are in the label_segment around ESM_event & drop columns
for label_segment in label_segments:
    df_decisionforest = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2))) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2)))]
    print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
    # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
    df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
    df_label_counts["total"] = df_label_counts.sum(axis=1)
    df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
    df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")
    df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, grid_search_space, n_permutations, path_storage, parameter_tuning)
    df_decisionforest_results.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_parameter_tuning-"+ parameter_tuning + "s_df_results.csv")

#endregion

#endregion

#region Notes
#region Where do I need more data?
## Human Motion: walking, cycling, running
## Public Transport: in public transport; in train
#endregion
#endregion

#testregion
df_test = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Benedikt/linear_accelerometer_labeled_esm_timestamps.csv")