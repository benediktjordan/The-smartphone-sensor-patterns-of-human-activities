##test

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

# for keeping track
from tqdm import tqdm



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

#region label transformation

# create csv and dictionary which maps users answers to activity classes
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed.csv")
## create .csv
df_esm_including_activity_classes = label_transformation.create_activity_dataframe(df_esm, human_motion, humanmotion_general, humanmotion_specific, before_after_sleep,
                                                        on_the_toilet_sittingsomewhereelse, on_the_toilet, publictransport_non_motorized, publictransport,
                                                        location, smartphonelocation, aligned)

## analytics: compare computed activity classes with user answers
df_esm_including_activity_classes["label_human motion"].value_counts()

df_esm_including_activity_classes.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")

## create dictionary
dict_esm_including_activity_classes = label_transformation.create_activity_dictionary_from_dataframe(df_esm_including_activity_classes)

with open(
        "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl",
        'wb') as f:
    pickle.dump(dict_esm_including_activity_classes, f, pickle.HIGHEST_PROTOCOL)


#endregion

# region sensor transformation
# merge all screen-sensor data
# Note: this is needed for adding the beginning and end of sessions to all ESM events
dir_databases = "/Volumes/INTENSO/In Usage new/Databases/"
sensor = "screen"
df_sensors_all = Merge_Transform.join_sensor_files(dir_databases, sensor, sensor_appendix = None)
df_sensors_all = Merge_Transform.convert_json_to_columns(df_sensors_all, "screen")
df_sensors_all.to_csv(dir_databases + "screen_all.csv")

# add to each ESM-event the beginning and end of the smartphone session
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
df_screen = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/screen_all.csv")
## convert all columns in which "time" is contained to datetime
for col in df_esm.columns:
    if "time" in col:
        df_esm[col] = pd.to_datetime(df_esm[col], unit="ms")
## add beginning and end of smartphone session to each ESM-event
df_esm_with_screen = Merge_Transform.add_smartphone_session_start_end_to_esm(df_esm, df_screen)

# add duration between beginning and and of ESM answer sessiong to each ESM-event
## compute duration between first and last answer
df_esm_with_screen["duration_between_first_and_last_answer (s)"] = (df_esm_with_screen["smartphonelocation_time"] - df_esm_with_screen[
    "location_time"]) / np.timedelta64(1, 's')
## drop rows for which the time it took between first and last ESM answer is above 5 minutes
df_esm_with_screen = df_esm_with_screen[df_esm_with_screen["duration_between_first_and_last_answer (s)"] < 5*60]

# Rules (NOT DOCUMENTED YET):
## ESM_timestamp is timestamp when "Send" button for first answer is clicked -> IMPLEMENT!!
## events are discarded if more than 5 minutes between first and last answer (location and smartphone location) (reduces number of events from 1113 to 1065)
## NOTE: there are events for which no smartphone session start or no end could be found


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
# visualize label data distribution
path_esm = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv"
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/labels/"

df_esm = data_exploration_labels.load_esm(path_esm)
## visualize for every answer and every activity the number of classes
data_exploration_labels.visualize_esm_activities_all(dir_results,df_esm)
## visualize in one plot the number of labels per activity (bar plot)
data_exploration_labels.visualize_esm_notNaN(dir_results, df_esm)


# calculate number of sensor-datapoints for each event
# find out how many labels have sensor data -> ##temporary -> make later into function!
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
gps_accuracy_min = 35
time_period = 90 #in seconds; timeperiod around events which will be included
sensors = ["linear_accelerometer", "gyroscope", "magnetometer", "barometer", "rotation", "locations"]
path_sensor_database = "/Users/benediktjordan/Downloads/"
df_esm = data_exploration_sensors.volume_sensor_data_for_events(df_esm, time_period, sensors, path_sensor_database, gps_accuracy_min)
df_esm.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/esm-events_with-number-of-sensorrecords_segment-around-events-" + str(time_period) + "_gps-accuracy-min-" + str(gps_accuracy_min) + " .csv")

# calculate mean/std & max of GPS and linear accelerometer data (over different time periods)
## Note: this data is relevant for, for example, public transport -> data exploration -> visualizing mean & std of GPS data
## calculate summary stats for GPS
dir_sensors = "/Users/benediktjordan/Downloads/"
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/summary_stats/"
sensors = ["GPS", "linear_accelerometer"]
sensors = [ "linear_accelerometer"]
test = pd.read_pickle("/Users/benediktjordan/Downloads/linear_accelerometer_esm_timeperiod_5 min.csv_JSONconverted.pkl")

gps_accuracy_min_list = [35]
time_periods = [10,20,40, 90, 180]
for sensor in sensors:
    for time_period in time_periods:
        for gps_accuracy_min in gps_accuracy_min_list:
            if sensor == "GPS":
                path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(gps_accuracy_min) + ".csv"
            else:
                path_sensor = dir_sensors + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.pkl"

            print("Started with time period: " + str(time_period))
            summary_statistics, missing_sensordata = data_exploration_sensors.create_summary_statistics_for_sensors(path_sensor, sensor, time_period, dir_results)
            missing_sensordata_df = pd.DataFrame(missing_sensordata)
            if sensor == "GPS":
                summary_statistics.to_csv(dir_results + sensor + "_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + ".csv",
                                          index=False)
                missing_sensordata_df.to_csv(dir_results + sensor + "_summary-stats_gps-accuracy-min-" + str(gps_accuracy_min) + "_time-period-around-event-" + str(time_period) + "_missing_sensordata.csv", index=False)
            else:
                summary_statistics.to_csv(dir_results + sensor + "_summary-stats_time-period-around-event-" + str(time_period) + ".csv", index=False)
                missing_sensordata_df.to_csv(dir_results + sensor + "_summary-stats_time-period-around-event-" + str(time_period) + "_missing_sensordata.csv", index=False)



#endregion

#region calcualate distance, speed & acceleration
# load data around events
df_locations_events = pd.read_pickle("/Users/benediktjordan/Downloads/locations_esm_timeperiod_5 min.csv_JSONconverted.pkl")

#drop duplicates (this step should be integrated into data preparation)
df_locations_events = df_locations_events.drop(columns=["Unnamed: 0", "0"])
df_locations_events = df_locations_events.drop_duplicates()

# calculate distance, speed and acceleration (class build in "FeatureExtraction_GPS.py")
df_features = FeatureExtraction_GPS().calculate_distance_speed_acceleration(df_locations_events)

# drop rows where distance, speed or acceleration contain NaN (they contain NaN if it is first entry of every event)
df_features= df_features.dropna(subset=["distance (m)", "speed (km/h)", "acceleration (m/s^2)"])
# drop rows with unrealistic speed values  & GPS accuracy values
df_features = df_features[df_features["speed (km/h)"] < 300]
# create different GPS accuracy thresholds
accuracy_thresholds = [10, 35, 50, 100] #accuracy is measured in meters
for accuracy_threshold in accuracy_thresholds:
    df_features_final = df_features[df_features["loc_accuracy"] < accuracy_threshold]
    df_features_final = df_features_final.reset_index(drop=True)
    print("Number of rows with accuracy < " + str(accuracy_threshold) + "m: " + str(len(df_features_final)))
    #save to csv
    df_features_final.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(accuracy_threshold) + ".csv")
#endregion

#region different activities

#region human motion
#TODO add noise removal & normalization to data preparation. Here are some ideas from the HAR review paper: Concerning the Noise Removal step, 48 CML-based articles and 12 DL-based articles make use of different noise removal techniques. Among all such techniques the most used ones are: z-normalization [75], [120], min-max [70], [127], and linear interpolation [102], [111] are the most used normalization steps, preceded by a filtering step based on the application of outlier detection [70], [117], [163], Butterworth [82], [101], [117], [123], [127], [128], [152], [155], [174], [189], median [74], [101], [117], [127], [132], [147], [155], [182], [183], high-pass [92], [96], [117], [128], [169], [173], [208], or statistical [58] filters.
#TODO create other features (using tsfresh). Here are the ones most used in the HAR literature (compare review): compare my documentation!

#region Laboratory Data

#region data transformation for human motion
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

# label sensordata & add ESM timestamp
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
        df_sensor = InitialTransforms_Iteration02.label_sensor_data(path_sensorfile, dict_label) #calculate labels
        if df_sensor is None or df_sensor.empty: #check if sensorfile is empty
            continue
        #analtics: compare number of unique labels with number of labels in dict_label
        df_sensor.to_csv(dir_dataset + user["Name"] + "/" + sensor["Sensor"]  + "_labeled.csv", index=False)
        df_sensor, dict_esm_labels = InitialTransforms_Iteration02.add_esm_timestamps(df_sensor, dict_label, length_of_esm_events) #add ESM_timestamp
        #analytics: find how many ESM_timestamps are in the dataset and how many are included in dict_esm_labels
        print("Number of ESM_timestamps in dataset: " + str(len(df_sensor["ESM_timestamp"].unique())))
        print("Number of ESM_timestamps in dict_esm_labels: " + str(len(dict_esm_labels)))
        print("Percentage of ESM_timestamps in dict_esm_labels: " + str(len(dict_esm_labels)/len(df_sensor["ESM_timestamp"].unique())))
        if int(sensor["Frequency (in Hz)"]) != 0:# analytics: find out how many labeled sensor minutes are in df_sensor and how many of those minutes are in ESM-events
            num_minutes = len(df_sensor["timestamp"]) / sensor["Frequency (in Hz)"] / 60
            print("Number of labeled sensor minutes for user " + user["Name"] + ": " + str(num_minutes) + " minutes")
            esm_events_total_length = (len(df_sensor["ESM_timestamp"].unique()) * length_of_esm_events)/60
            print("Percentage of labeled sensor minutes that are in ESM events: " + str(esm_events_total_length/num_minutes * 100) + "%")
        df_sensor.to_csv(dir_dataset + user["Name"] + "/" + sensor["Sensor"]  + "_labeled_esm_timestamps.csv", index=False)
        #merge dict_esm_labels to dicts_esm_labels
        dicts_esm_labels.update(dict_esm_labels)
        print("Lenght of dicts_esm_labels after completing user" + user["Name"]  + " and sensor " + sensor["Sensor"] + ":" + str(len(dicts_esm_labels)))
        print("Sensorfile labeled and ESM_timestamps added for user " + user["Name"] + " and sensor " + sensor["Sensor"])
#save dicts_esm_labels as pickle
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

#region data preparation for human motion
#merge high-frequency sensors
sensors_high_frequency = ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]
timedelta = "100ms" #must be in format compatible with pandas.Timedelta
columns_to_delete = ["device_id", "label_human motion - general", "ESM_timestamp"]
sensor_timeseries_are_merged = True

path_datasets = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/"
counter = 1
for sensor in sensors_high_frequency:
    print("start with sensor: " + sensor)
    if counter == 1:
        sensor_base = sensor
        df_base = pd.read_csv(path_datasets + sensor + "_labeled_esm_timestamps_allparticipants.csv")
        df_base = df_base.drop(columns=["timestamp"]) #delete "timestamp" column and rename "timestamp_datetime" to "timestamp"
        df_base = df_base.rename(columns={"timestamp_datetime": "timestamp"})
        for col in df_base.columns: #rename columns of df_base so that they can be still identified later on
        if col != "timestamp" and col != "device_id" and col != "timestamp_merged" and col != "ESM_timestamp" and col != "label_human motion - general":
            df_base = df_base.rename(columns={col: sensor_base[:3] + "_" + col})
        counter += 1
        continue
    if counter > 1:
        sensor_tomerge = sensor
        df_tomerge = pd.read_csv(path_datasets + sensor + "_labeled_esm_timestamps_allparticipants.csv")
        df_tomerge = df_tomerge.drop(columns=["timestamp"]) #delete "timestamp" column and rename "timestamp_datetime" to "timestamp"
        df_tomerge = df_tomerge.rename(columns={"timestamp_datetime": "timestamp"})
        df_base = Merge_and_Impute.merge(df_base, sensor_base, df_tomerge, sensor_tomerge, timedelta, columns_to_delete, sensor_timeseries_are_merged)
        counter += 1
#save to csv
df_base.to_csv("/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/timeseries_merged/highfrequencysensors_allparticipants.csv", index=False)

#region feature extraction: tsfresh features for high-frequency sensors
##NOTE: the following code regarding feature extraction and selection with tsfresh is copied from "pythonProject", as the
## code can only be deployed there (due to requirements of tsfresh); therefore this code needs to be updated regularly

sensors = ["rotation", "magnetometer", "linear_accelerometer", "gyroscope", "accelerometer"]
feature_segments = [30, 1,2,5,10] #in seconds
sensors = ["linear_accelerometer", "gyroscope", "accelerometer"]
feature_segments = [30, 1,2,5,10] #in seconds
frequency = 10 #in Hz
path_sensorfile = "/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/timeseries_merged/highfrequencysensors_allparticipants.csv"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/features/highfrequency_sensors/"

#feature extraction in chunks for every sensor separately
## note: calculating features for every sensor separately is a major improvement since it uses all
## available sensor data; when calculating features for all sensors together, only the records with no
## missing value in any of the sensors were used -> DOCUMENT AS FUCKUP!
for sensor in sensors:
    print("start with sensor: ", sensor)
    sensor_columns = []
    sensor_columns.append(database_sensor_columns[sensor])
    sensor_columns = [item for sublist in sensor_columns for item in sublist]

    for feature_segment in tqdm(feature_segments):
    # iterate through sensor dataframe in steps of 500000
        chunksize_counter = 1
        for df_sensor in pd.read_csv(path_sensorfile, chunksize=100000):
            # print the current time
            print(f"date: {datetime.datetime.now()}")
            # jump over chunksizes which are already done
            # if chunksize counter is smaller than 2, continue
            # if chunksize_counter < 51 and seconds == 1:
            #    print("jump over chunksize "+str(chunksize_counter))
            #    chunksize_counter += 1
            #    continue

            time.start = time.time()
            df_features = computeFeatures.feature_extraction(df_sensor, sensor_columns, feature_segment,
                                                             time_column_name="timestamp",
                                                             ESM_event_column_name="ESM_timestamp")
            # check if df_features is an empty DataFrame; if so, continues with next chunksize
            if df_features.empty:
                print("df_features is empty for chunksize ", chunksize_counter, " in time_period ", seconds,
                      ", was removed, continuing with next chunk")
                chunksize_counter += 1
                continue

            print(f"date: {datetime.datetime.now()}")
            print(
                "Time for " + str(feature_segment) + " seconds: " + str((time.time() - time.start) / 60) + " - without saving")

            # save features with pickle
            path_features = path_storage + sensor + "_feature_segment-" + str(
                feature_segment) + "s_tsfresh-features-extracted_chunknumber" + str(chunksize_counter) + ".pkl"

            with open(path_features, 'wb') as f:
                pickle.dump(df_features, f, pickle.HIGHEST_PROTOCOL)

            # df_features.to_csv(dir_sensorfiles + "features/" + str(seconds) + " seconds_high_frequency_sensors.csv")
            print("Time for sensor" + sensor + " and feature segment "+ str(feature_segment) + " seconds and chunknumber: " + str(chunksize_counter) + ":" + str(
                (time.time() - time.start) / 60) + " with saving")
            # print shape of df_features
            print(df_features.shape)
            # increase chunksize_counter
            chunksize_counter += 1

# combine chunks
df_test = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/features/highfrequency_sensors/linear_accelerometer_feature_segment-1s_tsfresh-features-extracted_chunknumber1.pkl")
df_test2 = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/features/highfrequency_sensors/linear_accelerometer_feature_segment-1s_tsfresh-features-extracted_chunknumber2.pkl")

sensors = ["rotation", "magnetometer", "linear_accelerometer", "gyroscope", "accelerometer"]
time_periods = [1,2,5,10,30] #in seconds
chunksize_counter = 2
for sensor in sensors:
    for second in time_periods:
        # create empty dataframe
        df_features = pd.DataFrame()
        # iterate through all chunks
        for chunknumber in range(1, chunksize_counter+1):

            # try to load chunk; if doesn´t exist - continue
            try:
                path_features = path_storage + sensor + "_feature_segment-" + str(second) + "s_tsfresh-features-extracted_chunknumber" + str(chunknumber)+ ".pkl"
                with open(path_features, 'rb') as f:
                    df_features_chunk = pickle.load(f)
            except:
                print("sensor "+ sensor + "_chunknumber ", chunknumber, " in time_period ", second, " does not exist, continuing with next chunk")
                continue

            # load chunk which exists
            path_features = path_storage + sensor + "_feature_segment-" + str(
                second) + "s_tsfresh-features-extracted_chunknumber" + str(chunknumber) + ".pkl"
            with open(path_features, 'rb') as f:
                df_features_chunk = pickle.load(f)

            # print size of chunk and df_features
            print("sensor " + sensor + "chunknumber ", chunknumber, " in time_period ", second, " has size ", df_features_chunk.shape)

            # concatenate chunk to df_features
            df_features = pd.concat([df_features, df_features_chunk], axis=0)
            print("df_features has size ", df_features.shape)

            print("chunknumber ", chunknumber, " in time_period ", second, " loaded and concatenated")

        # reset index
        df_features.reset_index(drop=True, inplace=True)

        # save df_features
        path_features = path_storage +  sensor + "_feature-segment-" + str(second) + "s_tsfresh-features-extracted.pkl"
        with open(path_features, 'wb') as f:
            pickle.dump(df_features, f, pickle.HIGHEST_PROTOCOL)

## combine different sensor files into one file
## concatenate dataframes based on sensor_timestamp column and device_id column
#Note: important to merge based not only on timestamp BUT ALSO on device_id, as for TIna & Selcuk the timestamp is the same
#(as well as the ESM_timestamp)
sensors = ["rotation", "magnetometer", "linear_accelerometer", "gyroscope", "accelerometer"]
feature_segments = [1,2,5,10,30] #in seconds
for feature_segment in feature_segments:
    df_features = pd.DataFrame()
    counter = 1
    print("start with feature segment ", feature_segment)
    for sensor in sensors:
        print("start with sensor ", sensor)
        path_features = path_storage + sensor + "_feature-segment-" + str(feature_segment) + "s_tsfresh-features-extracted.pkl"
        with open(path_features, 'rb') as f:
            df_features_sensor = pickle.load(f)
            print("df_features_sensor has size ", df_features_sensor.shape)
        if counter == 1:
            df_features = df_features_sensor
            print("df_features has size ", df_features.shape)
            counter += 1
            continue
        # if not first sensor merged, delete "ESM_timestamp", "device_id" columns
        df_features_sensor.drop(columns=["ESM_timestamp"], inplace=True)
        # merge dataframes based on sensor_timestamp column
        df_features = pd.merge(df_features, df_features_sensor, on=['sensor_timestamp', "device_id"], how = "outer")
        print("df_features has size ", df_features.shape)
        counter += 1
    # save df_features
    path_features = path_storage + "highfrequency-sensors_feature-segment-" + str(feature_segment) + "s_tsfresh-features-extracted.pkl"
    with open(path_features, 'wb') as f:
        pickle.dump(df_features, f, pickle.HIGHEST_PROTOCOL)
#endregion

# feature selection of high frequency sensors: data driven approach
##NOTE: the following code regarding feature extraction and selection with tsfresh is copied from "pythonProject", as the
## code can only be deployed there (due to requirements of tsfresh); therefore this code needs to be updated regularly
label_column_name = "label_human motion - general"
feature_segments = [30,10,5,2,1] # second 1 is excluded as my system always crushes
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/features/highfrequency_sensors/"

time_column_name = "timestamp"
ESM_identifier_column="ESM_timestamp"
with open("/Users/benediktjordan/Documents/MTS/Iteration02/datasets/dicts_esmtimestamp-label-mapping.pkl",
        'rb') as f:
    dict_label = pickle.load(f)

for feature_segment in feature_segments:
    print("seconds started: ", feature_segment)

    # exclude labels which have been processed already and seconds which have been processed already
    #if label_column_name == "label_human motion - general" and seconds == 1:
    #    print("label_human motion - general and 1 seconds already processed")
    #    continue

    t0 = time.time()
    # load df_features
    #path_features = dir_sensorfiles + "data_preparation/features/highfrequencysensors-" + str(sensors_included) + "_timeperiod-" + str(seconds) + " s.pkl"
    path_features = path_storage + "highfrequency-sensors_feature-segment-" + str(feature_segment) + "s_tsfresh-features-extracted.pkl"
    df_features = pd.read_pickle(path_features)

    #temporary: only select part of rows
    #df_features = df_features.iloc[0:10000, :].copy()
    #print("df_features loaded")
    #temporary set first row device_id == 1
    #df_features.at[0, "device_id"] = 1

    #temporary: drop column "timestamp_beginning_of_feature_segment"
    #df_features.drop(columns=["timestamp_beginning_of_feature_segment"], inplace=True)

    #temporary: drop column "timestamp_beginning_of_feature_segment"

    features_filtered = computeFeatures.feature_selection(df_features, label_column_name)

    # save df_features
    #path_features = dir_sensorfiles + "data_preparation/features/activity-" + label_column_name + "_highfrequencysensors-all_timeperiod-" + str(seconds) + " s_featureselection.pkl"
    path_features = path_storage + "activity-" + label_column_name  + "_feature-segment--" + str(feature_segment) + " s_featureselection.pkl"
    with open(path_features, 'wb') as f:
        pickle.dump(features_filtered, f, pickle.HIGHEST_PROTOCOL)
    print("df_features saved")


# feature creation for GPS data

# feature selection of high frequency sensors and GPS data: hypothesis driven approach

#testcase
df_test = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration02/data_preparation/features/highfrequency_sensors/highfrequency-sensors_timeperiod-30_tsfresh-features-extracted.pkl")
all_features = df_test.columns
# get list of features in all_features which contain "fft"
fft_features = [feature for feature in all_features if "fft" in feature]
lin_fft_features = [feature for feature in fft_features if "lin_double_values" in feature]
#convert lin_fft_features to dataframe
df_lin_fft_features = df_test[lin_fft_features]

#endregion

#region modeling with decision forest for human motion
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
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/label_human motion/"
path_sensordata = "/Users/benediktjordan/Downloads/INSERT_esm_timeperiod_5 min.csv_JSONconverted.pkl"
gps_accuracy_min = 10 # meters
path_GPS_features = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(gps_accuracy_min) + ".csv"
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

#create list of event times for which sensor data exists: double check with GPS data according to above accuracy
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


########## THE REST OF THIS REGION ORIGINATES FROM PUBLIC TRANSPORT AND IS NOT YET ADAPTED TO HUMAN MOTION ###########
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

#endregion

#region data preparation for human motion
label_column_name = "label_human motion - general"
# load labels
sensors_included = "all"
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_" + sensors_included + "_transformed_labeled_dict.pkl", 'rb') as f:
    dict_label = pickle.load(f)

# create different combinations of sensors & features
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

#region testing label segments for human motion
label_column_name = "label_human motion - general"
path_features = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-label_human motion - general_sensor-highfrequencysensors-all(1seconds)-and-GPS(1seconds).pkl"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/labelsegments_comparison/features-highfrequencysensors-all-1s-gps-1s/"
n_permutations = 2 # define number of permutations; better 1000
label_segments = list(range(56, 121, 1)) #in seconds; defines label segments to test
label_classes = ["standing", "lying", "sitting", "walking", "cycling"] # which label classes should be considered
parameter_tuning = "no" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = ["Unnamed: 0", "1", "3", "loc_accuracy", "loc_provider", "loc_device_id", "timestamp_merged"] # columns that should be dropped
df = pd.read_pickle(path_features)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
df = df.drop(columns=drop_cols)
df = Merge_Transform.merge_participantIDs(df, users) #temporary: merge participant ids
df = df[df[label_column_name].isin(label_classes)] # drop rows which don´t contain labels which are in label_classes

df_decisionforest_results_all = pd.DataFrame(columns=["Label", "Seconds around Event", "Balanced Accuracy", "Accuracy", "F1", "Precision", "Recall"])

for label_segment in label_segments:
    #check if label_segment is 40 or 50 or 60 seconds; if so, continue
    if label_segment == 40 or label_segment == 50 or label_segment == 60 or label_segment == 70 or label_segment == 80 or label_segment == 90 or label_segment == 100 or label_segment == 110 or label_segment == 120:
        continue
    t0 = time.time()
    df_decisionforest = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2))) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2)))]
    print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
    # create dataframe which counts values of label_column_name grouped by "device_id" and save it to csv
    df_label_counts = df_decisionforest.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
    df_label_counts["total"] = df_label_counts.sum(axis=1)
    df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
    df_label_counts.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "_label_counts.csv")
    df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, n_permutations, path_storage, parameter_tuning)
    df_decisionforest_results_all = pd.concat([df_decisionforest_results_all, df_decisionforest_results], axis=0)
    #intermediate saving
    df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "s_parameter_tuning-" + parameter_tuning + "_results.csv")
    print("finished with label_segment: " + str(label_segment) + " in " + str((time.time() - t0)/60) + " minutes")
df_decisionforest_results_all.to_csv(path_storage + label_column_name + "_timeperiod_around_event-2-120s_parameter_tuning-"+ parameter_tuning + "df_results.csv")

#visualizing the accuracies for different label segments
df_decisionforest_results_all = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/labelsegments_comparison/features-highfrequencysensors-all-1s-gps-1s/label_human motion - general_timeperiod_around_event-29s_parameter_tuning-no_results.csv")
#create lineplot for Seconds around Eventx x Balanaced Accuracy with Seaborn
fig, ax = plt.subplots(figsize=(10, 5))
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.lineplot(x="Seconds around Event", y="Balanced Accuracy", data=df_decisionforest_results_all)
plt.show()
#save plot
fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/labelsegments_comparison/features-highfrequencysensors-all-1s-gps-1s/label_human motion - general_timeperiod_around_event-29s_parameter_tuning-no_results.png")


#endregion

#region modeling for human motion

##region Decision Forest on motion & GPS features
### initialize parameters
parameter_segments = [10] #in seconds; define length of segments of parameters (over which duration they have been created)
#combinations_sensors = [["highfrequencysensors-all", "GPS"],
#                        ["linear_accelerometer", "rotation", "GPS"],
#                        ["linear_accelerometer", "GPS"]]  # define which sensor combinations should be considered
combinations_sensors = [["linear_accelerometer", "rotation", "GPS"]]

label_column_name = "label_human motion - general"
n_permutations = 100 # define number of permutations; better 1000
label_segment = 11 #define how much data around each event will be considered
label_classes = ["standing", "lying", "sitting", "walking", "cycling"] # which label classes should be considered
parameter_tuning = "yes" # if True: parameter tuning is performed; if False: default parameters are used
drop_cols = ["Unnamed: 0", "1", "3", "loc_accuracy", "loc_provider", "loc_device_id", "timestamp_merged"] # columns that should be dropped

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