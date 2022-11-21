
#region import
import pickle
#import tensorflow_decision_forests as tfdf

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time
import matplotlib.pyplot as plt

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

# region data transformation
#TODO merge ID´s if they are from the same participant for "xmin around events" and "merged timeseries" files


# region general data preparation (same for all activities)
## location data preparation

#region calcualate distance, speed & acceleration
# load data around events
df_locations_events = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/locations_esm_timeperiod_5 min.csv_JSONconverted.pkl")

#drop duplicates (this step should be integrated into data preparation)
df_locations_events = df_locations_events.drop(columns=["Unnamed: 0", "0"])
df_locations_events = df_locations_events.drop_duplicates()

# calculate distance, speed and acceleration (class build in "FeatureExtraction_GPS.py")
df_features = FeatureExtraction_GPS().calculate_distance_speed_acceleration(df_locations_events)

# drop rows where distance, speed or acceleration contain NaN (they contain NaN if it is first entry of every event)
df_features = df_features.dropna(subset=["distance (m)", "speed (km/h)", "acceleration (m/s^2)"])
# drop rows with unrealistic speed values
df_features = df_features[df_features["speed (km/h)"] < 300]
df_features = df_features.reset_index(drop=True)

#save to csv
df_features.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/locations-aroundevents_features-distance-speed-acceleration.csv")
#endregion

#region different activities

#region human motion
# load labels
sensors_included = "all"
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_" + sensors_included + "_transformed_labeled_dict.pkl",
        'rb') as f:
    dict_label = pickle.load(f)

## merge motion and GPS features
timedelta = "1000ms" # the time which merged points can be apart; has to be in a format compatible with pd.TimeDelta()
drop_cols = ['device_id', 'label_human motion - general', "ESM_timestamp", "label_human motion - general"] # columns that should be deleted from df_tomerge

df_base = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/locations-aroundevents_features-distance-speed-acceleration.csv") # load df_base: this should be the df with features with less "sampling rate"/frequency; # here: load motion features which are computed for 2 seconds
df_base = labeling_sensor_df(df_base, dict_label, label_column_name = "label_human motion - general" , ESM_identifier_column = "ESM_timestamp")
df_base = df_base.dropna(subset=["label_human motion - general"]) #drop NaN labels
df_tomerge = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-label_human motion - general_highfrequencysensors-all_timeperiod-1 s_featureselection.pkl") # load df_tomerge: this should be the df with features with higher "sampling rate"/frequency
df_final = Merge_and_Impute.merge(df_base, df_tomerge, timedelta, drop_cols) # merge df_base and df_tomerge
#TODO integrate "delete columns with only "NaN" values" into "features selection" section
df_final = df_final.dropna(axis=1, how='all') # delete columns with only "NaN" values
print("So many rows contain missing values after merging: " + str(df_final.isnull().values.any()))
df_final_nan = Merge_and_Impute.impute_deleteNaN(df_final) # delete rows which contain missing values
df_final_nan.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-label_human motion - general_sensor-highfrequencysensorsAll(1seconds)-and-locations(1seconds).pkl")

## train Decision Forest on motion & GPS features
### initialize parameters
df = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-label_human motion - general_sensor-highfrequencysensorsAll(1seconds)-and-locations(1seconds).pkl")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/features-motion-1s-gps-1s/"
n_permutations = 2 # define number of permutations; better 1000
label_segments = [2, 5, 10, 20, 60] #in seconds; define length of segments for label
label_column_name = "label_human motion - general"
label_classes = ["standing", "lying", "sitting"] # which label classes should be considered
parameter_tuning = "no" # if True: parameter tuning is performed; if False: best parameters are used
drop_cols = ["Unnamed: 0", "1", "3", "loc_accuracy", "loc_provider", "loc_device_id", "timestamp_merged"] # columns that should be dropped

# if parameter tuning is true, define search space
n_estimators = [50, 100, 500, 800]  # default 500
max_depth = [2, 5, 15]
min_samples_split = [2, 10, 30]
min_samples_leaf = [1, 5]
max_features = ["sqrt", 3, "log2", 20]
grid_search_space = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf, max_features=max_features)


df["timestamp"] = pd.to_datetime(df["timestamp"])
df["ESM_timestamp"] = pd.to_datetime(df["ESM_timestamp"])
df = df.drop(columns=drop_cols)
df = merge_participantIDs(df, users) #temporary: merge participant ids
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

#region public transport
## what can be improved?
#TODO add also motion features
#TODO train LSTM on GPS data (and maybe motion data)
#TODO add further control-group labels

label_column_name = "label_public transport"
# load labels
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")

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
print("Accuracy is " + str(df_locations["correct_classification"].value_counts()[1]/df_locations["correct_classification"].value_counts().sum()))

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
## How to improve the model?
#TODO train LSTM to detect the "walking to bathroom" session
#TODO create classifier from human motion model predictions over a few seconds to identify "walking to bathroom" session and use that as input

#endregion

#region Notes
#region Where do I need more data?
## Human Motion: walking, cycling, running
## Public Transport: in public transport; in train
#endregion
#endregion