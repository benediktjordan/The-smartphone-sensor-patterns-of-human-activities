
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
drop_cols = ["Unnamed: 0", "1", "3", "loc_accuracy", "loc_provider", "loc_device_id", "timestamp_merged"] # columns that should be dropped

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

# select only data which are in the label_segment around ESM_event & drop columns
for label_segment in label_segments:
    df_decisionforest = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=(label_segment/2))) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=(label_segment/2)))]
    print("Size of dataset for label_segment: " + str(label_segment) + " is " + str(df_decisionforest.shape))
    df_decisionforest_results = DecisionForest.DF_sklearn(df_decisionforest, label_segment, label_column_name, grid_search_space, n_permutations, path_storage)
    df_decisionforest_results.to_csv(path_storage + label_column_name + "_timeperiod_around_event-" + str(label_segment) + "s_df_results.csv")

#endregion
#print "Hello"
