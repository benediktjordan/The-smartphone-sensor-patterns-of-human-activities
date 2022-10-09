#region import
import pandas as pd
import tsfresh
from tsfresh import extract_features


#region feature extraction function
def feature_extraction(df, sensor_column_names, segment_size, time_column_name, ESM_event_column_name):
    '''
    :param df: dataframe with sensor data
    :param df:
    :param sensor_column_names:
    :param segment_size:
    :param time_column_name:
    :param ESM_event_column_name:
    :return:
    '''

    df_final = pd.DataFrame()
    # iterate through ESM events
    for event in df[ESM_event_column_name].unique():
        # select data for event
        df_event = df[df[ESM_event_column_name] == event]

        # reset index
        df_event = df_event.reset_index(drop=True)

        # create ID column for tsfresh (ID column contains the segment number)
        df_event['ID'] = df_event.index // segment_size

        # sort by timestamp
        df_event = df_event.sort_values(by=time_column_name)

        # get only sensor columns
        df_event = df_event[sensor_column_names, 'ID', time_column_name]

        #extract features
        extracted_features = extract_features(df_event, column_id='ID', column_sort=time_column_name)

        # apply feature selection
        features_filtered = select_features(extracted_features, y)

        # add ESM event columns (all ESM event columns)
        features_filtered["ESM_timestamp"] = df_event["ESM_timestamp"][0]
        features_filtered["ESM_location"] = df_event["ESM_location"][0]
        features_filtered["ESM_location_time"] = df_event["ESM_location_time"][0]
        features_filtered["ESM_bodyposition"] = df_event["ESM_bodyposition"][0]
        features_filtered["ESM_bodyposition_time"] = df_event["ESM_bodyposition_time"][0]
        features_filtered["ESM_activity"] = df_event["ESM_activity"][0]
        features_filtered["ESM_activity_time"] = df_event["ESM_activity_time"][0]
        features_filtered["ESM_smartphonelocation"] = df_event["ESM_smartphonelocation"][0]
        features_filtered["ESM_smartphonelocation_time"] = df_event["ESM_smartphonelocation_time"][0]
        features_filtered["ESM_aligned"] = df_event["ESM_aligned"][0]
        features_filtered["ESM_aligned_time"] = df_event["ESM_aligned_time"][0]

        # add to final dataframe
        df_final = df_final.append(features_filtered)

    return df_final

#impute missing values
extracted_features = extracted_features.fillna(extracted_features.mean())
return extracted_features


#region load testing data
df = pd.read_csv(
    "/Users/benediktjordan/Documents/MTS/Iteration01/Data/accelerometer_esm_timeperiod_5 min.csv_JSONconverted.csv",
    parse_dates=['timestamp'], infer_datetime_format=True, nrows= 10000)

# select only necessary columns
df = df[[ 'timestamp', "acc_double_values_0"]]
# remove rows with NaN values
df = df.dropna(how='any')
# create id column which creates a new id for every 200 rows
df['id'] = df.index // 200

extracted_features = extract_features(df, column_id="id", column_sort="timestamp")

# remove NaN features & select only relevant features
# TODO: find out how "relevant" is defined?
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, y)

# IMPORTANT: tsfresh seemed once to not  work with apple silicon; use rosetta 2 to run this code; but finally it now works with Apple Silicon
# here a guide how to create x86 environment: https://towardsdatascience.com/how-to-manage-conda-environments-on-an-apple-silicon-m1-mac-1e29cb3bad12


class computeFeatures:
    # Class Variable
    animal = 'dog'

    # The init method or constructor
    def __init__(self, breed, color):
        # Instance Variable
        self.breed = breed
        self.color = color


    # for battery_charges
    ## calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
    ## battery double_end_timestamp" timestamp

    def battery_charges_features(self, df):
        # calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
        # battery double_end_timestamp" timestamp

        return df



#region TSFEL: alternative library for feature extraction
import tsfel
import zipfile
import numpy as np
import pandas as pd
import wget
import os

# Load the dataset from online repository
wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip', '/Users/benediktjordan/Downloads/test.zip')

# Unzip the dataset
zip_ref = zipfile.ZipFile("/Users/benediktjordan/Downloads/UCI HAR Dataset.zip", 'r')
zip_ref.extractall()
zip_ref.close()

# Store the dataset as a Pandas dataframe.
x_train_sig = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt', dtype='float32')
X_train_sig = pd.DataFrame(np.hstack(x_train_sig), columns=["total_acc_x"])

X_train_sig.head()

cfg_file = tsfel.get_features_by_domain()                                                        # If no argument is passed retrieves all available features
X_train = tsfel.time_series_features_extractor(cfg_file, X_train_sig, fs=50, window_size=250)

#endregion