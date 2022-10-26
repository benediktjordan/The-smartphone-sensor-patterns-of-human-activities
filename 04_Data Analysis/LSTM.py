# region import libraries
## visualizations
import matplotlib.pyplot as plt
## for Keras LSTM
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib
#matplotlib.use('TkAgg') # for mac
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg') # for mac
import pickle

from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

#%matplotlib inline
#%config InlineBackend.figure_format='retina'
import time

import seaborn as sns
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

# for LSTM
from sklearn.metrics import accuracy_score

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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

#Feature Importance
import shap

#Statistical Testing
from scipy.stats import binom_test
from sklearn.model_selection import permutation_test_score

#Tune Hyperparameters
import keras_tuner

# note: to get keras_tuner working, I had to:
# downgrade protobuf to a version 3.20.x (I used_ pip install protobuf==3.20.1
# set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# because the following error message occured
# TypeError: Descriptors cannot not be created directly.
#If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
#If you cannot immediately regenerate your protos, some other possible workarounds are:
# 1. Downgrade the protobuf package to 3.20.x or lower.
# 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters



# endregion

# region data preparation: create merged timeseries
## import data
base_sensor = "linear_accelerometer"
sensors = [ "rotation"]

#sensors = ["accelerometer", "gravity",
#           "gyroscope", "magnetometer", "rotation"]



## merging function
def merge_unaligned_timeseries(df_base, df_tomerge, merge_sensor):
    df_final = pd.DataFrame()
    user_count = 0

    # iterate through devices and ESM_timestamps
    for user in df_base['2'].unique():
        time_a = time.time()
        print("Current user is: ", user)

        df_base_user = df_base[df_base['2'] == user]
        for event in df_base_user['ESM_timestamp'].unique():
            #print("Current event is: ", event)
            # get data for specific user and ESM event
            df_base_user_event = df_base_user[(df_base['2'] == user) & (df_base_user['ESM_timestamp'] == event)]
            df_sensor_user_event = df_tomerge[(df_tomerge['2'] == user) & (df_tomerge['ESM_timestamp'] == event)]

            # sort dataframes by timestamp
            df_base_user_event = df_base_user_event.sort_values(by='timestamp')
            df_sensor_user_event = df_sensor_user_event.sort_values(by='timestamp')

            # duplicate timestamp column for test purposes
            df_sensor_user_event['timestamp_' + str(merge_sensor)] = df_sensor_user_event['timestamp']

            # delete all ESM-related columns in df_sensor_user_event (otherwise they would be duplicated)
            df_sensor_user_event = df_sensor_user_event.drop(
                columns=['ESM_timestamp', "ESM_location", "ESM_location_time",
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
            # TODO: include functionality so that also sensors with lesser frequency can be merged (i.e.
            #  locations, open_wheather etc.)

            # add merged data to 00_general dataframe
            df_final = df_final.append(df_merged)

        time_b = time.time()
        print("User " + str(user_count) + "/" + str(len(df_base['2'].unique())))
        print("Time for user: ", time_b - time_a)
        user_count += 1

    return df_final


## iterate through sensors
df_base = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/" + str(
    base_sensor) + "_esm_timeperiod_5 min.csv_JSONconverted.csv",
                      parse_dates=['timestamp'], infer_datetime_format=True)

# region temporary
df_base = pd.read_csv(
    "/Users/benediktjordan/Documents/MTS/Iteration01/Data/magnetometer_esm_timeperiod_5 min_TimeseriesMerged.csv",
    parse_dates=['timestamp', "1", "timestamp_accelerometer"], infer_datetime_format=True)

test = df_base["1"] - df_base["timestamp"]
test2 = df_base["1"] - df_base["timestamp_accelerometer"]
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
                                  "ESM_activity_time_x": "ESM_activity_time",
                                  "ESM_smartphonelocation_x": "ESM_smartphonelocation",
                                  "ESM_smartphonelocation_time_x": "ESM_smartphonelocation_time",
                                  "ESM_aligned_x": "ESM_aligned",
                                  "ESM_aligned_time_x": "ESM_aligned_time"})

# endregion temporary
#TODO: also merge sensors with lesser frequency (i.e. locations, open_weather etc.)

for sensor in sensors:
    time_begin = time.time()
    print("Current sensor is: ", sensor)
    df_sensor = pd.read_csv(
        "/Users/benediktjordan/Documents/MTS/Iteration01/Data/" + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.csv",
        parse_dates=['timestamp'], infer_datetime_format=True)
    df_base = merge_unaligned_timeseries(df_base, df_tomerge=df_sensor, merge_sensor=sensor)
    df_base.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/" + str(
        sensor) + "_esm_timeperiod_5 min_TimeseriesMerged.csv", index=False)
    time_end = time.time()
    print("Time for sensor ", sensor, " is: ", time_end - time_begin)

# save merged data
df_base.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_timeperiod_5 min_TimeseriesMerged.csv",
               index=False)

# endregion





#region old Keras model
# build model (based on: https://towardsdatascience.com/time-series-classification-for-human-activity-recognition-with-lstms-using-tensorflow-2-and-keras-b816431afdff)
    ### check if GPU is available
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# build model
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
          keras.layers.LSTM(
              units=128,
              input_shape=[X_train.shape[1], X_train.shape[2]]
          )
        )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    ### train model
    history = model.fit(
        X_train, y_train,
        epochs=5, #initially: 20
        batch_size=64,
        validation_split=0.1,
        shuffle=True
    )
#### show accuracy
yhat = model.predict(X_test)


#endregion



#region implement LOSOCV

# load test data only 100000 entries
df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/merged/all_esm_timeperiod_5 min_TimeseriesMerged.csv",
    parse_dates=['timestamp', 'ESM_timestamp'], infer_datetime_format=True)

label_column_name = "label_human motion - general"
sensors_included = "all"
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_" + sensors_included + "_transformed_labeled_dict.pkl", 'rb') as f:
    dict_label = pickle.load(f)

# Set time steps and window-shifting size for LSTM
TIME_STEPS = 200
STEP = 40

# select only relevant columns
merged_sensors = ["accelerometer", "gravity", "gyroscope", "linear_accelerometer", "magnetometer", "rotation"]
# get sensor_columns for merged sensors
sensor_columns_list = []
for sensor in merged_sensors:
    sensor_columns_list.append(sensor_columns[sensor])
# combine list elements into one element
sensor_columns_list = [item for sublist in sensor_columns_list for item in sublist]

# set parameters
# decide label-segment: how much time before and after the ESM timestamp should be considered for the label?
label_segment = 10 # in seconds

# add to sensor columns other columns which are necessary for LSTM
sensor_columns_plus_others = sensor_columns_list.copy()
sensor_columns_plus_others.append("timestamp")
sensor_columns_plus_others.append("ESM_timestamp")
sensor_columns_plus_others.append("2")  # the device ID
sensor_columns_plus_others.append("1")  # timestamp of the sensor collection

# get only sensor columns
df = df[sensor_columns_plus_others]

# add label column to sensor data
df = labeling_sensor_df(df, dict_label, label_column_name)
print("Labelling done. Current label is: ", label_column_name)

# balance dataset based on the data exploration
# TODO: IMPROVE BALANCING
## only keep activities with at least 50.000 records
df[label_column_name].value_counts()
df = df[df[label_column_name].isin(df[label_column_name].value_counts()[df[label_column_name].value_counts() > 20000].index)]
df[label_column_name].value_counts()

## only keep data from participants with at least 50.000 records
df['2'].value_counts()
df = df[df['2'].isin(df['2'].value_counts()[df['2'].value_counts() > 50000].index)]
df['2'].value_counts()

# delete all data which is not in ESM_event +- label_segment
print("Shape before deleting data: ", df.shape)
df = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=label_segment)) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=label_segment))]
print("Number of records after deleting all data which is not in ESM_event +- label_segment: ", len(df))

# Make list of all ID's in idcolumn
IDlist = set(df["2"])
print("Number of participants: ", len(IDlist))


## create the dataset-creation function

def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

# create Confusion Matrix Plot Function
def plot_cm(y_true, y_pred, class_names):
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = plt.subplots(figsize=(18, 16))
  ax = sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap=sns.diverging_palette(220, 20, n=7),
      ax=ax
  )

  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.title('Confusion matrix with accuracy: {0:.4f}'.format(acc))
  #ax.set_xticklabels(class_names)
  #ax.set_yticklabels(class_names)
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values
  return plt # ta-da!



#initialize
test_proband = list()
outer_results_acc = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()

#for loop to iterate through LOSOCV "rounds"
t0 = time.time()
for i in IDlist:
    t0_inner = time.time()
    LOOCV_O = str(i)

    # select training and testing data for this LOSOCV round
    df["2"] = df["2"].apply(str)
    df_train = df[df["2"] != LOOCV_O].copy()
    df_test = df[df["2"] == LOOCV_O].copy()

    # create dataset structure needed for LSTM
    X_train, y_train = create_dataset(
    df_train[sensor_columns_list],
    df_train[label_column_name],
    TIME_STEPS, STEP)

    X_test, y_test = create_dataset(
    df_test[sensor_columns_list],
    df_test[label_column_name],
    TIME_STEPS,STEP)

    ## encode categorical labels into numeric values
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    #print(X_train.shape, y_train.shape)

    # this model & tuner implementation is based on the following tutorial: https://medium.com/analytics-vidhya/hypertuning-a-lstm-with-keras-tuner-to-forecast-solar-irradiance-7da7577e96eb

    def build_model(hp):
        model = Sequential()
        model.add(LSTM(hp.Int('input_unit', min_value=32, max_value=512, step=32), return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2])))
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32), return_sequences=True))
        model.add(LSTM(hp.Int('layer_2_neurons', min_value=32, max_value=512, step=32)))
        model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))
        model.add(Dense(y_train.shape[1],
                        activation=hp.Choice('dense_activation', values=['relu', 'sigmoid'], default='relu')))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        return model

    tuner = RandomSearch(
        build_model,
        objective='mse',
        max_trials=2,
        executions_per_trial=1,
        directory='/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/label_human motion - general/models'
    )

    tuner.search(
        x=X_train,
        y=y_train,
        epochs=20,
        batch_size=128,
        validation_data=(X_test, y_test),
    )

    # get the best model from tuner

    best_model = tuner.get_best_models(num_models=1)[0]

    ### evaluate model
    yhat = best_model.predict(X_test)

    ### save model
    save_model(best_model, "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included + "_test_proband-" + str(i) + "_LSTM.h5")

    # evaluate model
    #import accuracy score for LSTM
    acc = accuracy_score(y_test, yhat.round())
    #print("Accuracy of LSTM: ", accuracy_score(y_test, y_pred.round()))
    f1 = f1_score(y_true=y_test, y_pred=yhat.round(), pos_label=1  , average='macro')
    precision = precision_score(y_true=y_test, y_pred=yhat.round(), pos_label=1  , average='macro')
    recall = recall_score(y_true=y_test, y_pred=yhat.round(), pos_label=1, average='macro')

    # store the result
    test_proband.append(i)
    outer_results_acc.append(acc)
    outer_results_f1.append(f1)
    outer_results_precision.append(precision)
    outer_results_recall.append(recall)

    # report progress
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f' % (acc, f1, precision, recall))
    print("The proband taken as test-data for this iteration was " + str(i))

    # Visualize Confusion Matrix
    #figure = plot_cm(enc.inverse_transform(y_test), enc.inverse_transform(yhat), enc.categories_[0])
    #figure.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included + "_test_proband-" + str(i) + '_ConfusionMatrix.png')
    #figure.show()  # ta-da!

    t1_inner = time.time()
print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

#store all results in one dataframe:
df_stats = pd.DataFrame(data=[test_proband, outer_results_acc, outer_results_f1, outer_results_precision, outer_results_recall]).transpose()
df_stats.columns= ["test_proband","acc", "f1","precision", "recall"]
outer_filename1 = "OuterProgrammingResult.csv"
df_stats.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included + "_test_proband-" + str(i) + "_results.csv")

t1 = time.time()
print("The whole process has taken so many seconds: " + str(t1 - t0))

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
#os.system("shutdown /h") #hibernate
#endregion



#for loop to iterate through LOSOCV "rounds"
t0 = time.time()
for i in IDlist:
    t0_inner = time.time()
    LOOCV_O = str(i)

    # select training and testing data for this LOSOCV round
    df["2"] = df["2"].apply(str)
    df_train = df[df["2"] != LOOCV_O].copy()
    df_test = df[df["2"] == LOOCV_O].copy()

    # create dataset structure needed for LSTM
    X_train, y_train = create_dataset(
    df_train[sensor_columns_list],
    df_train[label_column_name],
    TIME_STEPS, STEP)

    X_test, y_test = create_dataset(
    df_test[sensor_columns_list],
    df_test[label_column_name],
    TIME_STEPS,STEP)

    ## encode categorical labels into numeric values
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    #print(X_train.shape, y_train.shape)

    # this model & tuner implementation is based on the following tutorial: https://medium.com/analytics-vidhya/hypertuning-a-lstm-with-keras-tuner-to-forecast-solar-irradiance-7da7577e96eb

    def build_model(hp):
        model = Sequential()
        model.add(LSTM(hp.Int('input_unit', min_value=32, max_value=512, step=32), return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2])))
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32), return_sequences=True))
        model.add(LSTM(hp.Int('layer_2_neurons', min_value=32, max_value=512, step=32)))
        model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))
        model.add(Dense(y_train.shape[1],
                        activation=hp.Choice('dense_activation', values=['relu', 'sigmoid'], default='relu')))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        return model

    tuner = RandomSearch(
        build_model,
        objective='mse',
        max_trials=2,
        executions_per_trial=1
    )

    tuner.search(
        x=X_train,
        y=y_train,
        epochs=20,
        batch_size=128,
        validation_data=(X_test, y_test),
    )

    best_model = tuner.get_best_models(num_models=1)[0]

    ### evaluate model
    yhat = best_model.predict(X_test[0].reshape((1, X_test[0].shape[0], X_test[0].shape[1])))

    ### save model
    save_model(best_model, "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included + "_test_proband-" + str(i) + "_LSTM.h5")

    # evaluate model
    #import accuracy score for LSTM
    acc = accuracy_score(y_test, yhat.round())
    #print("Accuracy of LSTM: ", accuracy_score(y_test, y_pred.round()))
    f1 = f1_score(y_true=y_test, y_pred=yhat.round(), pos_label=1  , average='macro')
    precision = precision_score(y_true=y_test, y_pred=yhat.round(), pos_label=1  , average='macro')
    recall = recall_score(y_true=y_test, y_pred=yhat.round(), pos_label=1, average='macro')

    # store the result
    test_proband.append(i)
    outer_results_acc.append(acc)
    outer_results_f1.append(f1)
    outer_results_precision.append(precision)
    outer_results_recall.append(recall)

    # report progress
	print('>acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f' % (acc, f1, precision, recall))
    print("The proband taken as test-data for this iteration was " + str(i))

    # Visualize Confusion Matrix
    figure = plot_cm(enc.inverse_transform(y_test), enc.inverse_transform(yhat), enc.categories_[0])
    figure.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included + "_test_proband-" + str(i) + '_ConfusionMatrix.png')
    figure.show()  # ta-da!

    t1_inner = time.time()
print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

#store all results in one dataframe:
df_stats = pd.DataFrame(data=[test_proband, outer_results_acc, outer_results_f1, outer_results_precision, outer_results_recall]).transpose()
df_stats.columns= ["test_proband","acc", "f1","precision", "recall"]
outer_filename1 = "OuterProgrammingResult.csv"
df_stats.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included + "_test_proband-" + str(i) + "_results.csv")

t1 = time.time()
print("The whole process has taken so many seconds: " + str(t1 - t0))

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results_acc), std(outer_results_acc)))
print('F1: %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
print('Precision: %.3f (%.3f)' % (mean(outer_results_precision), std(outer_results_precision)))
print('Recall: %.3f (%.3f)' % (mean(outer_results_recall), std(outer_results_recall)))
#os.system("shutdown /h") #hibernate
#endregion









#region: first try of creating LSTM

##load test data
df = pd.read_csv(
    "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/merged/all_esm_timeperiod_5 min_TimeseriesMerged.csv",
    parse_dates=['timestamp', 'ESM_timestamp'], infer_datetime_format=True)

# define some things
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl", 'rb') as f:
    dict_label = pickle.load(f)
label_column_name = "label_human motion - general"



# select only relevant columns
merged_sensors = ["accelerometer", "gravity", "gyroscope", "linear_accelerometer", "magnetometer", "rotation"]
# get sensor_columns for merged sensors
sensor_columns_list = []
for sensor in merged_sensors:
    sensor_columns_list.append(sensor_columns[sensor])
# combine list elements into one element
sensor_columns_list = [item for sublist in sensor_columns_list for item in sublist]

# set parameters
# decide label-segment: how much time before and after the ESM timestamp should be considered for the label?
label_segment = 30 # in seconds


# define LSTM function
def lstm(df, dict_label, label_column_name, label_segment, sensor_colums_list):

# add to sensor columns other columns which are necessary for LSTM
sensor_columns_plus_others = sensor_columns_list.copy()
sensor_columns_plus_others.append("timestamp")
sensor_columns_plus_others.append("ESM_timestamp")
sensor_columns_plus_others.append("2")  # the device ID
sensor_columns_plus_others.append("1")  # timestamp of the sensor collection

# get only sensor columns
df = df[sensor_columns_plus_others]

# add label column to sensor data
df = labeling_sensor_df(df, dict_label, label_column_name)


# region Human Activity Recognition
## iterate through ESM events and keep only data that is within the label segment

# data visualization: how much data per label level & participant
## visualize how many labels per label level
df[label_column_name].value_counts().plot(kind='bar')
plt.show()

sns.countplot(x=label_column_name, data=df, order=df[label_column_name].value_counts().index)
plt.title("Records per activity")
plt.show()


## how much data per participant
sns.countplot(x='2', data=df, order=df['2'].value_counts().iloc[:10].index)
plt.title("Records per user")
plt.show()


## plot some sensor data for different activities
def plot_activity(ESM_category, activity, df):
    data = df[df[ESM_category] == activity][['acc_double_values_0', 'acc_double_values_1', 'acc_double_values_2']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
    plt.show()

plot_activity(label_column_name, "Sitting", df)

# balance dataset based on the data exploration
# TODO: IMPROVE BALANCING
## only keep activities with at least 50.000 records
df = df[df[label_column_name].isin(df[label_column_name].value_counts()[df[label_column_name].value_counts() > 50000].index)]
df[label_column_name].value_counts()

## only keep data from participants with at least 50.000 records
df = df[df['2'].isin(df['2'].value_counts()[df['2'].value_counts() > 50000].index)]
df['2'].value_counts()

# scale the data

# convert labels into numeric data

# split train & test data (leave two users out for testing)

## randomly choose one user for testing
test_user = df['2'].sample(1).values[0]
df_test = df[df['2'] == test_user]
df_train = df[df['2'] != test_user]

## create the dataset

def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

TIME_STEPS = 200
STEP = 40

X_train, y_train = create_dataset(
    df_train[sensor_columns_list],
    df_train[label_column_name],
    TIME_STEPS, STEP)

X_test, y_test = create_dataset(
    df_test[sensor_columns_list],
    df_test[label_column_name],
    TIME_STEPS,STEP)

print(X_train.shape, y_train.shape)

## encode categorical labels into numeric values
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)
print(X_train.shape, y_train.shape)

## build model (based on: https://towardsdatascience.com/time-series-classification-for-human-activity-recognition-with-lstms-using-tensorflow-2-and-keras-b816431afdff)
### check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

### build model
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=128,
          input_shape=[X_train.shape[1], X_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

### train model
history = model.fit(
    X_train, y_train,
    epochs=5, #initially: 20
    batch_size=64,
    validation_split=0.1,
    shuffle=True
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

### evaluate model
#### show accuracy
y_pred = model.predict(X_test)

# compute accuracy of LSTM
#import accuracy score for LSTM
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred.round())
print("Accuracy of LSTM: ", accuracy_score(y_test, y_pred.round()))

#### plot confusion matrix

def plot_cm(y_true, y_pred, class_names):
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = plt.subplots(figsize=(18, 16))
  ax = sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap=sns.diverging_palette(220, 20, n=7),
      ax=ax
  )

  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.title('Confusion matrix with accuracy: {0:.4f}'.format(acc))
  #ax.set_xticklabels(class_names)
  #ax.set_yticklabels(class_names)
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values
  return plt # ta-da!

figure = plot_cm(
  enc.inverse_transform(y_test),
  enc.inverse_transform(y_pred),
  enc.categories_[0]
)
figure.show()  # ta-da!






fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/decision_forest/" + label_column_name + "/sensors_included-" + sensors_included + "_timeperiod_segments-" + timeperiod_segments + "_test_proband-' + str(i) + '_SHAPFeatureImportance.png')

#endregion