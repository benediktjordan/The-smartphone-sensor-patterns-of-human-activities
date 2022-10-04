# region import libraries
## for Keras LSTM
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

%matplotlib inline
%config InlineBackend.figure_format='retina'
import time

import seaborn as sns
from scipy import stats
from sklearn.preprocessing import OneHotEncoder


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

            # add merged data to general dataframe
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

# load test data
df = pd.read_csv(
    "/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_timeperiod_5 min_TimeseriesMerged.csv",
    parse_dates=['timestamp', 'ESM_timestamp'], infer_datetime_format=True)

# region Human Activity Recognition

# decide label-segment: how much time before and after the ESM timestamp should be considered for the label?
label_segment = 30 # in seconds
## iterate through ESM events and keep only data that is within the label segment




# data visualization: how much data per label level & participant
## how many labels per label level
sns.countplot(x='ESM_bodyposition', data=df, order=df['ESM_bodyposition'].value_counts().index)
plt.title("Records per activity")
plt.show()

## how much data per participant
sns.countplot(x='2', data=df, order=df['2'].value_counts().iloc[:10].index)
plt.title("Records per user")
plt.show()


## plot some sensor data for different activities
def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x_axis', 'y_axis', 'z_axis']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))


plot_activity("Sitting", df);

# balance dataset based on the data exploration
# TODO: IMPROVE BALANCING
## only keep activities with at least 50.000 records
df = df[df['ESM_bodyposition'].isin(df['ESM_bodyposition'].value_counts()[df['ESM_bodyposition'].value_counts() > 50000].index)]
df['ESM_bodyposition'].value_counts()

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
    df_train[['lin_double_values_0', 'lin_double_values_1', 'lin_double_values_2',
              'acc_double_values_0', 'acc_double_values_1', 'acc_double_values_2',
              "gra_double_values_0", "gra_double_values_1", "gra_double_values_2",
              'gyr_double_values_0', 'gyr_double_values_1', 'gyr_double_values_2',
              'mag_double_values_0', 'mag_double_values_1', 'mag_double_values_2',
              'rot_double_values_0', 'rot_double_values_1', 'rot_double_values_2']],
    df_train.ESM_bodyposition,
    TIME_STEPS, STEP)

X_test, y_test = create_dataset(
    df_test[['lin_double_values_0', 'lin_double_values_1', 'lin_double_values_2',
              'acc_double_values_0', 'acc_double_values_1', 'acc_double_values_2',
              "gra_double_values_0", "gra_double_values_1", "gra_double_values_2",
              'gyr_double_values_0', 'gyr_double_values_1', 'gyr_double_values_2',
              'mag_double_values_0', 'mag_double_values_1', 'mag_double_values_2',
              'rot_double_values_0', 'rot_double_values_1', 'rot_double_values_2']],
    df_test.ESM_bodyposition,
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

#### plot confusion matrix
from sklearn.metrics import confusion_matrix

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
  ax.set_xticklabels(class_names)
  ax.set_yticklabels(class_names)
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values
  plt.show() # ta-da!

plot_cm(
  enc.inverse_transform(y_test),
  enc.inverse_transform(y_pred),
  enc.categories_[0]
)


