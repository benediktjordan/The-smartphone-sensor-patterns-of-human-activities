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
from sklearn.metrics import balanced_accuracy_score


#Feature Importance
import shap

#Statistical Testing
from scipy.stats import binom_test
from sklearn.model_selection import permutation_test_score

# for scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

#import class weights from sklearn
from sklearn.utils import class_weight

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
from kerastuner.tuners import BayesianOptimization
from kerastuner.tuners import Hyperband

from kerastuner.engine.hyperparameters import HyperParameters
from keras.layers import LeakyReLU



# endregion

# added line



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
# initialize parameters
label_column_name = "label_human motion - general"
sensors_included = ["gyroscope", "linear_accelerometer"] #note: the first sensor in this list has to be the base sensor!
# Set time steps and window-shifting size for LSTM
#TIME_STEPS = 200 #20 seconds if frequency is 10Hz
#STEP = 40
TIME_STEPS = 40 #4 seconds if frequency is 10Hz
STEP = 20 # 2 seconds if frequency is 10Hz
tuner_enabled = "no"
label_segment = 10 # in seconds; how much time before and after the ESM timestamp should be considered for the label?
label_classes_included = ["standing", "walking"]

# load sensor data and labels
sensor_names = ""
counter = 0
for sensor in sensors_included:
    if counter == 0:
        sensor_names = sensor + "_with"
        counter = counter + 1
    else:
        sensor_names = sensor_names + "-" + sensor
        counter = counter + 1
df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/timeseries_merged/" + sensor_names  + "_esm_timeperiod_5 min_TimeseriesMerged.csv",
    parse_dates=['timestamp', 'ESM_timestamp'], infer_datetime_format=True)
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl", 'rb') as f:
    dict_label = pickle.load(f)

class LSTMClassifier:

    # select columns: select only the relevant columns for further processing
    def select_features(self, df, sensors_included):
        sensor_columns_list = []
        sensors_included_names = ""
        for sensor in sensors_included:
            sensor_columns_list.append(sensor_columns[sensor])
            sensors_included_names = sensors_included_names + sensor + "_"

    # preprocessing: delete data which is too far from event; add labels; delete NaN labels; merge IDs of same person

    # exploration: explore label distribution; explore sensor-label connection

    # create scalers

    # create dataset (with the correct time steps & format for LSTM)

    # create tuner

    # create LSTM model

    # create evaluation metrics

    # create confusion metrics

    # create LOSOCV iteration function




# select only relevant columns
#sensors = ["accelerometer", "gravity", "gyroscope", "linear_accelerometer", "magnetometer", "rotation"]
# get sensor_columns for merged sensors
sensor_columns_list = []
sensors_included_text = ""
for sensor in sensors_included:
    sensor_columns_list.append(sensor_columns[sensor])
    sensors_included_text = sensors_included_text + sensor + "_"
# combine list elements into one element
sensor_columns_list = [item for sublist in sensor_columns_list for item in sublist]

# add to sensor columns other columns which are necessary for LSTM
sensor_columns_plus_others = sensor_columns_list.copy()
sensor_columns_plus_others.append("timestamp")
sensor_columns_plus_others.append("ESM_timestamp")
sensor_columns_plus_others.append("device_id")  # the device ID
sensor_columns_plus_others.append("1")  # timestamp of the sensor collection

# get only sensor columns
df = df[sensor_columns_plus_others]

# delete all data which is not in ESM_event +- label_segment/2
print("Shape before deleting data: ", df.shape)
df = df[(df['timestamp'] >= df['ESM_timestamp'] - pd.Timedelta(seconds=label_segment/2)) & (df['timestamp'] <= df['ESM_timestamp'] + pd.Timedelta(seconds=label_segment/2))]
print("Number of records after deleting all data which is not in ESM_event +- label_segment: ", len(df))

# add label column to sensor data
df = labeling_sensor_df(df, dict_label, label_column_name)
print("Labelling done. Current label is: ", label_column_name)

# delete NaN values in label column
print("Shape before deleting NaN values in label column: ", df.shape)
df = df.dropna(subset=[label_column_name])
print("Number of records after deleting NaN values in label column: ", len(df))

#temporary: merge IDs of particiapnts
df = Merge_Transform.merge_participantIDs(df, users)#temporary: merge participant ids

# explore label distribution
# take only one record per "ESM_timestamp"
df_label_counting = df.drop_duplicates(subset=['ESM_timestamp'])
df_label_counts = df_label_counting.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
df_label_counts["total"] = df_label_counts.sum(axis=1)
df_label_counts.loc["total"] = df_label_counts.sum(axis=0)
# visually
sns.countplot(x = label_column_name, data = df_label_counting, order = df_label_counting[label_column_name].value_counts().index)
plt.show()

# explore sensor-label connection
def plot_activity(activity, df):
    data = df[df[label_column_name] == activity][sensor_columns_list][:50]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
    plt.show()
plot_activity("sitting", df)
plot_activity("standing", df)
plot_activity("lying", df)
plot_activity("walking", df)
plot_activity("running", df)
plot_activity("cycling", df)

# select labels included
df = df[df[label_column_name].isin(label_classes_included)]
df[label_column_name].value_counts()

# balance dataset based on the data exploration
# TODO: IMPROVE BALANCING
## only keep activities with at least 50.000 records
df[label_column_name].value_counts()
threshold_label_count = len(df) / (df[label_column_name].nunique()+1)
#df = df[df[label_column_name].isin(df[label_column_name].value_counts()[df[label_column_name].value_counts() > 10000].index)]
df[label_column_name].value_counts()

## only keep data from participants with at least 50.000 records
df['device_id'].value_counts()
threshold_participant_count = len(df) / (df['device_id'].nunique()+3)
df = df[df['device_id'].isin(df['device_id'].value_counts()[df['device_id'].value_counts() > 200].index)]
df['device_id'].value_counts()


# Make list of all ID's in idcolumn
IDlist = set(df["device_id"])
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

# create ROC Curve Plot Function
def plot_roc(y_true, y_pred, class_names):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0, len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_pred, pos_label=class_names[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    for i in range(0, len(class_names)):
        plt.plot(fpr[i], tpr[i], color='darkorange',
                lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(class_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

# create Precision-Recall Curve Plot Function
def plot_pr(y_true, y_pred, class_names):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(0, len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(y_true, y_pred, pos_label=class_names[i])
        average_precision[i] = average_precision_score(y_true, y_pred, pos_label=class_names[i])

    plt.figure()
    lw = 2
    for i in range(0, len(class_names)):
        plt.plot(recall[i], precision[i], color='darkorange',
                lw=lw, label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                ''.format(class_names[i], average_precision[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

#scale data with MinMax Scaler
def scale_data_MinMax(train_data, test_data, sensor_columns_list):
    f_transformer = MinMaxScaler()
    #t_transformer = MinMaxScaler()
    f_transformer = f_transformer.fit(train_data[sensor_columns_list].to_numpy())
    #t_transformer = t_transformer.fit(train_data[[label_column_name]])
    train_data.loc[:, sensor_columns_list] = f_transformer.transform(train_data[sensor_columns_list].to_numpy())
    #train_data[label_column_name] = t_transformer.transform(train_data[[label_column_name]])
    test_data.loc[:, sensor_columns_list] = f_transformer.transform(test_data[sensor_columns_list].to_numpy())
    #test_data[label_column_name] = t_transformer.transform(test_data[[label_column_name]])

    return f_transformer, train_data, test_data, #t_transformer

# scale data with Robust Scaler
def scale_data_Robust(df_train, df_test, sensor_columns_list):

    scaler = RobustScaler()

    scaler = scaler.fit(df_train[sensor_columns_list])

    df_train.loc[:, sensor_columns_list] = scaler.transform(
        df_train[sensor_columns_list].to_numpy()
    )

    df_test.loc[:, sensor_columns_list] = scaler.transform(
        df_test[sensor_columns_list].to_numpy()
    )

    return scaler, df_train, df_test



# create tuner_build function
def build_tuner(tuner_choice):
    if tuner_choice == "random":
        tuner = RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=10, #number of hyperparameter combinations the tuner will try
            executions_per_trial=1, #number of models that should be built and fit for each trial (due to its stochastic nature,
            # the same model can be trained and still different results)
            directory="/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/"+ label_column_name +"/models",
            project_name="lstm_"+ label_column_name +"_random",
        )
    elif tuner_choice == "bayesian":
        tuner = BayesianOptimization(
            build_model,
            objective='val_accuracy',
            max_trials=10,
            seed=42,
            directory="/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/"+ label_column_name +"/models",
            project_name="lstm_"+ label_column_name +"_bayesian"
        )
    elif tuner_choice == "hyperband":
        tuner = Hyperband(
            build_model,
            objective='val_accuracy',
            max_epochs=2,
            executions_per_trial=1,
            directory="/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/"+ label_column_name +"/models",
            project_name="lstm_"+ label_column_name +"_hyperband"
        )

    return tuner




    return tuner


#initialize
test_proband = list()
outer_results_acc = list()
outer_results_acc_balanced = list()
outer_results_f1 = list()
outer_results_precision = list()
outer_results_recall = list()
y_test_all = list()
yhat_all = list()

#for loop to iterate through LOSOCV "rounds"
t0 = time.time()
for i in IDlist:
    t0_inner = time.time()
    LOOCV_O = str(i)

    # select training and testing data for this LOSOCV round
    df["device_id"] = df["device_id"].apply(str)
    df_train = df[df["device_id"] != LOOCV_O].copy()
    df_test = df[df["device_id"] == LOOCV_O].copy()

    # scale data
    # based on: https://medium.com/analytics-vidhya/hypertuning-a-lstm-with-keras-tuner-to-forecast-solar-irradiance-7da7577e96eb
    # note: scaling should be done for training and testing data seperately
    #f_trans, df_train, df_test = scale_data_MinMax(df_train, df_test, sensor_columns_list, label_column_name)
    scaler, df_train, df_test = scale_data_Robust(df_train, df_test, sensor_columns_list)

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
    # use enc to reverse the encoding
    #print(enc.inverse_transform(y_train))
    # print the mapping of the ONeHotEncoder
    print(enc.categories_)

    # create class weights
    # get one-dimensional array of labels
    y_train_1D = y_train.argmax(axis=1)
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train_1D), y = y_train_1D)
    class_weights_dict = dict(enumerate(class_weights))

    # create tuner and model
    #TODO: problem with the tuner is that it doesnt work with the class weights; therefore useless at the moment
    if tuner_enabled == "yes":
        # this model & tuner implementation is based on the following tutorial: https://medium.com/analytics-vidhya/hypertuning-a-lstm-with-keras-tuner-to-forecast-solar-irradiance-7da7577e96eb
        # define model (which then will be optimised by the tuner)
        def build_model(hp):
            model = Sequential()
            model.add(LSTM(hp.Int('input_unit', min_value=32, max_value=512, step=32), return_sequences=True,
                           input_shape=(X_train.shape[1], X_train.shape[2])))
            for i in range(hp.Int('n_layers', 1, 4)):
                model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32), return_sequences=True))
            model.add(LSTM(hp.Int('layer_2_neurons', min_value=32, max_value=512, step=32)))
            model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))
            model.add(Dense(y_train.shape[1],
                            activation=hp.Choice('dense_activation', values=['LeakyReLU', 'tanh'], default='LeakyReLU')))
            #TODO insert into documentation: replaced the "ReLu" and "sigmoid" activation function by "LeakyReLU" and "tanh",
            # since the other two lead to all-zero predictions
            # replacement based on this forum answer: https://github.com/keras-team/keras/issues/3687

            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

            return model

        # tune hyperparameters
        tuner = build_tuner("random")

        print("Start tuning")
        tuner.search(
            x=X_train,
            y=y_train,
            epochs=20,
            batch_size=128,
            validation_data=(X_test, y_test),
        )
        print("Tuner search finished for proband " + LOOCV_O)
        print(tuner.results_summary())

        # get the best model from tuner
        model = tuner.get_best_models(num_models=1)[0]
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(bestHP.values)


    # define model
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

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    # train model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        class_weight=class_weights_dict
    )

    # plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included_text + "label_segment_around_ESMevent-" + str(label_segment) + "_test_proband-" + str(i) + "_model_history.png")
    #plt.show()
    plt.close()



    ### evaluate model
    yhat = model.predict(X_test)

    ### save model and hyperparameters (only if tuner is enabled)
    save_model(model, "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included_text + "label_segment_around_ESMevent-" + str(label_segment) + "_test_proband-" + str(i) + "_LSTM.h5")
    # save hyperparameters
    if tuner_enabled == "yes":
        with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name + "/sensors_included-" + sensors_included_text + "label_segment_around_ESMevent-" + str(label_segment) + "_test_proband-" + str(i) + "_LSTM_hyperparameters.txt", "w") as text_file:
            text_file.write("hyperparameters: %s" % bestHP.values)

    # transform yhat: set max value in yhat to 1 and all other values to 0
    yhat_rounded = np.where(yhat == np.amax(yhat, axis=1, keepdims=True), 1, 0)

    # transform them back into string labels
    y_test_strings = enc.inverse_transform(y_test)
    yhat_strings = enc.inverse_transform(yhat_rounded)
    # take one dimension out of the array
    y_test_strings = y_test_strings.ravel()
    yhat_strings = yhat_strings.ravel()
    print("y values have been transformed")

    # evaluate model
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(yhat_rounded, axis=1))
    acc_balanced = balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(yhat_rounded, axis=1))
    #print("Accuracy of LSTM: ", accuracy_score(y_test, y_pred.round()))
    f1 = f1_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(yhat_rounded, axis=1) , average='macro')
    precision = precision_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(yhat_rounded, axis=1), average='macro')
    recall = recall_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(yhat_rounded, axis=1), average='macro')
    #print accuracy for test proband
    print("Accuracy of LSTM for test proband " + str(i) + ": ", acc_balanced)

    # store the result
    test_proband.append(i)
    outer_results_acc.append(acc)
    outer_results_acc_balanced.append(acc_balanced)
    outer_results_f1.append(f1)
    outer_results_precision.append(precision)
    outer_results_recall.append(recall)

    y_test_all.append(y_test_strings)
    yhat_all.append(yhat_strings)

    # report progress
	print('>balanced acc=%.3f,f1=%.3f,precision=%.3f,recall=%.3f' % (acc_balanced, f1, precision, recall))
    print("The proband taken as test-data for this iteration was " + str(i))

    # convert categorical labels to numerical labels and save the mapping for later visualization
    # convert y_test to pandas series
    y_test_confusionmatrix = pd.Series(y_test_strings)
    y_test_confusionmatrix = y_test_confusionmatrix.astype('category')
    label_mapping = dict(enumerate(y_test_confusionmatrix.cat.categories))
    y_test_confusionmatrix = y_test_confusionmatrix.cat.codes
    y_test_confusionmatrix = y_test_confusionmatrix.to_numpy()
    # also convert yhat to numerical labels using same mapping
    yhat_confusionmatrix = pd.Series(yhat_strings)
    yhat_confusionmatrix = yhat_confusionmatrix.astype('category')
    label_mapping2 = dict(enumerate(yhat_confusionmatrix.cat.categories))
    # check if the mapping is the same
    if label_mapping == label_mapping2:
        print("The label mapping is the same")
    else:
        if len(label_mapping2) < len(label_mapping):
            #add "NaN" labels tolabel_mapping2
            for i in range(len(label_mapping2), len(label_mapping)):
                label_mapping2[i] = "NaN"
        elif len(label_mapping2) > len(label_mapping):
            #add "NaN" labels tolabel_mapping
            for i in range(len(label_mapping), len(label_mapping2)):
                label_mapping[i] = "NaN"

    yhat_confusionmatrix = yhat_confusionmatrix.cat.codes
    yhat_confusionmatrix = yhat_confusionmatrix.to_numpy()

    print("y values have been transformed for confusion matrix ")

    # Visualize Confusion Matrix with absolute values
    fig, ax = plt.subplots(figsize=(10, 5))
    mat = confusion_matrix(y_test_confusionmatrix, yhat_confusionmatrix)
    sns.heatmap(mat, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, fmt='d', cbar=False, linewidths=0.2)
    plt.title('Confusion Matrix Absolute Values with Test-Proband ' + str(i))
    plt.suptitle('Accuracy: {0:.3f}'.format(acc_balanced), fontsize=10)
    # add xticks and yticks from label_mapping (which is a dictionary)
    tick_marks = np.arange(len(label_mapping)) + 0.5
    plt.xticks(tick_marks, label_mapping2.values(), rotation=0)
    plt.yticks(tick_marks, label_mapping.values(), rotation=0)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    #plt.show()
    fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name +
                "/sensors_included-" + sensors_included_text + "label_segment_around_ESMevent-" + str(label_segment) +
                "_test_proband-" + str(i) + "_LSTM_confusionmatrix_absolute.png")
    #plt.close()
    print("Confusion Matrix with absolute values has been plotted")

    # visualize confusion matrix with percentages
    # Get and reshape confusion matrix data
    matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
    # Add labels to the plot
    plt.xticks(tick_marks, label_mapping.values(), rotation=0)
    plt.yticks(tick_marks, label_mapping.values(), rotation=0)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Relative Values with Test-Proband ' + str(i))
    plt.suptitle('Accuracy: {0:.3f}'.format(acc_balanced), fontsize=10)
    plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name +
                "/sensors_included-" + sensors_included_text + "label_segment_around_ESMevent-" + str(label_segment) +
                "_test_proband-" + str(i) + "_LSTM_confusionmatrix_relative.png")
    #plt.close()
    print("Confusion Matrix with relative values has been plotted")

    t1_inner = time.time()
    print("This inner iteration has taken so many minutes: " + str((t1_inner - t0_inner)/60))

#get the overall result for this LOSOCV from the combined y_test and yhat predictions of all inner iterations
# and append them to the outer_results lists
#combine all y_test and yhat values
y_test_all = np.concatenate(y_test_all)
yhat_all = np.concatenate(yhat_all)

test_proband.append("overall")
outer_results_acc.append(accuracy_score(y_test_all, yhat_all))
outer_results_acc_balanced.append(balanced_accuracy_score(y_test_all, yhat_all))
outer_results_f1.append(f1_score(y_test_all, yhat_all, average='macro'))
outer_results_precision.append(precision_score(y_test_all, yhat_all, average='macro'))
outer_results_recall.append(recall_score(y_true=y_test_all, y_pred=yhat_all, average='macro'))

#store all results in one dataframe:
df_stats = pd.DataFrame(data=[test_proband, outer_results_acc, outer_results_acc_balanced, outer_results_f1, outer_results_precision, outer_results_recall]).transpose()
df_stats.columns= ["test_proband","acc", "acc_balanced", "f1","precision", "recall"]
df_stats.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name +
                "/sensors_included-" + sensors_included_text + "label_segment_around_ESMevent-" + str(label_segment) +
                "_LSTM_results.csv", index=False)

# convert categorical labels to numerical labels and save the mapping for later visualization
# convert y_test to pandas series
y_test_confusionmatrix = pd.Series(y_test_all)
y_test_confusionmatrix = y_test_confusionmatrix.astype('category')
label_mapping = dict(enumerate(y_test_confusionmatrix.cat.categories))
y_test_confusionmatrix = y_test_confusionmatrix.cat.codes
y_test_confusionmatrix = y_test_confusionmatrix.to_numpy()
# also convert yhat to numerical labels using same mapping
yhat_confusionmatrix = pd.Series(yhat_all)
yhat_confusionmatrix = yhat_confusionmatrix.astype('category')
label_mapping2 = dict(enumerate(yhat_confusionmatrix.cat.categories))
yhat_confusionmatrix = yhat_confusionmatrix.cat.codes
yhat_confusionmatrix = yhat_confusionmatrix.to_numpy()

# Visualize Confusion Matrix with absolute values
fig, ax = plt.subplots(figsize=(10, 5))
mat = confusion_matrix(y_test_confusionmatrix, yhat_confusionmatrix)
sns.heatmap(mat, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, fmt='d', cbar=False, linewidths=0.2)
plt.title("Confusion Matrix Absolute Values for Combined LOSOCV Results" )
plt.suptitle('Accuracy: {0:.3f}'.format(acc_balanced), fontsize=10)
# add xticks and yticks from label_mapping (which is a dictionary)
tick_marks = np.arange(len(label_mapping)) + 0.5
plt.xticks(tick_marks, label_mapping.values(), rotation=0)
plt.yticks(tick_marks, label_mapping.values(), rotation=0)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
fig.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name +
            "/sensors_included-" + sensors_included_text + "label_segment_around_ESMevent-" + str(label_segment) +
            "_LSTM_confusionmatrix_absolute.png")

# visualize confusion matrix with percentages
# Get and reshape confusion matrix data
matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
# Add labels to the plot
plt.xticks(tick_marks, label_mapping.values(), rotation=0)
plt.yticks(tick_marks, label_mapping.values(), rotation=0)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("Confusion Matrix Relative Values for Combined LOSOCV Results")
plt.suptitle('Accuracy: {0:.3f}'.format(acc_balanced), fontsize=10)
plt.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_analysis/lstm/" + label_column_name +
            "/sensors_included-" + sensors_included_text + "label_segment_around_ESMevent-" + str(label_segment) +
            "_LSTM_confusionmatrix_relative.png")
plt.show()




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