#region import libraries
#pip install mixpanel-utils
from mixpanel_utils import MixpanelUtils
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import datetime
import seaborn as sns
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler


#pip install tensorflow_decision_forests
#pip install wurlitzer
import math
#endregion

#region import data
event = ["close", "First App Open", "set_screentime_limit", "snooze", "App Session", "App Updated", "interval_did_end",
         "interval_did_start", "interval_will_end_warning", "reached_threshold", "schedule_error", "set_screentime_budget",
         "set_shield_goal", "set_shield_goals", "set_shielded_apps", "will_reach_threshold"]
folder_to_save = "/Users/benediktjordan/Documents/MTS/QualityAssurance/Analytics/MixpanelData/Events/Downloaded_Python/"
from_date = "2021-09-21" #note: "time" columns shows CET time
to_date = "2022-09-22"
filename = "allevents_until_"+ to_date + ".csv"
export_format = "csv" ##you can use json as well
api_secret_export = "b25c98e37af48d2dd2982f59393150de"
project_token_export = "56b26a3aa0e0c2a184a2b53ccd6d1bc4"

m = MixpanelUtils(api_secret_export,token= project_token_export,eu= "true")
m.export_events(folder_to_save + filename,{'from_date':from_date,'to_date':to_date,'event':event}, format=export_format)
print("\n================Exported correctly================")
#endregion

# load data
to_date = "2022-09-22"
folder_to_save = "/Users/benediktjordan/Documents/MTS/QualityAssurance/Analytics/MixpanelData/Events/Downloaded_Python/"
filename = "allevents_until_"+ to_date + ".csv"
df = pd.read_csv(folder_to_save + filename)

# clean data
## delete all rows with "<null>" value in "threshold"
df = df[df["threshold"] != "<null>"]


# compute labels: successful shields
time_period_seconds = 60 #secons

for user in df["$user_id"].unique():
    # iterate through "reached_threshold" events for user
    for event in df.loc[(df["$user_id"] == user) & (df["event"] == "reached_threshold")].index:
        # check if user has "snooze" event in the next 10 minutes
        if df.loc[(df["$user_id"] == user) & (df["event"] == "snooze") & (df["time"] > df.loc[event, "time"]) & (df["time"] < df.loc[event, "time"] + time_period_seconds*1000)].shape[0] > 0:
            # set label to 0 = unsuccessfull shield
            df.loc[event, "label"] = 0
        else:
            # set label to 0
            df.loc[event, "label"] = 1


#TODO validate if this is correct

# compute features for each shield
## general features
#iterate through "reached_threshold" events for user
for user in df["$user_id"].unique():
    for event in df.loc[(df["$user_id"] == user) & (df["event"] == "reached_threshold")].index:
        # compute time of shield in 24:00 format
        df.loc[event, "shield_time_24"] = datetime.datetime.fromtimestamp(df.loc[event, "time"]).strftime("%H:%M")
        #TODO  transform 24:00 time format into decimal time (e.g. 12:30:00 = 12.5)

        #compute weekday of shield
        df.loc[event, "shield_weekday"] = datetime.datetime.fromtimestamp(df.loc[event, "time"]).weekday()

# create dataset containing only reached_threshold events and features; include only columns event, threshold, $user_id,
# shield_time_24, shield_weekday, label
df_shield= df.loc[df["event"] == "reached_threshold", ["event", "threshold", "$user_id", "shield_time_24", "shield_weekday", "label"]]
df_shield = df_shield.dropna()



## session properties
#TODO calculate "ongoing index" for each shield (if shield is in ongoing session and, if yes, the index of the shield in the session)
#TODO screentime budget for shield
## historic data
#TODO calculate minutes of screentime on problematic apps today/yesterday/last week
#TODO calculate % of budget reached today/yesterday/last week
# TODO calculate details of last/second last/third last/... shield

# compute time-series format for LSTM
#TODO research how that format could look like




# train decision forest - general model (for all users)

# Separating the features (x) and the Labels (y) & delete unnecessary columns
X = df_shield.drop(["label", "event", "$user_id"], axis=1)
y = df_shield["label"]

# convert all columns to float
X = X.astype(float)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Resolve imbalances in dataset: downsample overrepresented class by randomly droping events
under_sampler = RandomUnderSampler(random_state=42)
X_train, y_train = under_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y_test)}")

# create and train model
rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
rf_model.fit(X_train, y_train)

# test
y_pred_test = rf_model.predict(X_test) #make predictions on test dataset
y_pred_test

rf_model.predict_proba(X_test) #show probabilities for test dataset

rf_model.classes_ # Shows the order of the classes

# Feature Importances
rf_model.feature_importances_ #show feature importances
importances = rf_model.feature_importances_
columns = X.columns
i = 0
while i < len(columns):
    print(f" The importance of feature '{columns[i]}' is {round (importances[i] * 100, 2)}%.")
    i += 1

# Confusion matrix with absolute values
conf_mat = confusion_matrix(y_test, y_pred_test)
print(conf_mat)
sns.heatmap(conf_mat)
plt.show()

## other visualization of Confusion Matrix with percentages
# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,16))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Unsuccessful Shields', 'Successful Shields']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

# Compute Accuracy
accuracy_score(y_test, y_pred_test)

#Classification report
# View the classification report for test data and predictions
print(classification_report(y_test, y_pred_test))

# train LSTM
