#region transform unix timestamp into datetime

def convert_unix_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], unit='ms')
    return df

#endregion

# region data preparation: label sensorfiles
#function to label sensorfiles -> put into "general functions" file
## Note: this function assumes that the time-information is in datetime format
def labeling_sensor_df(df_sensor, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp"):
    # iterate through df_sensor and find labels for each row
    df_sensor[label_column_name] = np.nan
    for index, row in df_sensor.iterrows():
        # find label for each row
        label = dict_label[pd.Timestamp(row[ESM_identifier_column])][label_column_name]
        # add label to df_sensor
        df_sensor.at[index, label_column_name] = label

    return df_sensor
#endregion