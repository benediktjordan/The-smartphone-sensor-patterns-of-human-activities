# This class contains several methods for visualizing the ESM data
class data_exploration_labels:

    def load_esm(esm_path):
        # load ESM data
        df_esm = pd.read_csv(esm_path)
        return df_esm

    # for all contexts: barplot of classes x number of events
    def visualize_esm_activities_all(dir_results, df_esm, threshold):

        df_analytics = pd.DataFrame(columns=["activity", "count (not NaN)", "count above threshold", "percentage above threshold"])

        # create summary of ESM data
        esm_summary = {
            "Location Question": df_esm["location"].value_counts(),
            "Body-Position Question": df_esm["bodyposition"].value_counts(),
            "Activity Question": df_esm["activity"].value_counts(),
            "Smartphone Location Question": df_esm["smartphonelocation"].value_counts(),
            "Alignment Question": df_esm["aligned"].value_counts(),
            "Human Motion": df_esm["label_human motion"].value_counts(),
            "Public Transport": df_esm["label_public transport"].value_counts(),
            "Before & After Sleep ": df_esm["label_before and after sleep"].value_counts(),
            "Lavatory Use": df_esm["label_on the toilet"].value_counts(),
            "Location": df_esm["location"].value_counts(),
            "Smartphone Location": df_esm["smartphonelocation"].value_counts()
        }


        # create bar plots for every question
        for activity in esm_summary:

            # fill df_analytics with data: add one row with activity, count (not NaN), count above threshold, percentage above threshold
            df_analytics.loc[len(df_analytics)] = [activity, esm_summary[activity].sum(), esm_summary[activity][esm_summary[activity] > threshold].sum(), esm_summary[activity][esm_summary[activity] > threshold].sum() / esm_summary[activity].sum()]

            # delete all rows with value < 10
            esm_summary[activity] = esm_summary[activity][esm_summary[activity] > threshold]

            plt.figure(figsize=(15, 10))
            plt.suptitle("Answer Distribution for " + activity, fontsize=24)
            sns.barplot(x=esm_summary[activity][0:15].index, y=esm_summary[activity][0:15].values,
                        palette="Blues_d")

            plt.xticks(rotation=15)
            # if there are more than 10 classes, rotate x-axis labels more
            if len(esm_summary[activity][0:15].index) > 10:
                plt.xticks(rotation=45)
            # include actual number of events in bar plot
            for i, v in enumerate(esm_summary[activity][0:15].values):
                plt.text(i - 0.1, v + 0.1, str(v), color='black')
            # add y-label
            plt.ylabel("number of events")

            plt.tight_layout()
            plt.savefig(dir_results + "/" + activity + "_ESM activity count.png", dpi=400, bbox_inches='tight')
            #plt.show()

        return df_analytics

    # for one context: barplot of classes x number of events
    def visualize_esm_activity(df_esm, column_name, fig_title):
        fig = plt.figure(figsize=(15, 10))
        plt.title(fig_title, fontsize=16)
        sns.barplot(x=df_esm[column_name].value_counts()[0:20].index,
                    y=df_esm[column_name].value_counts()[0:20].values,
                    palette="Blues_d")
        if len(df_esm[column_name].value_counts()[0:20].index) > 10:
            plt.xticks(rotation=20)
        else:
            plt.xticks(rotation=15)
        # include actual number of events in bar plot
        for i, v in enumerate(df_esm[column_name].value_counts()[0:20].values):
            plt.text(i - 0.1, v + 0.1, str(v), color='black')

        plt.show()
        return fig

    # for one context: barplot of classes x number of minutes (ESM_timestamp * esm_segment)
    ## Note: this function is relevant only for laboratory data. It visualizes an approximation
    ## for the minutes of data recorded per activity class by multiplying the number of ESM events
    ## with the length of the ESM segments (in minutes)
    def visualize_esm_activity_minutes(df_esm, column_name, esm_segment_length, fig_title):
        esm_segment_length_in_minutes = esm_segment_length / 60
        fig = plt.figure(figsize=(15, 10))
        plt.title(fig_title, fontsize=16)
        sns.barplot(x=df_esm[column_name].value_counts()[0:20].index,
                    y=(df_esm[column_name].value_counts()[0:20].values * esm_segment_length_in_minutes),
                    palette="Blues_d")
        if len(df_esm[column_name].value_counts()[0:20].index) > 10:
            plt.xticks(rotation=20)
        else:
            plt.xticks(rotation=15)
        # include actual number of events in bar plot
        for i, v in enumerate(df_esm[column_name].value_counts()[0:20].values * esm_segment_length_in_minutes):
            plt.text(i - 0.1, v + 0.1, str(v), color='black')

        #label x-axis and y-axis
        plt.xlabel("classes")
        plt.ylabel("total minutes")

        plt.show()
        return fig

    # visualize sum of ES data
    def visualize_esm_notNaN(df_esm):
        # rename columns
        df_esm = df_esm.rename(columns={"label_human motion - general": "Human Motion General",
                                        "label_human motion - specific": "Human Motion Specific",
                                        "label_public transport": "Public Transport",
                                        "label_before and after sleep": "Before & After Sleep",
                                        "label_on the toilet": "Bathroom",
                                        "location": "Location",
                                        "smartphonelocation": "Smartphone Location",
                                        "aligned": "Aligned",
                                        "activity": "Activity",
                                        "bodyposition": "Body Position",})


        # create bar plot which shows sums of all columns of relevant columns
        relevant_columns = ["Location", "Smartphone Location",
                            "Aligned", "Activity", "Body Position",
                            "Human Motion General", "Human Motion Specific", "Public Transport",
                            "Before & After Sleep", "Bathroom", ]

        # create dataframe which contains the number of rows which are not NaN for every column
        df_esm_notNaN = pd.DataFrame(columns=["column", "notNaN"])
        for column in relevant_columns:
            df_esm_notNaN = df_esm_notNaN.append({"column": column, "notNaN": df_esm[column].notna().sum()}, ignore_index=True)

        plt.figure(figsize=(15, 10))
        plt.title("Number of not NaN values")
        sns.barplot(x=df_esm_notNaN["column"], y=df_esm_notNaN["notNaN"],
                    palette="Blues_d")
        plt.xticks(rotation=15)
        # include actual number of events in bar plot
        for i, v in enumerate(df_esm_notNaN["notNaN"]):
            plt.text(i - 0.1, v + 0.1, str(v), color='black')
        plt.savefig(dir_results + "/" + "ESM not NaN count.png")

    # create table which shows users x classes x number of events
    ## Note: relevant for naturalistic data only
    def create_table_user_classes_eventcount(df_esm, label_column_name):

        #create label counts per user
        df_label_counts = df_esm.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts = df_label_counts.astype(int)

        # add sum for rows and columns
        # add sum for rows but without including values from the column "User ID"
        df_label_counts["total"] = df_label_counts.iloc[:, 0:].sum(axis=1)
        #df_label_counts["total"] = df_label_counts.sum(axis=1)

        # create User_ID column from index
        df_label_counts["User ID"] = df_label_counts.index

        # add column with the city of user as the second column by matching the User_ID with the city of the user in df_labels_publictransport with lambda function
        df_label_counts["City"] = df_label_counts.apply(lambda x: df_esm[df_esm["device_id"] == x["User ID"]]["city"].values[0], axis=1)

        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)

        # make sure that User ID and City are the first two columns
        cols = df_label_counts.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df_label_counts = df_label_counts[cols]

        return df_label_counts

    # create table which shows users x classes x number of minutes recorded for this class
    ## Note: relevant for laboratory data only.
    def create_table_user_classes_minutes(df_esm, label_column_name, esm_segment_length):

        #create label counts per user
        df_label_counts = df_esm.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        # multiply every cell with the length of the ESM segments (in minutes)
        df_label_counts = df_label_counts * (esm_segment_length / 60)
        # round to 1 decimal
        df_label_counts = df_label_counts.round(1)

        # add sum for rows and columns
        # add sum for rows but without including values from the column "User ID"
        df_label_counts["total"] = df_label_counts.iloc[:, 0:].sum(axis=1)
        #df_label_counts["total"] = df_label_counts.sum(axis=1)

        # create User_ID column from index
        df_label_counts["User ID"] = df_label_counts.index

        # add column with the city of user as the second column by matching the User_ID with the city of the user in df_labels_publictransport with lambda function
        df_label_counts["City"] = df_label_counts.apply(lambda x: df_esm[df_esm["device_id"] == x["User ID"]]["city"].values[0], axis=1)

        df_label_counts.loc["total"] = df_label_counts.sum(axis=0)

        # make sure that User ID and City are the first two columns
        cols = df_label_counts.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df_label_counts = df_label_counts[cols]

        return df_label_counts

    # create table which shows users x activities FOR WHICH SENSOR DATA IS AVAILABLE
    # dependencies of this function:
    ## functions
    ## labeling_sensor()
    ## data_exploration_labels.create_table_user_activity()
    ## Merge_Transform.merge_participantIDs()
    ## data_exploration_labels.create_table_user_activity()
    ## databases
    ### users
    ### sensors_and_frequencies
    def create_table_user_activity_including_sensordata(df_esm_including_number_sensordata, dict_label,
                                                        label_column_name, sensors_included, segment_around_events):
        df_esm = df_esm_including_number_sensordata.copy()

        # add label column to df_esm
        df_esm = labeling_sensor_df(df_esm, dict_label, label_column_name, ESM_identifier_column="ESM_timestamp")

        # drop any NaN values in label column
        print("Number of rows before dropping NaN values in label column: " + str(df_esm.shape[0]))
        df_esm = df_esm.dropna(subset=[label_column_name])
        if label_column_name == "label_public transport":
            # drop all "exclude" labels: these are instances of "walking" in which people are "at home"
            df_esm = df_esm[df_esm[label_column_name] != "exclude"]
        print("Number of ESM_timestamps after deleting NaN label values: " + str(df_esm.shape[0]))

        # delete all columns which are not "timestamp", "ESM_timestam", label_column_name, or in sensors_included
        df_esm = df_esm[[label_column_name, "ESM_timestamp", "device_id"] + sensors_included]

        for sensor in sensors_included:
            print("started with sensor: " + sensor)
            # set all records for which the number of sensor data points is less than required by the minimum percentage to 0
            print(
                "Number of ESM_timestamps with sensor data BEFORE DELETING DUE TO min_percentage for sensor: " + sensor + ": " + str(
                    df_esm[df_esm[sensor] > 0].shape[0]))
            sensor_frequency = sensors_and_frequencies[sensors_and_frequencies["Sensor"] == sensor]["Frequency (in Hz)"]
            min_number_sensorrecords = float(
                (segment_around_events * sensor_frequency) * (min_sensordata_percentage / 100))

            # set all values below the minimum number of sensor records to 0 with np.where
            df_esm[sensor] = np.where(df_esm[sensor] < min_number_sensorrecords, 0, df_esm[sensor])
            print(
                "Number of ESM_timestamps with sensor data AFTER DELETING DUE TO min_percentage for sensor: " + sensor + ": " + str(
                    df_esm[df_esm[sensor] > 0].shape[0]))

        # drop all rows in which there is a 0 in any of the sensors_included
        for sensor in sensors_included:
            df_esm = df_esm[df_esm[sensor] > 0]

        # merge the participant IDs
        df_esm = Merge_Transform.merge_participantIDs(df_esm, users, include_cities=True)

        # make a table which shows participants x count of events per label class; use groupby
        df_esm_label_counts = data_exploration_labels.create_table_user_classes_eventcount(df_esm, label_column_name)

        return df_esm, df_esm_label_counts






