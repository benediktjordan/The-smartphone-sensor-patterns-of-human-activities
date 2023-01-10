
class data_exploration_labels:

    def load_esm(esm_path):
        # load ESM data
        df_esm = pd.read_csv(esm_path)
        return df_esm

    # visualize summary of ESM data for every question
    def visualize_esm_activities_all(dir_results, df_esm):
        # create summary of ESM data
        esm_summary = {
            "location": df_esm["location"].value_counts(),
            "bodyposition": df_esm["bodyposition"].value_counts(),
            "activity": df_esm["activity"].value_counts(),
            "smartphonelocation": df_esm["smartphonelocation"].value_counts(),
            "aligned": df_esm["aligned"].value_counts(),
            "Human Motion General": df_esm["label_human motion - general"].value_counts(),
            "Human Motion Specific": df_esm["label_human motion - specific"].value_counts(),
            "Public Transport": df_esm["label_public transport"].value_counts(),
            "Before & After Sleep ": df_esm["label_before and after sleep"].value_counts(),
            "Bathroom": df_esm["label_on the toilet"].value_counts(),
            "Location": df_esm["location"].value_counts(),
            "Smartphone Location": df_esm["smartphonelocation"].value_counts()
        }

        # create bar plots for every question
        for activity in esm_summary:
            plt.figure(figsize=(15, 10))
            plt.title("Events for " + activity)
            sns.barplot(x=esm_summary[activity][0:10].index, y=esm_summary[activity][0:10].values,
                        palette="Blues_d")
            plt.xticks(rotation=15)
            # include actual number of events in bar plot
            for i, v in enumerate(esm_summary[activity][0:10].values):
                plt.text(i - 0.1, v + 0.1, str(v), color='black')

            plt.savefig(dir_results + "/" + activity + "_ESM activity count.png")
            #plt.show()

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

    # visualize one activity as barplot
    def visualize_esm_activity(df_esm, column_name, fig_title):

        fig = plt.figure(figsize=(15, 10))
        plt.title(fig_title)
        sns.barplot(x=df_esm[column_name].value_counts()[0:10].index, y=df_esm[column_name].value_counts()[0:10].values,
                    palette="Blues_d")
        plt.xticks(rotation=15)
        # include actual number of events in bar plot
        for i, v in enumerate(df_esm[column_name].value_counts()[0:10].values):
            plt.text(i - 0.1, v + 0.1, str(v), color='black')

        plt.show()
        return fig

    # create table which shows users x activities (so number of labels per user and label class)
    def create_table_user_activity(df_esm, label_column_name):

        #create label counts per user
        df_label_counts = df_esm.groupby("device_id")[label_column_name].value_counts().unstack().fillna(0)
        df_label_counts = df_label_counts.astype(int)

        # add sum for rows and columns
        df_label_counts["total"] = df_label_counts.sum(axis=1)


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





