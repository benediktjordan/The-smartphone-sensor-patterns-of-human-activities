class Static_Periods:

    # compute long static periods for a participant
    def compute_long_periods_of_static(df_participant_all, threshold_expected, threshold_statis, window_size_minutes, window_step_minutes, sensor_frequency):
        # compute long periods of static
        df_results = pd.DataFrame(columns = ["participant", "start_time", "end_time", "duration","actual_values", "expected_values", "percentage_expected", "static_values", "percentage_static"])
        df_analytics = pd.DataFrame(columns = ["participant", "start_time", "end_time", "duration","actual_values", "expected_values", "percentage_expected", "static_values", "percentage_static"])

        # truncate the sensor_column to 3 decimals (dont round but cut): to decrease sensitivity of the sensor
        df_participant_all["double_values_0"] = df_participant_all["double_values_0"].apply(lambda x: float("{:.3f}".format(x)))

        # expected values for every window
        expected_values = window_size_minutes * 60 * sensor_frequency
        # compute start and end time of data
        start_time = df_participant_all["timestamp"].min()
        end_time = df_participant_all["timestamp"].max()

        # calculate total number of windows
        total_number_of_windows = int((end_time - start_time).total_seconds() / 60 / window_step_minutes) - int(window_size_minutes / window_step_minutes)

        # iterate through windows in windows of window_size_minutes and in steps of window_step_minutes: increase start time for every window in steps of window_step_minutes
        start_time_window = start_time
        end_time_window = start_time + datetime.timedelta(minutes=window_size_minutes)
        counter = 0
        while end_time_window < end_time:
            counter = counter +1
            # print counter all 100 windows
            if counter % 100 == 0:
                print("counter: " + str(counter) + " of " + str(total_number_of_windows))

            # get value counts of double_values_0
            df_participant_window = df_participant_all[(df_participant_all["timestamp"] >= start_time_window) & (df_participant_all["timestamp"] < end_time_window)]

            # jump over window if it is empty
            if len(df_participant_window) == 0:
                # increase start and end time of window
                start_time_window = start_time_window + datetime.timedelta(minutes=window_step_minutes)
                end_time_window = end_time_window + datetime.timedelta(minutes=window_step_minutes)
                continue

            # check if window contains at least 50% of expected values
            if len(df_participant_window) >= expected_values * threshold_expected:
                # get value counts of double_values_0
                value_counts = df_participant_window["double_values_0"].value_counts()
                most_occuring_sum = value_counts.iloc[0:10].sum()
                # check if most frequent value is at least 95% of all values
                if most_occuring_sum >= len(df_participant_window) * threshold_statis:
                    # add to df_results
                    df_results.loc[len(df_results)] = [participant, start_time_window, end_time_window, end_time_window - start_time_window, len(df_participant_window), expected_values, len(df_participant_window) / expected_values, most_occuring_sum, most_occuring_sum / len(df_participant_window)]
                    print("long period of static for start time window: " + str(start_time_window) + " and end time window: " + str(end_time_window) )


                # get analytics
                df_analytics.loc[len(df_analytics)] = [participant, start_time_window, end_time_window,
                                                     end_time_window - start_time_window, len(df_participant_window),
                                                     expected_values, len(df_participant_window) / expected_values,
                                                     most_occuring_sum, most_occuring_sum / len(df_participant_window)]

            else:
                # get analytics
                value_counts = df_participant_window["double_values_0"].value_counts()
                most_occuring_sum = value_counts.iloc[0:10].sum()
                df_analytics.loc[len(df_analytics)] = [participant, start_time_window, end_time_window,
                                                     end_time_window - start_time_window, len(df_participant_window),
                                                     expected_values, len(df_participant_window) / expected_values,
                                                     most_occuring_sum, most_occuring_sum / len(df_participant_window)]

            # increase start and end time of window
            start_time_window = start_time_window + datetime.timedelta(minutes=window_step_minutes)
            end_time_window = end_time_window + datetime.timedelta(minutes=window_step_minutes)

        # merge overlapping windows
        index_new = 0
        df_length = len(df_results)
        for index, row in df_results.iterrows():
            print("start with index " + str(index))
            # check if index is below index_new
            if index < index_new:
                print("jump over index " + str(index))
                continue
            # check if start time of next row is before end time of current row
            if index < df_length - 1:
                index_new = index
                while df_results.loc[index_new + 1]["start_time"] < row["end_time"]:
                    print("merge with index " + str(index_new))
                    # set the end time of the current row to the end time of the next row
                    df_results.at[index, "end_time"] = df_results.loc[index_new + 1]["end_time"]
                    # add expected values, actual values and static values of next row to current row
                    df_results.at[index, "expected_values"] = df_results.loc[index_new + 1]["expected_values"] + df_results.loc[index]["expected_values"]
                    df_results.at[index, "actual_values"] = df_results.loc[index_new + 1]["actual_values"] + df_results.loc[index]["actual_values"]
                    df_results.at[index, "static_values"] = df_results.loc[index_new + 1]["static_values"] + df_results.loc[index]["static_values"]
                    # update row time
                    row["end_time"] = df_results.loc[index_new + 1]["end_time"]

                    # increase index_new
                    index_new = index_new + 1

                    # drop the row
                    df_results = df_results.drop(index_new)

                    # check if index_new is the last row
                    if index_new == df_length - 1:
                        break

        # update duration, percentage expected and percentage static
        df_results["duration"] = df_results["end_time"] - df_results["start_time"]
        df_results["percentage_expected"] = df_results["actual_values"] / df_results["expected_values"]
        df_results["percentage_static"] = df_results["static_values"] / df_results["actual_values"]

        return df_results, df_analytics

    # compute distance of static periods to events
    def static_periods_distance_to_events(df_events, df_static_periods, threshold_time_distance):
        df_events["static_period_start_distance"] = np.nan
        df_events["static_period_end_distance"] = np.nan

        # convert start_time and end_time from string to datetime
        df_static_periods["start_time"] = pd.to_datetime(df_static_periods["start_time"])
        df_static_periods["end_time"] = pd.to_datetime(df_static_periods["end_time"])

        # iterate through rows in df_events
        for index, row in df_events.iterrows():
            # compute "static_period_start_distance"
            ## get static_period with closest start_time to event
            df_closest_static_period_start = df_static_periods[df_static_periods["participant"] == row["device_id"]].iloc[(df_static_periods[df_static_periods["participant"] == row["device_id"]]["start_time"] - row["timestamp"]).abs().argsort()[:1]]

            # check if distance is smaller than threshold
            if len(df_closest_static_period_start) > 0:
                time_distance = ( df_closest_static_period_start["start_time"].iloc[0] - row["timestamp"]).total_seconds() / 60
                if abs(time_distance) < threshold_time_distance:
                    # add time distance in minutes
                    df_events.loc[index, "static_period_start_distance"] = time_distance


            # compute "static_period_end_distance"
            ## get static_period with closest end_time to event
            df_closest_static_period_end = df_static_periods[df_static_periods["participant"] == row["device_id"]].iloc[(df_static_periods[df_static_periods["participant"] == row["device_id"]]["end_time"] - row["timestamp"]).abs().argsort()[:1]]
            if len(df_closest_static_period_end) > 0:
                time_distance = (row["timestamp"] - df_closest_static_period_end["end_time"].iloc[0]).total_seconds() / 60
                if abs(time_distance) < threshold_time_distance:
                    # add time distance in minutes
                    df_events.loc[index, "static_period_end_distance"] = time_distance

        return df_events

