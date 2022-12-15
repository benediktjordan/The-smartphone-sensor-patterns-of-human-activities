#region import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import datetime
#from matplotlib import rc
#rc('text', usetex=True)
#endregion

class data_exploration_sensors:

    # load sensor data
    def load_sensordata(sensordata_path):
        # check if sensor_path is .csv or .pkl
        if sensordata_path.endswith(".csv"):
            df = pd.read_csv(sensordata_path)
        elif sensordata_path.endswith(".pkl"):
            df = pd.read_pickle(sensordata_path)
        else:
            print("Error: sensordata_path is not .csv or .pkl")
        return df

    # sensor-based: create function which visualizes the data of one sensor in lineplot for specific ES-event in timeperiod
    def vis_sensordata_around_event_onesensor(df_sensor, event_time, time_period, sensor_name, label_column_name, axs_limitations):
        # find names of sensor columns and rename columns
        for col in df_sensor.columns:
            if "double_values_0" in col:
                df_sensor = df_sensor.rename(columns={col: "x-axis"})
            elif "double_values_1" in col:
                df_sensor = df_sensor.rename(columns={col: "y-axis"})
            elif "double_values_2" in col:
                df_sensor = df_sensor.rename(columns={col: "z-axis"})

        # make sure that timestamp columns are datetime format
        df_sensor["timestamp"] = pd.to_datetime(df_sensor["timestamp"])
        df_sensor["ESM_timestamp"] = pd.to_datetime(df_sensor["ESM_timestamp"])
        event_time = pd.to_datetime(event_time)

        # create df with sensor data around event_time of of the event and participant
        df_event = df_sensor[df_sensor["ESM_timestamp"] == event_time].copy()
        df_event = df_event[(df_event['timestamp'] >= event_time - pd.Timedelta(seconds=(time_period / 2))) & (
                    df_event['timestamp'] <= event_time + pd.Timedelta(seconds=(time_period / 2)))]

        # check if df_event is empty: if yes, break function and return None
        if df_event.empty:
            print("df_event is empty with event_time: " + str(event_time))
            return None

        # transform unix time into seconds relative to event_time
        df_event["time relative to event (s)"] = df_event["timestamp"] - event_time
        df_event["time relative to event (s)"] = df_event["time relative to event (s)"].dt.total_seconds()

        # create axis limitations for every sensor
        if axs_limitations == "general":
            # get 1% and 99% quantiles of x, y and z axis
            x_01 = df_sensor["x-axis"].quantile(0.01)
            x_99 = df_sensor["x-axis"].quantile(0.99)
            y_01 = df_sensor["y-axis"].quantile(0.01)
            y_99 = df_sensor["y-axis"].quantile(0.99)
            z_01 = df_sensor["z-axis"].quantile(0.01)
            z_99 = df_sensor["z-axis"].quantile(0.99)

        elif axs_limitations == "event":
            # get 1% and 99% quantiles of x, y and z axis
            x_01 = df_event["x-axis"].quantile(0.01)
            x_99 = df_event["x-axis"].quantile(0.99)
            y_01 = df_event["y-axis"].quantile(0.01)
            y_99 = df_event["y-axis"].quantile(0.99)
            z_01 = df_event["z-axis"].quantile(0.01)
            z_99 = df_event["z-axis"].quantile(0.99)
        else:
            print("Error: axs_limitations is not 'general' or 'event'")
            return None

        # compute participant ID: take the first one of df_event
        #print("ESM timestamp: ", event_time)
        participant_id = df_event["device_id"].iloc[0]
        # compute esm_answer
        activity_name = df_event[label_column_name].iloc[0]

        # visualize for every one of the three axis the data in a lineplot with seaborn
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        #create title for entire plot
        sns.lineplot(x="time relative to event (s)", y="x-axis", data=df_event, ax=axs[0])
        sns.lineplot(x="time relative to event (s)", y="y-axis", data=df_event, ax=axs[1])
        sns.lineplot(x="time relative to event (s)", y="z-axis", data=df_event, ax=axs[2])
        # set x and y limits as min and max values of the axis
        axs[0].set_ylim(x_01, x_99)
        axs[1].set_ylim(y_01, y_99)
        axs[2].set_ylim(z_01, z_99)

        plt.suptitle(sensor_name + " Around the Activity " + activity_name +
                     "\n(Participant " + str(participant_id) +" at " + str(event_time) + " with axis based on " + axs_limitations + " data)", fontsize = 20)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        #plt.show()

        return fig, activity_name

    # apply vis_sensordata_around_event for all events which contain specific activity class
    def vis_sensordata_around_event_onesensor_foractivity(activity_name, df_sensor, time_period, sensor_name, label_column_name, path_to_save):
        # create df with only events which contain the activity_name
        df_sensor = df_sensor[df_sensor[label_column_name] == activity_name]

        # create list with all event times
        event_times = df_sensor["ESM_timestamp"].unique()

        # apply vis_sensordata_around_event for every event time in event_times
        for event_time in event_times:
            fig = data_exploration_sensors.vis_sensordata_around_event(df_sensor, event_time, time_period, sensor_name, label_column_name)

            # check if fig is None: if yes, continue with next event_time
            if fig is None:
                continue
            fig.savefig(path_to_save + activity_name + "/" + activity_name + "_EventTime-"  + str(event_time) + "_Segment-" + str(time_period) + "_Sensor-"+ sensor_name + ".png")
            plt.close(fig)

    # event-based: visualize several sensors (2-4; must be high-frequency) around every event
    #TODO include in this function the possibility to visualize also for data from second iteration:
    #-> make labeling optional DONE
    #-> create "ESM_timestamp" column in the new dataset below
    #-> add new User_ID to the users list
    # -> rename "label" column in "label_human motion - general"
    # -> think what other adaptations need to be done
    # -> visualize also GPS feature speed
    # -> compare (in another function) activity recognition quality of Google automatic activity recognition
    def vis_sensordata_around_event_severalsensors(list_sensors, event_time, time_period, label_column_name, path_sensordata, axs_limitations, label_data = False):
        """
        :param list_sensors: list of sensors to visualize
        :param event_time:
        :param time_period:
        :param label_column_name:
        :param path_sensordata:
        :param label_data:
        :param axs_limitations: decide if axis limitations should be 0.01 and 0.99 quantiles of
         -> whole sensordata (general)
         -> sensordata around event_time (event)
        :return:
        """
        #initialize
        dict_df_event = {}
        dict_axs = {}
        # iterate through all sensors and create df_sensor_event for every sensor
        for sensor in list_sensors:
            time0 = time.time()
            #load sensor data
            # replace "INSERT" in path_sensordata with sensor name
            path_sensordata_adapted = path_sensordata.replace("INSERT", sensor[1])
            df_sensor = data_exploration_sensors.load_sensordata(path_sensordata_adapted)

            #preprocessing
            df_sensor = Merge_Transform.merge_participantIDs(df_sensor, users)  # merge participant IDs
            if label_data == True: #only if data from first iteration, labelling of sensordata is needed
                df_sensor = labeling_sensor_df(df_sensor, dict_label, label_column_name)
            df_sensor = df_sensor.dropna(subset=[label_column_name])  # delete all rows which contain NaN values in the label column

            # find names of sensor columns and rename columns
            for col in df_sensor.columns:
                if "double_values_0" in col:
                    df_sensor = df_sensor.rename(columns={col: "x-axis"})
                elif "double_values_1" in col:
                    df_sensor = df_sensor.rename(columns={col: "y-axis"})
                elif "double_values_2" in col:
                    df_sensor = df_sensor.rename(columns={col: "z-axis"})

            # make sure that timestamp columns are datetime format
            df_sensor["timestamp"] = pd.to_datetime(df_sensor["timestamp"])
            df_sensor["ESM_timestamp"] = pd.to_datetime(df_sensor["ESM_timestamp"])
            event_time = pd.to_datetime(event_time)

            # create df with sensor data around event_time of the event and participant
            df_event = df_sensor[df_sensor["ESM_timestamp"] == event_time].copy()
            df_event = df_event[(df_event['timestamp'] >= event_time - pd.Timedelta(seconds=(time_period / 2))) & (
                    df_event['timestamp'] <= event_time + pd.Timedelta(seconds=(time_period / 2)))]

            #create axis limitations for every sensor
            dict_axs[sensor[0]] = {}
            if axs_limitations == "general":
                # get 1% and 99% quantiles of x, y and z axis and put in dictionary
                dict_axs[sensor[0]]["x_01"] = df_sensor["x-axis"].quantile(0.01)
                dict_axs[sensor[0]]["x_99"] = df_sensor["x-axis"].quantile(0.99)
                dict_axs[sensor[0]]["y_01"] = df_sensor["y-axis"].quantile(0.01)
                dict_axs[sensor[0]]["y_99"] = df_sensor["y-axis"].quantile(0.99)
                dict_axs[sensor[0]]["z_01"] = df_sensor["z-axis"].quantile(0.01)
                dict_axs[sensor[0]]["z_99"] = df_sensor["z-axis"].quantile(0.99)

            elif axs_limitations == "event":
                # get 1% and 99% quantiles of x, y and z axis and put in dictionary
                dict_axs[sensor[0]]["x_01"] = df_event["x-axis"].quantile(0.01)
                dict_axs[sensor[0]]["x_99"] = df_event["x-axis"].quantile(0.99)
                dict_axs[sensor[0]]["y_01"] = df_event["y-axis"].quantile(0.01)
                dict_axs[sensor[0]]["y_99"] = df_event["y-axis"].quantile(0.99)
                dict_axs[sensor[0]]["z_01"] = df_event["z-axis"].quantile(0.01)
                dict_axs[sensor[0]]["z_99"] = df_event["z-axis"].quantile(0.99)

            # check if df_event is empty: if yes, break function and return None
            if df_event.empty:
                print("df_event is empty with event_time: " + str(event_time) + " and sensor: " + sensor[0])
                return None, None

            # transform unix time into seconds relative to event_time
            df_event["time relative to event (s)"] = df_event["timestamp"] - event_time
            df_event["time relative to event (s)"] = df_event["time relative to event (s)"].dt.total_seconds()

            #store this df_event in a dictionary with sensor name as key
            dict_df_event[sensor[0]] = df_event

            # print progress
            #print("Sensor " + sensor[0] + " done and took " + str((time.time() - time0)/60) + " minutes")

        # compute participant ID: take the first one of df_event
        # print("ESM timestamp: ", event_time)
        participant_id = df_event["device_id"].iloc[0]
        # compute esm_answer
        activity_name = df_event[label_column_name].iloc[0]

        # visualize all df_events in one plot
        ## create number of subplots based on number of sensors
        if len(list_sensors) == 4 or len(list_sensors) == 3:
            fig, axs = plt.subplots(6, 2, figsize=(15, 10))
            # plot sensordata from first sensor in dict_df_event on first subplot
            # create title for first subplot
            axs[0, 0].set_title(list_sensors[0][0], fontsize=15)
            sns.lineplot(x="time relative to event (s)", y="x-axis", data=dict_df_event[list_sensors[0][0]], ax=axs[0, 0])
            sns.lineplot(x="time relative to event (s)", y="y-axis", data=dict_df_event[list_sensors[0][0]], ax=axs[1, 0])
            sns.lineplot(x="time relative to event (s)", y="z-axis", data=dict_df_event[list_sensors[0][0]], ax=axs[2, 0])
            # limit axes
            axs[0, 0].set_ylim(dict_axs[list_sensors[0][0]]["x_01"], dict_axs[list_sensors[0][0]]["x_99"])
            axs[1, 0].set_ylim(dict_axs[list_sensors[0][0]]["y_01"], dict_axs[list_sensors[0][0]]["y_99"])
            axs[2, 0].set_ylim(dict_axs[list_sensors[0][0]]["z_01"], dict_axs[list_sensors[0][0]]["z_99"])

            # plot sensordata from second sensor in dict_df_event on second  subplots
            axs[0, 1].set_title(list_sensors[1][0], fontsize=15)
            sns.lineplot(x="time relative to event (s)", y="x-axis", data=dict_df_event[list_sensors[1][0]], ax=axs[0, 1])
            sns.lineplot(x="time relative to event (s)", y="y-axis", data=dict_df_event[list_sensors[1][0]], ax=axs[1, 1])
            sns.lineplot(x="time relative to event (s)", y="z-axis", data=dict_df_event[list_sensors[1][0]], ax=axs[2, 1])
            # limit axes
            axs[0, 1].set_ylim(dict_axs[list_sensors[1][0]]["x_01"], dict_axs[list_sensors[1][0]]["x_99"])
            axs[1, 1].set_ylim(dict_axs[list_sensors[1][0]]["y_01"], dict_axs[list_sensors[1][0]]["y_99"])
            axs[2, 1].set_ylim(dict_axs[list_sensors[1][0]]["z_01"], dict_axs[list_sensors[1][0]]["z_99"])

            # plot sensordata from third sensor in dict_df_event on third subplots
            axs[3, 0].set_title(list_sensors[2][0], fontsize=15)
            sns.lineplot(x="time relative to event (s)", y="x-axis", data=dict_df_event[list_sensors[2][0]], ax=axs[3, 0])
            sns.lineplot(x="time relative to event (s)", y="y-axis", data=dict_df_event[list_sensors[2][0]], ax=axs[4, 0])
            sns.lineplot(x="time relative to event (s)", y="z-axis", data=dict_df_event[list_sensors[2][0]], ax=axs[5, 0])
            # limit axes
            axs[3, 0].set_ylim(dict_axs[list_sensors[2][0]]["x_01"], dict_axs[list_sensors[2][0]]["x_99"])
            axs[4, 0].set_ylim(dict_axs[list_sensors[2][0]]["y_01"], dict_axs[list_sensors[2][0]]["y_99"])
            axs[5, 0].set_ylim(dict_axs[list_sensors[2][0]]["z_01"], dict_axs[list_sensors[2][0]]["z_99"])

            # if there are four sensors, plot sensordata from fourth sensor in dict_df_event on fourth subplots
            if len(list_sensors) == 4:
                axs[3, 1].set_title(list_sensors[3][0], fontsize=15)
                sns.lineplot(x="time relative to event (s)", y="x-axis", data=dict_df_event[list_sensors[3][0]], ax=axs[3, 1])
                sns.lineplot(x="time relative to event (s)", y="y-axis", data=dict_df_event[list_sensors[3][0]], ax=axs[4, 1])
                sns.lineplot(x="time relative to event (s)", y="z-axis", data=dict_df_event[list_sensors[3][0]], ax=axs[5, 1])
                # limit axes
                axs[3, 1].set_ylim(dict_axs[list_sensors[3][0]]["x_01"], dict_axs[list_sensors[3][0]]["x_99"])
                axs[4, 1].set_ylim(dict_axs[list_sensors[3][0]]["y_01"], dict_axs[list_sensors[3][0]]["y_99"])
                axs[5, 1].set_ylim(dict_axs[list_sensors[3][0]]["z_01"], dict_axs[list_sensors[3][0]]["z_99"])

        elif len(list_sensors) == 2:
            fig, axs = plt.subplots(3, 2, figsize=(15, 10))
            # plot sensordata from first sensor in dict_df_event on first three subplots
            axs[0, 0].set_title(list_sensors[0][0], fontsize=15)
            sns.lineplot(x="time relative to event (s)", y="x-axis", data=dict_df_event[list_sensors[0][0]], ax=axs[0, 0])
            sns.lineplot(x="time relative to event (s)", y="y-axis", data=dict_df_event[list_sensors[0][0]], ax=axs[1, 0])
            sns.lineplot(x="time relative to event (s)", y="z-axis", data=dict_df_event[list_sensors[0][0]], ax=axs[2, 0])
            # limit axes
            axs[0, 0].set_ylim(dict_axs[list_sensors[0][0]]["x_01"], dict_axs[list_sensors[0][0]]["x_99"])
            axs[1, 0].set_ylim(dict_axs[list_sensors[0][0]]["y_01"], dict_axs[list_sensors[0][0]]["y_99"])
            axs[2, 0].set_ylim(dict_axs[list_sensors[0][0]]["z_01"], dict_axs[list_sensors[0][0]]["z_99"])
            # plot sensordata from second sensor in dict_df_event on second three subplots
            axs[0, 1].set_title(list_sensors[1][0], fontsize=15)
            sns.lineplot(x="time relative to event (s)", y="x-axis", data=dict_df_event[list_sensors[1][0]], ax=axs[0, 1])
            sns.lineplot(x="time relative to event (s)", y="y-axis", data=dict_df_event[list_sensors[1][0]], ax=axs[1, 1])
            sns.lineplot(x="time relative to event (s)", y="z-axis", data=dict_df_event[list_sensors[1][0]], ax=axs[2, 1])
            # limit axes
            axs[0, 1].set_ylim(dict_axs[list_sensors[1][0]]["x_01"], dict_axs[list_sensors[1][0]]["x_99"])
            axs[1, 1].set_ylim(dict_axs[list_sensors[1][0]]["y_01"], dict_axs[list_sensors[1][0]]["y_99"])
            axs[2, 1].set_ylim(dict_axs[list_sensors[1][0]]["z_01"], dict_axs[list_sensors[1][0]]["z_99"])

        # set title for whole plot
        plt.suptitle("Sensordata around Activity " + activity_name +
                     "\n(Participant " + str(participant_id) + " at " + str(event_time) + " with axis based on " + axs_limitations + " data)", fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        # plt.show()

        return fig, activity_name


#region Iteration 01
#load data & initialize
label_column_name = "label_human motion - general"
time_period = 90  # seconds
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/human_motion_general/"

#region sensor-based: visualize data around events: one plot per event & sensor
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/human_motion_general/"

sensor_list = [["Linear Accelerometer", "linear_accelerometer"],
               ["Gyroscope", "gyroscope"],
                ["Magnetometer", "magnetometer"],
                ["Rotation", "rotation"]]
for sensor in sensor_list:
    print("Visualize sensor: ", sensor[0])
    sensordata_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/" + sensor[1] + "_esm_timeperiod_5 min.csv_JSONconverted.pkl"
    sensor_name = sensor[0]
    # load & preprocess data
    df_sensor = data_exploration_sensors(sensordata_path).load_sensordata()
    df_sensor = Merge_Transform.merge_participantIDs(df_sensor, users)  # merge participant IDs
    df_sensor = labeling_sensor_df(df_sensor, dict_label, label_column_name)
    df_sensor = df_sensor.dropna(subset=[label_column_name])  # delete all rows which contain NaN values in the label column

    for activity_name in df_sensor[label_column_name].unique():
        print("Started visualization for activity: " + activity_name)
        data_exploration_sensors.vis_sensordata_around_event_foractivity(activity_name, df_sensor, time_period, sensor_name, label_column_name, path_to_save)
#endregion

#region sensor-based: find visualization for single events (one plot per event & sensor)
sensordata_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/linear_accelerometer_esm_timeperiod_5 min.csv_JSONconverted.pkl"
sensor_name = "Linear Accelerometer"
df_sensor = data_exploration_sensors(sensordata_path).load_sensordata()
df_sensor = Merge_Transform.merge_participantIDs(df_sensor, users) #merge participant IDs
df_sensor = labeling_sensor_df(df_sensor, dict_label, label_column_name)
df_sensor = df_sensor.dropna(subset=[label_column_name]) # delete all rows which contain NaN values in the label column

event_time = "2022-07-02 04:23:01.163000064"
fig = data_exploration_sensors.vis_sensordata_around_event(df_sensor, event_time, time_period, sensor_name, label_column_name)
plt.show()
#endregion

#region event-based_ visualize data around events: one plot per event & several sensors: for all events
path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/INSERT_esm_timeperiod_5 min.csv_JSONconverted.pkl"
list_sensors =  ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "locations","plugin_ios_activity_recognition"]
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/human_motion_general/"
axs_limitations = "general"
label_data = True

sensor_list = [["Linear Accelerometer", "linear_accelerometer"],
               ["Gyroscope", "gyroscope"],
               ["Magnetometer", "magnetometer"],
                ["Rotation", "rotation"]]

# create list with all event times
event_times = []
for key in dict_label.keys():
    event_times.append(key)

# apply vis_sensordata_around_event for every event time in event_times
num_events = 1
sensor_names = ""
for sensor in sensor_list:
    sensor_names += sensor[1] + "_"
for event_time in event_times:
    time0 = time.time()
    fig, activity_name = data_exploration_sensors.vis_sensordata_around_event_severalsensors(sensor_list, event_time, time_period,
                                                                           label_column_name, path_sensordata,axs_limitations, label_data)
    # check if fig is None: if yes, continue with next event_time
    if fig is None:
        continue
    fig.savefig(
        path_to_save + activity_name + "/" + activity_name + "_EventTime-" + str(event_time) + "_Segment-" + str(
            time_period) + "_Sensors-" + sensor_names + ".png")
    plt.close(fig)
    # print progress
    print("Finished event " + str(num_events) + " of " + str(len(event_times)) + " in " + str((time.time() - time0)/60) + " minutes.")
    num_events += 1
#endregion

#region event-based_ visualize data around events: single event (one plot per event & several sensors)
path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/INSERT_esm_timeperiod_5 min.csv_JSONconverted.pkl"
#path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/perfectdata/INSERT_labeled.csv"
list_sensors =  ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "locations","plugin_ios_activity_recognition"]
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/sensors/human_motion_general/"
event_time = "2022-07-03 10:00:04.428000000"
sensor_list = [["Linear Accelerometer", "linear_accelerometer"],
               ["Magnetometer", "magnetometer"],
               ["Gyroscope", "gyroscope"],
                ["Rotation", "rotation"]]
sensor_names = ""
for sensor in sensor_list:
    sensor_names += sensor[1] + "_"
fig, activity_name = data_exploration_sensors.vis_sensordata_around_event_foursensors(sensor_list, event_time, time_period,
                                                                       label_column_name, path_sensordata)
# check if fig is None: if yes, continue with next event_time
if fig is None:
    continue
fig.savefig(
    path_to_save + activity_name + "/" + activity_name + "_EventTime-" + str(event_time) + "_Segment-" + str(
        time_period) + "_Sensors-" + sensor_names + ".png")
plt.show()
#endregion

#endregion

#region Iteration 02
#load data & initialize
label_column_name = "label_human motion - general"
time_period = 90  # seconds
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben/dict_label_iteration02_Ben.pkl")
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration02/data_exploration/sensors/human_motion_general/"

# create events (=ESM_timestamps) column in the sensordata (based on
#iterate through all sensorfiles and add ESM_timestamps column
for sensor in list_sensors:
    print("start with sensor " + sensor[1])
    path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben/" + sensor[1] + "_labeled.csv"
    df_sensor = pd.read_csv(path_sensordata)
    # convert timestamp to datetime
    df_sensor["timestamp"] = pd.to_datetime(df_sensor["timestamp"])

    # iterate through dict_label and create a list of all events (=ESM_timestamps)
    df_sensor["ESM_timestamp"] = ""
    for key in dict_label.keys():
        print("start with key " + str(key))
        # create list of times: every "time_period" second from start_session ongoing one timestamp (until end_session)
        event_times = []
        start_session = dict_label[key]["start_session"]
        end_session = dict_label[key]["end_session"]
        event_time = start_session + datetime.timedelta(seconds=(time_period/2))
        #run while loop until event_time is bigger than end_session
        while event_time < end_session:
            event_times.append(event_time)
            event_time = event_time + datetime.timedelta(seconds=time_period)

        # iterate through event_times: for each timestamp in df_sensor, check if it is in "time_period" around event_time
        # if yes, add event_time to df_sensor["ESM_timestamps"]
        for event_time in event_times:
            row_count = 1
            for index, row in df_sensor.iterrows():
                if row["timestamp"] >= event_time - datetime.timedelta(seconds=(time_period/2)) and row["timestamp"] <= event_time + datetime.timedelta(seconds=(time_period/2)):
                    df_sensor.at[index, "ESM_timestamp"] = event_time
                row_count += 1

    # save df_sensor
    df_sensor.to_csv(path_sensordata + "_with_ESM_timestamps.csv", index=False)
    #df_sensor.to_pickle(path_sensordata + "_with_ESM_timestamps.pkl")

#create visualization for df_sensor


#region sensor-based: find visualization for single events (one plot per event & sensor)
sensor_name = "linear_accelerometer"
event_time = "2022-11-30 15:56:45"
path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben/INSERT_labeled.csv"
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration02/data_exploration/sensors/label_human motion - general/"
axs_limitations = "general" #if y-axis limits should be based on whole sensor data (general) or on data around event (event)

path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben/" + sensor_name + "_labeled.csv_with_ESM_timestamps.csv"

df_sensor = data_exploration_sensors.load_sensordata(path_sensordata)
df_sensor = Merge_Transform.merge_participantIDs(df_sensor, users) #merge participant IDs
df_sensor = df_sensor.dropna(subset=[label_column_name]) # delete all rows which contain NaN values in the label column

fig, activity_name = data_exploration_sensors.vis_sensordata_around_event_onesensor(df_sensor, event_time, time_period, sensor_name, label_column_name, axs_limitations)
plt.show()
fig.savefig(path_to_save + activity_name + "_EventTime-" + str(event_time) + "_Segment-" + str(
            time_period) + "_Sensors-" + sensor_name + "_axis_limitations-" + axs_limitations + ".png")
#endregion

#region event-based_ visualize data around events: single event (one plot per event & several sensors) NOT FINISHED
path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/perfectdata/INSERT_labeled.csv"
list_sensors =  ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "locations","plugin_ios_activity_recognition"]
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration02/data_exploration/sensors/human_motion_general/"
event_time = "2022-07-03 10:00:04.428000000"
sensor_list = [["Linear Accelerometer", "linear_accelerometer"],
               ["Magnetometer", "magnetometer"],
               ["Gyroscope", "gyroscope"],
                ["Rotation", "rotation"]]
sensor_names = ""
for sensor in sensor_list:
    sensor_names += sensor[1] + "_"
fig, activity_name = data_exploration_sensors.vis_sensordata_around_event_foursensors(sensor_list, event_time, time_period,
                                                                       label_column_name, path_sensordata)
# check if fig is None: if yes, continue with next event_time
if fig is None:
    continue
fig.savefig(
    path_to_save + activity_name + "/" + activity_name + "_EventTime-" + str(event_time) + "_Segment-" + str(
        time_period) + "_Sensors-" + sensor_names + ".png")
plt.show()
#endregion

#region event-based_ visualize data around events: one plot per event & several sensors: for all events
path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben/INSERT_labeled.csv"
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration02/data_exploration/sensors/human_motion_general/"
list_sensors = [["Linear Accelerometer", "linear_accelerometer"],
               ["Magnetometer", "magnetometer"],
               ["Gyroscope", "gyroscope"],
                ["Rotation", "rotation"]]
axs_limitations = "general" #if y-axis limits should be based on whole sensor data (general) or on data around event (event)


#create list of events (go through events from one sensor)
path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben/" + sensor_list[0][1] + "_labeled.csv_with_ESM_timestamps.csv"
df_sensor = pd.read_csv(path_sensordata)
event_times = df_sensor["ESM_timestamp"].unique()
event_times = event_times[~pd.isnull(event_times)]# delete nan

# apply vis_sensordata_around_event for every event time in event_times
path_sensordata = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/Ben/INSERT_labeled.csv_with_ESM_timestamps.csv"
path_to_save = "/Users/benediktjordan/Documents/MTS/Iteration02/data_exploration/sensors/label_human motion - general/"
num_events = 1
sensor_names = ""
for sensor in sensor_list:
    sensor_names += sensor[1] + "_"

for event_time in event_times:
    time_0 = time.time()
    fig, activity_name = data_exploration_sensors.vis_sensordata_around_event_severalsensors(list_sensors, event_time, time_period,
                                                                           label_column_name, path_sensordata, axs_limitations)
    # check if fig is None: if yes, continue with next event_time
    if fig is None:
        print("fig is None")
        continue
    fig.savefig(
        path_to_save + activity_name + "_EventTime-" + str(event_time) + "_Segment-" + str(
            time_period) + "_Sensors-" + sensor_names + "_axis_limitations-" + axs_limitations + ".png")
    plt.close(fig)
    # print progress
    print("Finished event " + str(num_events) + " of " + str(len(event_times)) + " in " + str((time.time() - time_0)/60) + " minutes.")#
    # print time needed for processing
    num_events += 1
#endregion



#endregion





#testrun
test = df_sensor["ESM_timestamp"][df_sensor["ESM_timestamp"] == event_time]
## create df_event by filtering for +-30 seconds around timestamp
#convert timestamp to datetime in df_sensor
df_sensor["timestamp"] = pd.to_datetime(df_sensor["timestamp"])
event_time = pd.to_datetime(event_time)
df_event = df_sensor[(df_sensor["ESM_timestamp"] == event_time) & (df_sensor["timestamp"] >= (pd.to_datetime(event_time) - pd.Timedelta("30s"))) & (df_sensor["timestamp"] <= (pd.to_datetime(event_time) + pd.Timedelta("30s")))]
#create time relative to event time
df_event["time relative to event (s)"] = df_event["timestamp"] - pd.to_datetime(event_time)
# rename lin_double_values_0, 1, and 2 in x-axis, y-axis, z-axis
df_event = df_event.rename(columns={"lin_double_values_0": "x-axis", "lin_double_values_1": "y-axis", "lin_double_values_2": "z-axis"})
##create 12 lineplots
fig, axs = plt.subplots(6, 2, figsize=(15, 10))
sns.lineplot(x="time relative to event (s)", y="x-axis", data=df_event, ax=axs[0,0])
sns.lineplot(x="time relative to event (s)", y="y-axis", data=df_event, ax=axs[0,1])
sns.lineplot(x="time relative to event (s)", y="z-axis", data=df_event, ax=axs[1,0])
sns.lineplot(x="time relative to event (s)", y="x-axis", data=df_event, ax=axs[1,1])
sns.lineplot(x="time relative to event (s)", y="y-axis", data=df_event, ax=axs[2,0])
sns.lineplot(x="time relative to event (s)", y="z-axis", data=df_event, ax=axs[2,1])
sns.lineplot(x="time relative to event (s)", y="x-axis", data=df_event, ax=axs[3,0])
sns.lineplot(x="time relative to event (s)", y="y-axis", data=df_event, ax=axs[3,1])
sns.lineplot(x="time relative to event (s)", y="z-axis", data=df_event, ax=axs[4,0])
sns.lineplot(x="time relative to event (s)", y="x-axis", data=df_event, ax=axs[4,1])
sns.lineplot(x="time relative to event (s)", y="y-axis", data=df_event, ax=axs[5,0])
sns.lineplot(x="time relative to event (s)", y="z-axis", data=df_event, ax=axs[5,1])
#include title for every subplot
axs[0,0].set_title("x-axis")
axs[0,1].set_title("y-axis")
axs[1,0].set_title("z-axis")
axs[1,1].set_title("x-axis")
axs[2,0].set_title("y-axis")
axs[2,1].set_title("z-axis")
axs[3,0].set_title("x-axis")
axs[3,1].set_title("y-axis")
axs[4,0].set_title("z-axis")
axs[4,1].set_title("x-axis")
axs[5,0].set_title("y-axis")
axs[5,1].set_title("z-axis")
# tight layout
plt.tight_layout()
plt.show()







#troubleshooting
event_time = "2022-07-12 06:34:08.707000064"
event_time = pd.to_datetime(event_time)
df_sensor["ESM_timestamp"] = pd.to_datetime(df_sensor["ESM_timestamp"])
testhouse = df_sensor[df_sensor["ESM_timestamp"] == event_time]
# print number of entries for every esm timestamp
event_times = df_sensor["ESM_timestamp"].value_counts()
# order event_times by index ascending
event_times = event_times.sort_index()
event_times = df_sensor["ESM_timestamp"].unique()
#order them ascending
event_times.sort()

# get type of esm timestamp
type(df_sensor["ESM_timestamp"])




# region create function which visualizes the data in lineplot for sensor, participant, ESN event, ESM answer and in timeperiod
dir_databases = "H:/InUsage/Databases"
esm_all_transformed = pd.read_csv(dir_databases + "/esm_all_transformed.csv")

# only data for one participant
esm_all_transformed_p1 = esm_all_transformed[
    esm_all_transformed["device_id"] == "8206b501-20e7-4851-8b79-601af9e4485f"]

participant_id = "8206b501-20e7-4851-8b79-601af9e4485f"
sensor = "accelerometer"
event_time = 1656166542710.00000
esm_answer = "walking"
time_period = 60  # seconds
sensordata_path = "H:/InUsage/Databases/db_iteration1_20220628/accelerometer.csv"

def vis_event_acc(participant_ID, event_time, esm_answer, time_period, sensordata_path):
    # create df with sensor data around event_time of participant_ID
    df = pd.read_csv(sensordata_path)
    df_filtered = df[(df["1"] >= event_time - time_period * 1000) & (df["1"] <= event_time + time_period * 1000)]
    df_filtered = df_filtered[df_filtered["2"] == participant_id]

    # transform unix time into seconds relative to event_time
    df_filtered.sort_values(by=['1'], inplace=True)
    df_filtered["1"] = (df_filtered["1"] - event_time) / 1000
    # df_filtered["1"] = (df_filtered["1"] - df_filtered["1"].iloc[0])/1000

    # transform JSON data into columns
    df_filtered = df_filtered.reset_index(drop=True)
    stdf = df_filtered["3"].apply(json.loads)
    df_transformed = pd.DataFrame(stdf.tolist())  # or stdf.apply(pd.Series)
    df_transformed = pd.concat([df_filtered, df_transformed], axis=1)

    # visualize for every one of the three axis the data in a lineplot
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    axs[0].plot(df_transformed["1"], df_transformed["double_values_0"], color="blue", label="x")
    axs[1].plot(df_transformed["1"], df_transformed["double_values_1"], color="red", label="y")
    axs[2].plot(df_transformed["1"], df_transformed["double_values_2"], color="green", label="z")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

# test runs

## decide event time
dir_databases = "H:/InUsage/Databases"
esm_all_transformed = pd.read_csv(dir_databases + "/esm_all_transformed.csv")
esm_all_transformed.sort_values(by=['timestamp'], inplace=True)
esm_all_transformed["timestamp"] = pd.to_datetime(esm_all_transformed['timestamp'], unit='ms')

# only data for one participant
esm_all_transformed_p1 = esm_all_transformed[esm_all_transformed["device_id"] == participant_id]
esm_all_transformed_p1_sitting = esm_all_transformed[
    esm_all_transformed["bodyposition"] == "[\"sitting: at a table\"]"]
esm_all_transformed_p1_standing = esm_all_transformed[esm_all_transformed["bodyposition"] == "[\"standing\"]"]
esm_all_transformed_p1_lying = esm_all_transformed[esm_all_transformed["bodyposition"] == "[\"lying\"]"]

# participant: "8206b501-20e7-4851-8b79-601af9e4485f"
## walking (db_iteration1_20220624)
event_time_list = [1656166542710.00000,
                   1656219883193.00000,
                   1656312067352.00000,
                   1656327932865.00000,
                   1656338500955.00000,
                   1656347904183.00000,
                   1656397267213.00000]
event_time = 1656166542710.00000
event_time = 1656219883193.00000
event_time = 1656312067352.00000
event_time = 1656327932865.00000
event_time = 1656338500955.00000
event_time = 1656347904183.00000
event_time = 1656397267213.00000

## cycling (db_iteration1_20220628)
event_time = 1656486039637.00000
event_time = 1656625509247.00000

## sitting: at a table, smartphone in hand (db_iteration1_20220624)
event_time_list = [1656079231617.00000,
                   1656338426511.00000]

## standing, smartphone in hand (db_iteration1_20220624)
event_time_list = [1656091225973.00000,
                   1656100136895.00000,
                   1656153702404.00000,
                   1656324595078.00000,
                   1656331226610.00000,
                   1656361973954.00000]

## lying, smartphone in hand (db_iteration1_20220624)
event_time_list = [1656104411911.00000,
                   1656280341624.00000,
                   1656349255223.00000]

participant_id = "8206b501-20e7-4851-8b79-601af9e4485f"
sensor = "gyroscope"
esm_answer = "sitting: at a table; smartphone in hand"
time_period = 30  # seconds
time_period_list = [30, 300, 600]
sensordata_path = "H:/InUsage/Databases/db_iteration1_20220624/" + sensor + ".csv"

## load data
time_begin = time.time()
df = pd.read_csv(sensordata_path)
time_end = time.time()
print("read csv in " + str(time_end - time_begin) + " seconds")

def vis_event_acc_linacc(df_sensordata, participant_ID, event_time, esm_answer, time_period):

    ## filter data for time period around event time
    df_filtered = df[(df["1"] >= event_time - time_period * 1000) & (df["1"] <= event_time + time_period * 1000)]
    df_filtered = df_filtered[df_filtered["2"] == participant_id]

    # transform unix time into seconds relative to event_time
    df_filtered.sort_values(by=['1'], inplace=True)
    df_filtered["1"] = (df_filtered["1"] - event_time) / 1000
    # df_filtered["1"] = (df_filtered["1"] - df_filtered["1"].iloc[0])/1000

    # transform JSON data into columns
    df_filtered = df_filtered.reset_index(drop=True)
    stdf = df_filtered["3"].apply(json.loads)
    df_transformed = pd.DataFrame(stdf.tolist())  # or stdf.apply(pd.Series)
    df_transformed = pd.concat([df_filtered, df_transformed], axis=1)

    # visualize for every one of the three axis the data in a lineplot

    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle("participant 1 " + "" + sensor + " " + esm_answer, fontsize=36)

    axs[0].plot(df_transformed["1"], df_transformed["double_values_0"], color="blue", label="x")
    axs[1].plot(df_transformed["1"], df_transformed["double_values_1"], color="red", label="y")
    axs[2].plot(df_transformed["1"], df_transformed["double_values_2"], color="green", label="z")

    ## set y-axis limi
    axs[0].set_ylim([-1, 1])
    axs[1].set_ylim([-1, 1])
    axs[2].set_ylim([-1, 1])

    ## Tight layout often produces nice results
    ## but requires the title to be spaced accordingly
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()

    # save figure
    fig.savefig(
        "D:/MasterThesis @nlbb/Iteration01/Results/visualizations/bodyposition/" + participant_id + "_" + sensor + "_" + esm_answer + "_" + str(
            time_period) + "s_event_time_" + str(event_time) + ".png")
    plt.close()

for event_time in event_time_list:
    for time_period in time_period_list:
        vis_event_acc_linacc(df, participant_id, event_time, esm_answer, time_period)

# endregion

# region create summary statistics for every sensor for different time periods around ESM events
def create_summary_statistics_for_sensors(path_sensor, sensor, time_period, dir_results):
    # create summary stats dataframe
    device_id = []
    ESM_timestamp = []
    ESM_location = []
    ESM_location_time = []
    ESM_bodyposition = []
    ESM_bodyposition_time = []
    ESM_activity = []
    ESM_activity_time = []
    ESM_smartphonelocation = []
    ESM_smartphonelocation_time = []
    ESM_aligned = []
    ESM_aligned_time = []
    double_values_0_mean = []
    double_values_0_std = []
    double_values_1_mean = []
    double_values_1_std = []
    double_values_2_mean = []
    double_values_2_std = []
    double_values_3_mean = []
    double_values_3_std = []
    double_speed_mean = []
    double_speed_std = []
    double_bearing_mean = []
    double_bearing_std = []
    double_altitude_mean = []
    double_altitude_std = []
    double_altitude_mean = []
    double_altitude_std = []
    double_longitude_mean = []
    double_longitude_std = []
    accuracy_mean = []
    accuracy_std = []

    # track missing sensordata around events
    missing_sensordata = []

    summary_stats = pd.DataFrame()

    # load sensor csv
    sensordata = pd.read_csv(path_sensor, dtype={"0": 'int64', "1": 'float64', "2": object, "3": object})

    # iterate through esm events
    esm_events = sensordata["ESM_timestamp"].unique()
    for esm_event in esm_events:
        print("esm_event: " + str(esm_event))
        # create dataframe for every esm event and time segment around event
        esm_event_df = sensordata[sensordata["ESM_timestamp"] == esm_event]
        esm_event_df = esm_event_df[(esm_event_df["1"] >= esm_event - time_period * 1000) & (
                    esm_event_df["1"] <= esm_event + time_period * 1000)]

        if len(esm_event_df) > 0:
            # transform JSON data into columns
            esm_event_df = esm_event_df.reset_index(drop=True)
            stdf = esm_event_df["3"].apply(json.loads)
            df_transformed = pd.DataFrame(stdf.tolist())  # or stdf.apply(pd.Series)
            df_transformed = pd.concat([esm_event_df, df_transformed], axis=1)

            # create summary stats for every esm event
            summary_stats_esm_event = df_transformed.describe()

            # add esm event to summary stats lists
            device_id.append(esm_event_df["2"].iloc[0])
            ESM_timestamp.append(esm_event)
            ESM_location.append(esm_event_df["ESM_location"].iloc[0])
            ESM_location_time.append(esm_event_df["ESM_location_time"].iloc[0])
            ESM_bodyposition.append(esm_event_df["ESM_bodyposition"].iloc[0])
            ESM_bodyposition_time.append(esm_event_df["ESM_bodyposition_time"].iloc[0])
            ESM_activity.append(esm_event_df["ESM_activity"].iloc[0])
            ESM_activity_time.append(esm_event_df["ESM_activity_time"].iloc[0])
            ESM_smartphonelocation.append(esm_event_df["ESM_smartphonelocation"].iloc[0])
            ESM_smartphonelocation_time.append(esm_event_df["ESM_smartphonelocation_time"].iloc[0])
            ESM_aligned.append(esm_event_df["ESM_aligned"].iloc[0])
            ESM_aligned_time.append(esm_event_df["ESM_aligned_time"].iloc[0])

            # differentiate between sensor types
            if sensor in ["accelerometer", "magnetometer"]:
                accuracy_mean.append(summary_stats_esm_event["accuracy"]["mean"])
                accuracy_std.append(summary_stats_esm_event["accuracy"]["std"])
                double_values_0_mean.append(summary_stats_esm_event["double_values_0"]["mean"])
                double_values_0_std.append(summary_stats_esm_event["double_values_0"]["std"])
                double_values_1_mean.append(summary_stats_esm_event["double_values_1"]["mean"])
                double_values_1_std.append(summary_stats_esm_event["double_values_1"]["std"])
                double_values_2_mean.append(summary_stats_esm_event["double_values_2"]["mean"])
                double_values_2_std.append(summary_stats_esm_event["double_values_2"]["std"])

            elif sensor in ["barometer"]:
                accuracy_mean.append(summary_stats_esm_event["accuracy"]["mean"])
                accuracy_std.append(summary_stats_esm_event["accuracy"]["std"])
                double_values_0_mean.append(summary_stats_esm_event["double_values_0"]["mean"])
                double_values_0_std.append(summary_stats_esm_event["double_values_0"]["std"])

            elif sensor in ["locations"]:
                accuracy_mean.append(summary_stats_esm_event["accuracy"]["mean"])
                accuracy_std.append(summary_stats_esm_event["accuracy"]["std"])
                double_speed_mean.append(summary_stats_esm_event["double_speed"]["mean"])
                double_speed_std.append(summary_stats_esm_event["double_speed"]["std"])
                double_bearing_mean.append(summary_stats_esm_event["double_bearing"]["mean"])
                double_bearing_std.append(summary_stats_esm_event["double_bearing"]["std"])
                double_altitude_mean.append(summary_stats_esm_event["double_altitude"]["mean"])
                double_altitude_std.append(summary_stats_esm_event["double_altitude"]["std"])
                double_altitude_meanitude_mean.append(summary_stats_esm_event["double_latitude"]["mean"])
                double_altitude_std.append(summary_stats_esm_event["double_latitude"]["std"])
                double_longitude_mean.append(summary_stats_esm_event["double_longitude"]["mean"])
                double_longitude_std.append(summary_stats_esm_event["double_longitude"]["std"])

            elif sensor in ["rotations"]:
                accuracy_mean.append(summary_stats_esm_event["accuracy"]["mean"])
                accuracy_std.append(summary_stats_esm_event["accuracy"]["std"])
                double_values_0_mean.append(summary_stats_esm_event["double_values_0"]["mean"])
                double_values_0_std.append(summary_stats_esm_event["double_values_0"]["std"])
                double_values_1_mean.append(summary_stats_esm_event["double_values_1"]["mean"])
                double_values_1_std.append(summary_stats_esm_event["double_values_1"]["std"])
                double_values_2_mean.append(summary_stats_esm_event["double_values_2"]["mean"])
                double_values_2_std.append(summary_stats_esm_event["double_values_2"]["std"])
                double_values_3_mean.append(summary_stats_esm_event["double_values_3"]["mean"])
                double_values_3_std.append(summary_stats_esm_event["double_values_3"]["std"])

            print("Progress: " + str(esm_events.tolist().index(esm_event)) + "/" + str(len(esm_events)))

        else:
            missing_sensordata.append(esm_event)
            print("missing sensordata for esm event: " + str(esm_event))

    # create summary stats dataframe
    summary_stats["ESM_timestamp"] = ESM_timestamp
    summary_stats["device_id"] = device_id
    summary_stats["ESM_location"] = ESM_location
    summary_stats["ESM_location_time"] = ESM_location_time
    summary_stats["ESM_bodyposition"] = ESM_bodyposition
    summary_stats["ESM_bodyposition_time"] = ESM_bodyposition_time
    summary_stats["ESM_activity"] = ESM_activity
    summary_stats["ESM_activity_time"] = ESM_activity_time
    summary_stats["ESM_smartphonelocation"] = ESM_smartphonelocation
    summary_stats["ESM_smartphonelocation_time"] = ESM_smartphonelocation_time
    summary_stats["ESM_aligned"] = ESM_aligned
    summary_stats["ESM_aligned_time"] = ESM_aligned_time
    summary_stats["double_values_0_mean"] = double_values_0_mean
    summary_stats["double_values_0_std"] = double_values_0_std
    summary_stats["double_values_1_mean"] = double_values_1_mean
    summary_stats["double_values_1_std"] = double_values_1_std
    summary_stats["double_values_2_mean"] = double_values_2_mean
    summary_stats["double_values_2_std"] = double_values_2_std

    return summary_stats, missing_sensordata

path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_barometer_esm_timeperiod_5 min.csv"
test_df = pd.read_csv(path_sensor, dtype={"0": 'int64', "1": 'float64', "2": object, "3": object}, nrows=100)

dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/summary_stats"
sensor_list = ["locations", "rotation"]
sensor_list = ["linear_accelerometer", "gyroscope"]
# time_periods = [10,20,40]
time_periods = [5]
for sensor in sensor_list:
    for time_period in time_periods:
        # path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_" + sensor + "_esm_timeperiod_5 min.csv"
        path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/" + sensor + "_esm_timeperiod_5 min.csv"
        summary_statistics, missing_sensordata = create_summary_statistics_for_sensors(path_sensor, sensor,
                                                                                       time_period, dir_results)
        summary_statistics.to_csv(dir_results + "/" + sensor + "_summary_stats_" + str(time_period) + ".csv",
                                  index=False)
        missing_sensordata_df = pd.DataFrame(missing_sensordata)
        missing_sensordata_df.to_csv(
            dir_results + "/" + sensor + "_summary_stats_missing_sensordata" + str(time_period) + ".csv",
            index=False)

# to check which columns to select: temporary
path_sensor = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_magnetometer_esm_timeperiod_5 min.csv_JSONconverted.csv"
df = pd.read_csv(path_sensor, nrows=100)
summary = df.describe()
df["double_values_0"].value_counts()
df["double_values_1"].value_counts()
df["double_values_2"].value_counts()
df["double_values_3"].value_counts()
df["accuracy"].value_counts()

time_begin = time.time()
sensor = "accelerometer"
time_period = 10  # seconds
create_summary_statistics_for_sensors(path_sensor, sensor, time_period, dir_results)
time_end = time.time()
print("Time: " + str(time_end - time_begin))

# endregion

# region visualize summary statistics: mean (sensordata) x std (sensordata) x participant x activity for every sensor

def vis_summary_stats(path_summary_stats, sensor, time_period, activity_type, activity_name, list_activities,
                      dir_results):
    # load summary stats
    summary_stats = pd.read_csv(path_summary_stats)

    # create dataframe including only the relevant activities
    summary_stats = summary_stats[summary_stats[activity_type].isin(list_activities)]

    # rename device_id to participant
    summary_stats = summary_stats.rename(columns={"device_id": "participant"})
    # replace participant id with last three digits
    summary_stats["participant"] = summary_stats["participant"].str[-3:]

    # visualize four dimensional scatterplot with seaborn: mean x std x participant x activity
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))

    if activity_name == "gravity":
        # set x limits
        axs[0].set_xlim(-1, 1)
        axs[1].set_xlim(-1, 1)
        axs[2].set_xlim(-1, 1)
        # set y limits
        axs[0].set_ylim(0, 0.5)
        axs[1].set_ylim(0, 0.5)
        axs[2].set_ylim(0, 0.5)

    if activity_name == "gyroscope":
        # set x limits
        axs[0].set_xlim(-0.1, 0.1)
        axs[1].set_xlim(-0.1, 0.1)
        axs[2].set_xlim(-0.1, 0.1)
        # set y limits
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)
        axs[2].set_ylim(0, 1)

    if activity_name == "linear_accelerometer":
        # set x limits
        axs[0].set_xlim(-0.04, 0.04)
        axs[1].set_xlim(-0.04, 0.04)
        axs[2].set_xlim(-0.04, 0.04)
        # set y limits
        axs[0].set_ylim(0, 0.3)
        axs[1].set_ylim(0, 0.3)
        axs[2].set_ylim(0, 0.3)

    if activity_name == "accelerometer":
        # set x limits
        axs[0].set_xlim(-1, 1)
        axs[1].set_xlim(-1, 1)
        axs[2].set_xlim(-1, 1)
        # set y limits
        axs[0].set_ylim(0, 0.8)
        axs[1].set_ylim(0, 0.8)
        axs[2].set_ylim(0, 0.8)

    fig.suptitle(sensor + " " + activity_name + " " + str(time_period) + " seconds ", fontsize=36)
    sns.scatterplot(x="double_values_0_mean", y="double_values_0_std", hue=activity_type, style="participant",
                    data=summary_stats, ax=axs[0])
    sns.scatterplot(x="double_values_1_mean", y="double_values_1_std", hue=activity_type, style="participant",
                    data=summary_stats, ax=axs[1])
    sns.scatterplot(x="double_values_2_mean", y="double_values_2_std", hue=activity_type, style="participant",
                    data=summary_stats, ax=axs[2])

    # plt.show()

    return fig

# body positions
list_activities = ["walking", "running", "cycling", "sitting: at a table", "standing", "lying"]
activity_type = "ESM_bodyposition"
activity_name = "bodyposition"
time_periods = [5, 10, 20, 40]  # seconds
sensors = ["accelerometer", "linear_accelerometer", "gyroscope", "gravity"]
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/visualizations/" + activity_name + "/"
for sensor in sensors:
    for time_period in time_periods:
        path_summary_stats = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/summary_stats/" + sensor + "_summary_stats_" + str(
            time_period) + ".csv"
        fig = vis_summary_stats(path_summary_stats, sensor, time_period, activity_type, activity_name,
                                list_activities, dir_results)
        fig.savefig(dir_results + "/summary_stats/" + activity_name + " _summarystats_" + sensor + "_" + str(
            time_period) + ".png")

# (public) transport
list_activities = ["on the way: in public transport", "on the way: in train", "on the way: standing",
                   "on the way: walking\\/cycling"]
activity_type = "ESM_location"
activity_name = "public transport"
time_periods = [5, 10, 20, 40]  # seconds
sensors = ["accelerometer", "linear_accelerometer", "gyroscope", "gravity"]
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/visualizations/" + activity_name + "/"
for sensor in sensors:
    for time_period in time_periods:
        path_summary_stats = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/summary_stats/" + sensor + "_summary_stats_" + str(
            time_period) + ".csv"
        fig = vis_summary_stats(path_summary_stats, sensor, time_period, activity_type, activity_name,
                                list_activities, dir_results)
        fig.savefig(dir_results + "/summary_stats/" + activity_name + " _summarystats_" + sensor + "_" + str(
            time_period) + ".png")

# on the toilet
list_activities = ["on the toilet", "working: at computer\\/laptop"]
activity_type = "ESM_activity"
activity_name = "on the toilet"
time_periods = [5, 10, 20, 40]  # seconds
sensors = ["accelerometer", "linear_accelerometer", "gyroscope", "gravity"]
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/visualizations/" + activity_name + "/"
for sensor in sensors:
    for time_period in time_periods:
        path_summary_stats = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/summary_stats/" + sensor + "_summary_stats_" + str(
            time_period) + ".csv"
        fig = vis_summary_stats(path_summary_stats, sensor, time_period, activity_type, activity_name,
                                list_activities, dir_results)
        fig.savefig(dir_results + "summary_stats/" + activity_name + " _summarystats_" + sensor + "_" + str(
            time_period) + ".png")

# endregion

# region visualize time of events (ESM) x activity for every participant

path_esm_final = "/Volumes/INTENSO/In Usage new/Databases/esm_all_transformed.csv"
activity_type = "activity"
# list_activities = ["lying in bed after sleeping ", "lying in bed before sleeping "]
list_activities = ["lying in bed before sleeping", "lying in bed after sleeping"]
activity_description = "sleeping_times"
dir_results = "/Users/benediktjordan/Documents/MTS/Iteration01/Results/visualizations/sleeping_times"

def vis_time_of_events(path_esm_final, activity_type, list_activities, activity_description, dir_results):
    # load esm final
    esm_final = pd.read_csv(path_esm_final)

    # create dataframe including only the relevant activities
    # esm_final = esm_final[esm_final[activity_type].str.contains("|".join(list_activities), na=False)]
    esm_final = esm_final[esm_final[activity_type].isin(list_activities)]

    # rename device_id to participant
    esm_final = esm_final.rename(columns={"device_id": "participant"})
    # replace participant id with last three digits
    esm_final["participant"] = esm_final["participant"].str[-3:]

    # create column containing the hour of the event
    esm_final["timestamp_hour"] = pd.to_datetime(esm_final['timestamp'], unit='ms').astype("string").str[11:16]
    # esm_final["timestamp_hour"]  = datetime.strptime(esm_final["timestamp_hour"][i], '%H:%M')
    esm_final = esm_final.reset_index(drop=True)
    esm_final["timestamp_hour_decimal"] = 0
    for i in range(0, len(esm_final)):
        esm_final["timestamp_hour_decimal"][i] = datetime.strptime(esm_final["timestamp_hour"][i], '%H:%M')
        esm_final["timestamp_hour_decimal"][i] = (datetime.combine(date.min, esm_final["timestamp_hour_decimal"][
            i].time()) - datetime.min).total_seconds() / 3600

    # visualize time of events x participant x activity
    import matplotlib.dates as mdates
    # create plot
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle("Time of events (ESM) x participant x " + activity_description, fontsize=18)
    sns.scatterplot(x="timestamp_hour_decimal", y="participant", style=activity_type, data=esm_final, ax=axs)
    axs.set_xlim(0, 24)
    axs.set_xticks(range(1, 24))
    plt.show()
    # save figure
    fig.savefig(dir_results + "/" + activity_description + "_time_of_events_" + ".png")

vis_time_of_events(path_esm_final, activity_type, list_activities, activity_description, dir_results)
# endregion