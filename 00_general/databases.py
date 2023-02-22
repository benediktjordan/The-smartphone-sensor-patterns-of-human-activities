# user and sensor database
import numpy as np
import pandas as pd
import os

#region activity database: matching user answers to activity classes
# region human motion

human_motion = {
    "sitting: at a table (in one hand)": pd.DataFrame(
        data={"bodyposition": ["sitting: at a table", "sitting: at a table"],
                "smartphonelocation": ["In my hand", "in one hand"]}),
    "sitting: at a table (in two hands)": pd.DataFrame(
        data={"bodyposition": ["sitting: at a table"],
                "smartphonelocation": ["in two hands"]}),
    "sitting: at a table (on flat surface)": pd.DataFrame(
        data={"bodyposition": ["sitting: at a table"],
                "smartphonelocation": ["on a flat surface in front of me (i.e. table, bed, etc.)"]}),

    "sitting: on the couch (in one hand)": pd.DataFrame(
        data={"bodyposition": ["sitting: on the couch", "sitting: on the couch"],
              "smartphonelocation": ["In my hand", "in one hand"]}),
    "sitting: on the couch (in two hands)": pd.DataFrame(
        data={"bodyposition": ["sitting: on the couch"],
              "smartphonelocation": ["in two hands"]}),

    "sitting: somewhere else (in one hand)": pd.DataFrame(
        data={"bodyposition": ["sitting: somewhere else", "sitting: somewhere else"],
              "smartphonelocation": ["In my hand", "in one hand"]}),
    "sitting: somewhere else (in two hands)": pd.DataFrame(
        data={"bodyposition": ["sitting: somewhere else"],
              "smartphonelocation": ["in two hands"]}),

    "standing (in one hand)": pd.DataFrame(
        data={"bodyposition": ["standing", "standing"],
                "smartphonelocation": ["in one hand", "In my hand"]}),
    "standing (in two hands)": pd.DataFrame(
        data={"bodyposition": ["standing"],
                "smartphonelocation": ["in two hands"]}),
    "standing (on flat surface)": pd.DataFrame(
        data={"bodyposition": ["standing"],
                "smartphonelocation": ["on a flat surface in front of me (i.e. table, bed, etc.)"]}),

    "lying (in one hand)": pd.DataFrame(
        data={"bodyposition": ["lying", "lying"],
                "smartphonelocation": ["in one hand", "In my hand"]}),
    "lying (in two hands)": pd.DataFrame(
        data={"bodyposition": ["lying"],
                "smartphonelocation": ["in two hands"]}),
    "lying (on flat surface)": pd.DataFrame(
        data={"bodyposition": ["lying"],
                "smartphonelocation": ["on a flat surface in front of me (i.e. table, bed, etc.)"]}),

    "walking (in one hand)": pd.DataFrame(
        data={"bodyposition": ["walking", "walking"],
              "smartphonelocation": ["In my hand", "in one hand"],
              "location": ["on the way: walking\/cycling", "on the way: walking\/cycling"]}),
    "walking (in two hands)": pd.DataFrame(
        data={"bodyposition": ["walking"],
              "smartphonelocation": ["in two hands"],
              "location": ["on the way: walking\/cycling"]}),

    "running (in one hand)": pd.DataFrame(
        data={"bodyposition": ["running", "running"],
              "smartphonelocation": ["In my hand", "in one hand"]}),
    "running (in two hands)": pd.DataFrame(
        data={"bodyposition": ["running"],
              "smartphonelocation": ["in two hands"]}),

    "cycling (in one hand)": pd.DataFrame(
        data={"bodyposition": ["cycling", "cycling"],
              "smartphonelocation": ["In my hand", "in one hand"]}),
    "cycling (in two hands)": pd.DataFrame(
        data={"bodyposition": ["cycling"],
              "smartphonelocation": ["in two hands"]}),

    np.nan: pd.DataFrame(
        data={"location": ["on the way: in train", "on the way: other means of transport",
                           "on the way: in public transport", "on the way: in car\/bus\/train\/tram"]})

}


humanmotion_general = {
    "sitting: in car\/bus\/tram\/train": ["sitting"],
    "sitting: somewhere else": ["sitting"],
    "sitting: on the couch": ["sitting"],
    "sitting: at a table": ["sitting"],
    "sitting: somewhere else\",\"Other (please specify)": ["sitting"],
    "sitting: somewhere else\",\"Other: sitting on bed": ["sitting"],
    "sitting: on the couch\",\"sitting: somewhere else": ["sitting"],
    "sitting: at a table\",\"sitting: in car\/bus\/tram\/train": ["sitting"],

    "standing": ["standing"],
    "standing\",\"Other (please specify)": ["standing"],
    "standing\",\"Other: in kitchen cooking": ["standing"],
    "standing\",\"Other: in train standing": ["standing"],

    "walking": ["walking"],
    "Other: walking": ["walking"],

    "running": ["running"],
    "cycling": ["cycling"],
    "lying": ["lying"],
    "lying\",\"Other: auf der Shakti matte": ["lying"]
}

humanmotion_specific = {
    "lying": ["lying"],
    "lying\",\"Other: auf der Shakti matte": ["lying"],

    "sitting: at a table": ["sitting at a table"],
    "sitting: at a table\",\"sitting: in car\/bus\/tram\/train": ["sitting at a table"],

    "sitting: on the couch": ["sitting on the couch"],
    "sitting: on the couch\",\"sitting: somewhere else": ["sitting on the couch"],

    "sitting: in car\/bus\/tram\/train": ["sitting in car/bus/tram/train"],

    "sitting: somewhere else": ["sitting somewhere else"],

    "standing": ["standing"],
    "walking": ["walking"],
    "running": ["running"],
    "cycling": ["cycling"],

}
#endregion

# region public transport
#activity_list = df_esm["bodyposition"].value_counts()
#location_list = df_esm["location"].value_counts()

publictransport_non_motorized = {
    "sitting: at a table": ["stationary"],

    "walking": ["walking"],
    "Other: walking": ["walking"],

    "running": ["running"],

    "cycling": ["cycling"]

}

publictransport = {
    "on the way: in public transport": ["public transport"],

    "on the way: in train": ["train"],

    "on the way: in a car": ["car"],

    "on the way: in car\/bus\/train\/tram": ["car/bus/train/tram"],

    "at home": ["exclude"] # because in 15 cases, the person was "walking" "at home" which is not a good
    # representation of walking!
}



#endregion

#region before & after sleep
before_after_sleep = {
    "lying in bed after sleeping": ["lying in bed after sleeping"],

    "lying in bed before sleeping": ["lying in bed before sleeping"],

    "lying in bed at another time": ["lying in bed at other times"],

    "lying on couch": ["lying on the couch"]

}

before_after_sleep_updated = {
    "lying in bed after sleeping": pd.DataFrame(
        data={"activity": ["lying in bed after sleeping"]}),
    "lying in bed before sleeping": pd.DataFrame(
        data={"activity": ["lying in bed before sleeping"]}),
    "lying in bed at other times": pd.DataFrame(
        data={"activity": ["lying in bed at another time"]}),
    "lying on the couch": pd.DataFrame(
        data={"activity": ["lying on couch"]}),

    "not lying: stationary": pd.DataFrame(
        data={"bodyposition": ["sitting: at a table", "sitting: on the couch",
                               "sitting: somewhere else", "standing"]}),
    "not lying: dynamic": pd.DataFrame(
        data={"bodyposition": ["walking", "cycling"]}),

    np.nan: pd.DataFrame(
        data={"bodyposition": ["sitting: on the couch", "sitting: somewhere else", "sitting: somewhere else",
                               "sitting: somewhere else\",\"Other: sitting on bed", "Other: sitting in bed",
                               "sitting: on the couch", "sitting: somewhere else",
                               "Other: sitting on bed"],
              "activity": ["lying on couch", "lying in bed after sleeping", "lying in bed at another time",
                           "lying in bed before sleeping", "lying in bed before sleeping",
                           "lying in bed before sleeping", "lying in bed before sleeping",
                           "lying in bed before sleeping"]})
}

#endregion

#region on the toilet
on_the_toilet_sittingsomewhereelse = {
    "sitting: on the couch": ["sitting (not on the toilet)"],
    "sitting: at a table": ["sitting (not on the toilet)"],
}

on_the_toilet = {
    "on the toilet": ["on the toilet"]
}
#endregion

#region location
#df_esm["activity"].value_counts()
location = {
    "in the office": ["in the office"],
    "at another workplace": ["at another workplace"],

    "in home-office": ["at home"],
    "at home": ["at home"],

    "at a friends place": ["at another place"],
    "Other: boyfriend's home": ["at another place"],
    "Other: at boyfriend's home": ["at another place"],
    "Other: Boyfriend's home": ["at another place"],
    "Other: at boyfriends home": ["at another place"],

    "in restaurant": ["at another place"],

    "shopping": ["at another place"],

    "with friends outside": ["at another place"],

    "on the way: in train": ["on the way"],
    "on the way: walking/\cycling": ["on the way"],
    "on the way: in car": ["on the way"],
    "on the way: standing": ["on the way"],
    "on the way: in public transport": ["on the way"]
}
#endregion

# region smartphonelocation
#df_esm["smartphonelocation"].value_counts()
smartphonelocation = {
    "in one hand": ["in one hand"],
    "in two hands": ["in two hands"],
    "on a flat surface in front of me (i.e. table, bed, etc.)": ["on flat surface"],
    "in smartphone mount (i.e. on bicycle, in car, etc.)" : ["in smartphone mount"]
}
#endregion

# region aligned
#df_esm["aligned"].value_counts()
aligned = {
    "Aligned with my intentions": ["aligned"],
    "Not aligned with my intentions": ["not aligned"]
}
#endregion

#endregion


#region user database matching Names to IDs to newIDs; last column is iteration number
users = pd.DataFrame([["Simone_1","61bf23e5-0a6b-4d3c-b393-1a23d4f64e88", 1, 1, "Leipzig", "Europe/Berlin"],
                      ["Simone_2","4c4e5063-1b23-4dfc-886d-c6a202225ed6", 1, 1,"Leipzig", "Europe/Berlin"],
                      ["Simone_3","53c73807-3756-4303-b4dc-5e7e232e528c", 1, 1, "Leipzig", "Europe/Berlin"],
                      ["Simone_4", "f83ed117-9279-4ef8-ab74-83d7b8b268b8", 1, 1, "Leipzig", "Europe/Berlin"],
                      ["Tina", "0d620b8a-c2d4-48fc-9c75-80ce80aeea3e", 2, 1, "Berlin", "Europe/Berlin"],
                      ["Tina_2", "3936a8f9-8be0-4523-bd0f-2a03943cb5f0", 2, 1, "Berlin", "Europe/Berlin"],
                      ["Lea","590f0faf-d932-4a57-998d-e3da667a91dc", 3, 1, "Maastricht", "Europe/Berlin"],
                      ["Lotte","410df445-af3d-4cf7-8bf1-f92160f1c41f", 4, 1, "Berlin", "Europe/Berlin"],
                      ["Bella","ad064def-9d72-44ff-96b2-f3b3d90d1079", 5, 1, "Rotterdam", "Europe/Berlin"],
                      ["Madleine","36cf6eb2-9d88-46db-a90c-3f24cb4b7228", 6, 1, "Windsor", "Europe/London"],
                      ["Miranda","cf2dfa9b-596b-4a72-a4da-cb3777a31cc7", 7, 1, "Innsbruck", "Europe/Berlin"],
                      ["Lena", "212a5ebe-0714-47ac-a887-964c24e0ae43", 8, 1, "Leipzig", "Europe/Berlin"],
                      ["Paul", "15cdbd7c-132e-4bb8-9dab-192cf909daec", 9, 1, "Kampala", "Africa/Kampala"],
                      ["Selcuk", "2294d0b0-67ef-4af2-8ffb-69db607920c9", 10, 1, "Berlin", "Europe/Berlin"],
                      ["Selcuk_2", "c838b909-f782-4441-aa3c-10c6c7765ba3", 10, 1, "Berlin", "Europe/Berlin"],
                      ["Rosa_2", "e6c1d093-148e-47f8-8054-6663dc5c366a", 11, 1, "Leipzig", "Europe/Berlin"],
                      ["Bini", "6388b5d9-367b-427e-a2f5-912014c69a5e", 12, 1, "Witten", "Europe/Berlin"],
                      ["Bini_2", "84afe4cb-3572-46bc-bc29-d982ac375341", 12, 1, "Witten", "Europe/Berlin"],
                      ["Tanzi", "e9d3ed5e-1d52-445c-82ac-8bbe8066b3d7", 13, 1, "Berlin", "Europe/Berlin"],
                      ["Pauli", "6ab9716e-e6d8-4492-ad86-f051a9a4b62a", 14, 1, "Leipzig", "Europe/Berlin"],
                      ["Margherita ? (office is in Factory)", "b23b3f4e-7fc1-452f-be16-b9388451f3f6", 15, 1, "Berlin", "Europe/Berlin"],
                      ["Margherita_2", "25f1657f-5a39-4dba-8a3b-e6efbfec0e4d", 19, 1, "Turin", "Europe/Berlin"],
                      ["Unknown", "3936a8f9-8be0-4523-bd0f-2a03943cb5f0", 18, 1, "Berlin", "Europe/Berlin"],
                      ["Felix", "b7b013b7-f78c-4325-a7ab-2dfc128fba27", 16, 1, "Leipzig", "Europe/Berlin"],
                      ["Benedikt_1", "8206b501-20e7-4851-8b79-601af9e4485f", 17, 1, "Leipzig", "Europe/Berlin"],
                      ["Benedikt_2", "f1dc0e69-a548-4771-85f9-28db9060d4c6", 17, 1, "Leipzig", "Europe/Berlin"],
                      ["Benedikt_3", "eab5ef78-8eb2-4587-801f-834fc1f86f31", 17, 1, "Leipzig", "Europe/Berlin"],
                      ["Benedikt_4", "f1db4f27-2e1c-4558-85a1-b43ef2c5af59", 17, 1, "Leipzig", "Europe/Berlin"],
                      ["Benedikt_5 (vorher Unknown_2)", "57fc9641-9f4d-409b-bffd-f333b01c33c9", 17, 1, "Leipzig","Europe/Berlin"],
                      ["Benedikt_secondiPhone", "4c1db32b-48fc-4fa6-a4fe-c44f079b7ca4", 17, 1, "Leipzig", "Europe/Berlin"],
                      ["Benedikt_tablet","0960f02f-8c67-486c-b8db-7850d4a7070b", 17, 1, "Leipzig", "Europe/Berlin"],
                      ["Benedikt_seconditeration", "ba683866-dfc3-47e0-a75a-61c07cf33505", 17, 2, "Leipzig", "Europe/Berlin"]],
                     columns = ["Name", "ID", "new_ID", "Iteration", "City", "Timezone"])
#endregion

#region sensors and frequencies
sensors_and_frequencies = pd.DataFrame([["barometer",0.1],
                                        ["accelerometer",10],
                                          ["battery",0], #event based
                                          ["battery_charges",0], #event based
                                          ["battery_discharges",0], #event based
                                          ["bluetooth", 0.0333333],
                                          ["esm", 0.00055555555], #12 times a day while every time I get 4 events
                                          ["gravity", 10],
                                          ["gyroscope", 10],
                                          ["ios_status_monitor", 0.01666666666],
                                          ["linear_accelerometer", 10],
                                          ["locations", 1],
                                          ["magnetometer", 10],
                                          ["network", 0], #event based
                                          ["plugin_ambient_noise", 0.01666666666], #once per minute
                                          ["plugin_device_usage", 0], #event based
                                          ["plugin_ios_activity_recognition", 0.1],
                                          ["plugin_ios_esm", 0.00055555555], #12 times a day while every time I get 4 events
                                          ["plugin_openweather", 0.00055555555], #every 30 minutes
                                          ["plugin_studentlife_audio", 0.00055555555], #every 4 minutes
                                          ["push_notification", 0.00013888888], #18 times a day
                                          ["screen", 0], #event based
                                          ["rotation", 10],
                                          ["sensor_wifi", 1],
                                          ["significant_motion", 0],
                                          ["timezone", 0.00027777777] #once per hour
                                          ], columns = ["Sensor", "Frequency (in Hz)"])
#endregion

#region sensor columns: create dictionary with sensor names and sensor columns
sensor_columns = {}
sensor_columns["accelerometer"] = ["acc_double_values_0", "acc_double_values_1", "acc_double_values_2"]
sensor_columns["barometer"] = ["bar_double_values_0"]
sensor_columns["battery"] = ["bat_battery_level", "bat_battery_scale", "bat_battery_status", "bat_battery_adaptor",
                             "bat_battery_voltage", "bat_battery_temperature"]
sensor_columns["battery_charges"] = ["bat_battery_end", "bat_battery_start", "bat_double_end_timestamp"]
sensor_columns["battery_discharges"] = ["bat_battery_end", "bat_battery_start", "bat_double_end_timestamp"]
sensor_columns["bluetooth"] = ['blu_label', 'blu_bt_name', 'blu_bt_rssi', 'blu_bt_address']
sensor_columns["gravity"] = ['gra_double_values_0', 'gra_double_values_1', 'gra_double_values_2']
sensor_columns["gyroscope"] = ['gyr_double_values_0', 'gyr_double_values_1', 'gyr_double_values_2']
sensor_columns["ios_status_monitor"] = ['ios_tz', 'ios_info', 'ios_trigger', 'ios_datetime']
sensor_columns["linear_accelerometer"] = ['lin_double_values_0', 'lin_double_values_1', 'lin_double_values_2']
sensor_columns["locations"] = ["double_speed", "double_bearing", "double_altitude", "double_latitude", "double_longitude"]
sensor_columns["magnetometer"] = ['mag_double_values_0', 'mag_double_values_1', 'mag_double_values_2']
sensor_columns["network"] = ['net_network_type', 'net_network_state', 'net_network_subtype']
sensor_columns["plugin_ambient_noise"] = ['plu_is_silent', 'plu_double_rms', 'plu_double_decibels', 'plu_double_frequency',
                                          'plu_double_silent_threshold']
sensor_columns["plugin_device_usage"] = ['plu_elapsed_device_on', 'plu_elapsed_device_off']
sensor_columns["plugin_ios_activity_recognition"] = ['plu_label', 'plu_cycling', 'plu_running', 'plu_unknown', 'plu_walking',
                                                     'plu_device_id', 'plu_activities', 'plu_automotive',
                                                     'plu_confidence', 'plu_stationary']
sensor_columns["plugin_openweather"] = ['plu_city', 'plu_rain', 'plu_snow', 'plu_unit', 'plu_sunset', 'plu_sunrise',
                                        'plu_humidity', 'plu_pressure', 'plu_cloudiness', 'plu_wind_speed', 'plu_temperature',
                                        'plu_wind_degrees', 'plu_temperature_max', 'plu_temperature_min', 'plu_weather_icon_id',
                                        'plu_weather_description']
sensor_columns["plugin_studentlife_audio"] = ['plu_datatype', 'plu_inference', 'plu_double_energy', 'plu_double_convo_end',
                                              'plu_double_convo_start']
sensor_columns["push_notification"] = ["pus_token"]
sensor_columns["rotation"] = ['rot_double_values_0', 'rot_double_values_1', 'rot_double_values_2', 'rot_double_values_3']
sensor_columns["screen"] = ["scr_screen_status"]
sensor_columns["sensor_wifi"] = ['sen_ssid', 'sen_bssid']
sensor_columns["significant_motion"] = ["sig_is_moving"]
sensor_columns["timezone"] = ["tim_timezone"]
#endregion

#region numeric sensors: define which sensors are numeric (important for features extraction)
sensors_numeric = ["accelerometer", "barometer", "gravity", "gyroscope", "linear_accelerometer", "locations", "magnetometer",
                   "plugin_ambient_noise", "plugin_openweather", "plugin_studentlife_audio", "rotation"]
#endregion





# test: check for different sensors
dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"
sensor = "sensor_wifi"
df_check = pd.read_csv(dir_sensorfiles + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.csv", nrows=1000)
df_check.columns

#endregion

