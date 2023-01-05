# user and sensor database
import pandas as pd
import os

# user database matching Names to IDs to newIDs; last column is iteration number
users = pd.DataFrame([["Simone_1","61bf23e5-0a6b-4d3c-b393-1a23d4f64e88", 1, 1],
                      ["Simone_2","4c4e5063-1b23-4dfc-886d-c6a202225ed6", 1, 1],
                      ["Simone_3","53c73807-3756-4303-b4dc-5e7e232e528c", 1, 1],
                      ["Simone_4", "f83ed117-9279-4ef8-ab74-83d7b8b268b8", 1, 1],
                      ["Tina", "0d620b8a-c2d4-48fc-9c75-80ce80aeea3e", 2, 1],
                      ["Tina_2", "3936a8f9-8be0-4523-bd0f-2a03943cb5f0", 2, 1],
                      ["Lea","590f0faf-d932-4a57-998d-e3da667a91dc", 3, 1],
                      ["Lotte","410df445-af3d-4cf7-8bf1-f92160f1c41f", 4, 1],
                      ["Bella","ad064def-9d72-44ff-96b2-f3b3d90d1079", 5, 1],
                      ["Madleine","36cf6eb2-9d88-46db-a90c-3f24cb4b7228", 6, 1],
                      ["Miranda","cf2dfa9b-596b-4a72-a4da-cb3777a31cc7", 7, 1],
                      ["Lena", "212a5ebe-0714-47ac-a887-964c24e0ae43", 8, 1],
                      ["Paul", "15cdbd7c-132e-4bb8-9dab-192cf909daec", 9, 1],
                      ["Selcuk", "2294d0b0-67ef-4af2-8ffb-69db607920c9", 10, 1],
                      ["Selcuk_2", "c838b909-f782-4441-aa3c-10c6c7765ba3", 10, 1],
                      ["Rosa", "84afe4cb-3572-46bc-bc29-d982ac375341", 11, 1],
                      ["Rosa_2", "e6c1d093-148e-47f8-8054-6663dc5c366a", 11, 1],
                      ["Bini", "6388b5d9-367b-427e-a2f5-912014c69a5e", 12, 1],
                      ["Tanzi", "e9d3ed5e-1d52-445c-82ac-8bbe8066b3d7", 13, 1],
                      ["Pauli", "6ab9716e-e6d8-4492-ad86-f051a9a4b62a", 14, 1],
                      ["Margherita ?", "b23b3f4e-7fc1-452f-be16-b9388451f3f6", 15, 1],
                      ["Margherita_2", "25f1657f-5a39-4dba-8a3b-e6efbfec0e4d", 15, 1],
                      ["Unknown", "3936a8f9-8be0-4523-bd0f-2a03943cb5f0", 18, 1],
                      ["Felix", "b7b013b7-f78c-4325-a7ab-2dfc128fba27", 16, 1],
                      ["Benedikt_1", "8206b501-20e7-4851-8b79-601af9e4485f", 17, 1],
                      ["Benedikt_2", "f1dc0e69-a548-4771-85f9-28db9060d4c6", 17, 1],
                      ["Benedikt_3", "eab5ef78-8eb2-4587-801f-834fc1f86f31", 17, 1],
                      ["Benedikt_4", "f1db4f27-2e1c-4558-85a1-b43ef2c5af59", 17, 1],
                      ["Benedikt_5 (vorher Unknown_2)", "57fc9641-9f4d-409b-bffd-f333b01c33c9", 17, 1],
                      ["Benedikt_secondiPhone", "4c1db32b-48fc-4fa6-a4fe-c44f079b7ca4", 17, 1],
                      ["Benedikt_tablet","0960f02f-8c67-486c-b8db-7850d4a7070b", 17, 1],
                      ["Benedikt_seconditeration", "ba683866-dfc3-47e0-a75a-61c07cf33505", 17, 2]],
                     columns = ["Name", "ID", "new_ID", "iteration"])

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

# test: check for different sensors
dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"
sensor = "sensor_wifi"
df_check = pd.read_csv(dir_sensorfiles + sensor + "_esm_timeperiod_5 min.csv_JSONconverted.csv", nrows=1000)
df_check.columns

#endregion

#region numeric sensors: define which sensors are numeric (important for features extraction)
sensors_numeric = ["accelerometer", "barometer", "gravity", "gyroscope", "linear_accelerometer", "locations", "magnetometer",
                   "plugin_ambient_noise", "plugin_openweather", "plugin_studentlife_audio", "rotation"]
#endregion