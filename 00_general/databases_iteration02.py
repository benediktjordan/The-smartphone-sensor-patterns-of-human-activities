#region import
import pickle

import pandas as pd

#endregion

#region users
users_iteration02 = pd.DataFrame([["Benedikt","", 17, 2],
                        ["Cordula","", 20, 2],
                        ["Simone","", 1, 2],
                        ["Selcuk","", 10, 2],
                        ["Tina","", 2, 2],
                        ["Tanzi","", 13, 2]
                      ],
                     columns = ["Name", "ID", "new_ID", "iteration"])


#endregion

#region labels
# create dictionary with start and end times of different activities; start_session and end_session should be datetime objects
dict_label_iteration02_Benedikt = {"01_standing": {"start_session": pd.Timestamp("2022-11-30 15:44:00"), "end_session": pd.Timestamp("2022-11-30 15:46:00")},
                "02_walking_mediumspeed": {"start_session": pd.Timestamp("2022-11-30 15:46:00"), "end_session": pd.Timestamp("2022-11-30 15:48:00")},
                "03_walking_lowspeed": {"start_session": pd.Timestamp("2022-11-30 15:48:00"), "end_session": pd.Timestamp("2022-11-30 15:50:00")},
              "04_sitting_onthecouch": {"start_session": pd.Timestamp("2022-11-30 15:50:00"), "end_session": pd.Timestamp("2022-11-30 15:52:00")},
                "05_walking_fastspeed": {"start_session": pd.Timestamp("2022-11-30 15:52:00"), "end_session": pd.Timestamp("2022-11-30 15:54:00")},
                "06_cycling": {"start_session": pd.Timestamp("2022-11-30 15:56:00"), "end_session": pd.Timestamp("2022-11-30 15:58:00")},
            "07_cycling_includingstops": {"start_session": pd.Timestamp("2022-11-30 15:58:00"), "end_session": pd.Timestamp("2022-11-30 16:01:00")},
                "08_running": {"start_session": pd.Timestamp("2022-11-30 16:03:00"), "end_session": pd.Timestamp("2022-11-30 16:05:00")},
                              "09_on the toilet": {"start_session": pd.Timestamp("2022-12-06 12:01:00"), "end_session": pd.Timestamp("2022-12-06 12:06:00")},
                                "10_lying_phoneinfront_onstomach": {"start_session": pd.Timestamp("2022-12-06 12:07:30"), "end_session": pd.Timestamp("2022-12-06 12:12:30")},
                                "11_lying_phoneoverhead": {"start_session": pd.Timestamp("2022-12-06 12:12:30"), "end_session": pd.Timestamp("2022-12-06 12:17:30")},
                                "12_lying_phoneonbed": {"start_session": pd.Timestamp("2022-12-06 12:17:30"), "end_session": pd.Timestamp("2022-12-06 12:22:30")},
                                "13_sitting_attable_phoneinhand": {"start_session": pd.Timestamp("2022-12-06 12:23:00"), "end_session": pd.Timestamp("2022-12-06 12:28:00")},
                                "14_sitting_attable_phoneontable": {"start_session": pd.Timestamp("2022-12-06 12:28:30"), "end_session": pd.Timestamp("2022-12-06 12:33:30")},
      "15_on the toilet": {
          "start_session": pd.Timestamp("2022-12-07 12:12:00"),
          "end_session": pd.Timestamp("2022-12-07 12:18:00")
      },
      "16_sitting_onthecouch": {
          "start_session": pd.Timestamp("2022-12-07 12:21:30"),
          "end_session": pd.Timestamp("2022-12-07 12:29:00")
      },
      "17_walking_mediumspeed": {
          "start_session": pd.Timestamp("2022-12-07 16:46:00"),
          "end_session": pd.Timestamp("2022-12-07 16:51:00")
      },
      "18_running": {
          "start_session": pd.Timestamp("2022-12-07 16:53:30"),
          "end_session": pd.Timestamp("2022-12-07 16:58:30")
      },
      "19_walking_mediumspeed": {
          "start_session": pd.Timestamp("2022-12-07 17:00:00"),
          "end_session": pd.Timestamp("2022-12-07 17:05:00")
      },
      "20_standing": {
          "start_session": pd.Timestamp("2022-12-07 17:06:00"),
          "end_session": pd.Timestamp("2022-12-07 17:12:00")
      },
      "21_walking_mediumspeed": {
          "start_session": pd.Timestamp("2022-12-07 17:13:00"),
          "end_session": pd.Timestamp("2022-12-07 17:17:30")
      },
      "22_on the toilet": {
          "start_session": pd.Timestamp("2022-12-07 17:23:00"),
          "end_session": pd.Timestamp("2022-12-07 17:28:00")
      },
      "23_sitting_onthecouch": {
          "start_session": pd.Timestamp("2022-12-07 17:32:00"),
          "end_session": pd.Timestamp("2022-12-07 17:37:00")
      },
      "24_lying_ontheside": {
          "start_session": pd.Timestamp("2022-12-07 17:38:30"),
          "end_session": pd.Timestamp("2022-12-07 17:43:30")
      },
      "25_standing": {
          "start_session": pd.Timestamp("2022-12-09 13:21:00"),
          "end_session": pd.Timestamp("2022-12-09 13:27:00")
      },
      "26_cycling": {
          "start_session": pd.Timestamp("2022-12-09 15:22:20"),
          "end_session": pd.Timestamp("2022-12-09 15:27:30")
      },
      "27_walking_lowspeed": {
          "start_session": pd.Timestamp("2022-12-09 15:30:00"),
          "end_session": pd.Timestamp("2022-12-09 15:34:00")
      },
      "28_walking": {
          "start_session": pd.Timestamp("2022-12-09 15:39:45"),
          "end_session": pd.Timestamp("2022-12-09 15:44:40")
      },
      "29_lying_phoneinfront_onstomach": {
          "start_session": pd.Timestamp("2022-12-09 17:10:00"),
          "end_session": pd.Timestamp("2022-12-09 17:13:00")
      }



                              }

dict_label_iteration02_Cordula = {
    "01_walking": {
        "start_session": pd.Timestamp("2022-12-27 17:53:30"),
        "end_session": pd.Timestamp("2022-12-27 17:58:30")
    },
    "02_standing": {
        "start_session": pd.Timestamp("2022-12-27 18:02:00"),
        "end_session": pd.Timestamp("2022-12-27 18:07:00")
    },
    "03_on the toilet": {
        "start_session": pd.Timestamp("2022-12-27 18:08:30"),
        "end_session": pd.Timestamp("2022-12-27 18:13:30")
    },
    "04_lying_phoneinfront_onback": {
        "start_session": pd.Timestamp("2022-12-27 18:15:30"),
        "end_session": pd.Timestamp("2022-12-27 18:20:30")
    },
    "05_sitting_attable_phoneinhand": {
        "start_session": pd.Timestamp("2022-12-27 18:22:30"),
        "end_session": pd.Timestamp("2022-12-27 18:28:30")
    }
}

dict_label_iteration02_Simone = {
    "06_cycling": {
        "start_session": pd.Timestamp("2022-12-07 16:46:00"),
        "end_session": pd.Timestamp("2022-12-07 16:51:00")
    },
    "07_running": {
        "start_session": pd.Timestamp("2022-12-07 16:53:30"),
        "end_session": pd.Timestamp("2022-12-07 16:58:30")
    },
    "08_walking": {
        "start_session": pd.Timestamp("2022-12-07 17:00:00"),
        "end_session": pd.Timestamp("2022-12-07 17:05:00")
    },
    "09_standing": {
        "start_session": pd.Timestamp("2022-12-07 17:06:00"),
        "end_session": pd.Timestamp("2022-12-07 17:12:00")
    },
    "10_sitting_on_the_toilet": {
        "start_session": pd.Timestamp("2022-12-07 17:23:00"),
        "end_session": pd.Timestamp("2022-12-07 17:28:00"),
    },
    "11_sitting_attable_phoneinhand": {
    "start_session": pd.Timestamp("2022-12-07 17:32:00"),
    "end_session": pd.Timestamp("2022-12-07 17:37:00")
    },
    "12_lying_ontheside": {
    "start_session": pd.Timestamp("2022-12-07 17:38:30"),
    "end_session": pd.Timestamp("2022-12-07 17:43:30")
    }
}

dict_label_iteration02_Selcuk = {
"01_sitting_attable_phoneinhand": {
"start_session": pd.Timestamp("2022-12-09 12:40:40"),
"end_session": pd.Timestamp("2022-12-09 12:45:30")
},
"02_standing": {
"start_session": pd.Timestamp("2022-12-09 12:46:00"),
"end_session": pd.Timestamp("2022-12-09 12:51:00")
},
"03_on_the_toilet": {
"start_session": pd.Timestamp("2022-12-09 12:52:00"),
"end_session": pd.Timestamp("2022-12-09 12:57:00")
},
"04_cycling": {
"start_session": pd.Timestamp("2022-12-09 13:09:30"),
"end_session": pd.Timestamp("2022-12-09 13:16:30")
},
"05_running": {
"start_session": pd.Timestamp("2022-12-09 13:18:30"),
"end_session": pd.Timestamp("2022-12-09 13:23:30")
},
"06_walking": {
"start_session": pd.Timestamp("2022-12-09 13:23:30"),
"end_session": pd.Timestamp("2022-12-09 13:29:15")
}
}

dict_label_iteration02_Tina = {
"01_sitting_attable_phoneinhand": {
"start_session": pd.Timestamp("2022-12-09 12:40:40"),
"end_session": pd.Timestamp("2022-12-09 12:45:30")
},
"02_standing": {
"start_session": pd.Timestamp("2022-12-09 12:46:00"),
"end_session": pd.Timestamp("2022-12-09 12:51:00")
},
"03_on_the_toilet": {
"start_session": pd.Timestamp("2022-12-09 12:52:00"),
"end_session": pd.Timestamp("2022-12-09 12:57:00")
},
"04_cycling": {
"start_session": pd.Timestamp("2022-12-09 13:09:30"),
"end_session": pd.Timestamp("2022-12-09 13:16:30")
},
"05_running": {
"start_session": pd.Timestamp("2022-12-09 13:18:30"),
"end_session": pd.Timestamp("2022-12-09 13:23:30")
},
"06_walking": {
"start_session": pd.Timestamp("2022-12-09 13:23:30"),
"end_session": pd.Timestamp("2022-12-09 13:29:15")
}
}

dict_label_iteration02_Tanzi = {
"01_cycling": {
"start_session": pd.Timestamp("2022-12-09 15:22:20"),
"end_session": pd.Timestamp("2022-12-09 15:27:30")
},
"02_running": {
"start_session": pd.Timestamp("2022-12-09 15:29:20"),
"end_session": pd.Timestamp("2022-12-09 15:35:50")
},
"03_walking": {
"start_session": pd.Timestamp("2022-12-09 15:39:45"),
"end_session": pd.Timestamp("2022-12-09 15:44:45")
},
"04_sitting_attable_phoneinhand": {
"start_session": pd.Timestamp("2022-12-09 16:15:15"),
"end_session": pd.Timestamp("2022-12-09 16:22:15")
},
"05_on_the_toilet": {
"start_session": pd.Timestamp("2022-12-09 16:23:15"),
"end_session": pd.Timestamp("2022-12-09 16:28:15")
},
"06_standing": {
"start_session": pd.Timestamp("2022-12-09 16:28:45"),
"end_session": pd.Timestamp("2022-12-09 16:34:30")
},
"07_lying_phoneinfront_onback": {
"start_session": pd.Timestamp("2022-12-09 17:08:30"),
"end_session": pd.Timestamp("2022-12-09 17:13:30")
}
}

#save all dictionaries as pickle files
path_save = "/Users/benediktjordan/Documents/MTS/Iteration02/datasets/"
#iterate through dataframe users_iteration02 using iterrows
for index, user in users_iteration02.iterrows():
    with open(path_save + user["Name"] + "/dict_label_iteration02_"+ user["Name"] +".pkl", "wb") as f:
        pickle.dump(eval("dict_label_iteration02_" + user["Name"]), f)

#endregion









