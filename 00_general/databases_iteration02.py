#region import
import pickle

import pandas as pd

#endregion

#region users
users_iteration02 = pd.DataFrame([["Benedikt","ba683866-dfc3-47e0-a75a-61c07cf33505", 17, 2, "Leipzig"],
                        ["Cordula", "ed78d3a5-3431-411c-9cb4-4455db51f2af", 20, 2, "Innsbruck"],
                        ["Simone", "dfeec08e-fcf8-4bab-b5dd-8bad5f4ab3e9", 1, 2, "Leipzig"],
                        ["Selcuk", "b350fe77-d2bd-49b6-8275-d933ba2b4608", 10, 2, "Berlin"],
                        ["Tina", "c754d352-dd80-45a3-a207-8c4e546e6e90", 2, 2, "Berlin"],
                        ["Tanzi","90ef3835-80fb-4c53-8ca6-ca993a2ce914", 13, 2, "Berlin"]
                      ],
                     columns = ["Name", "ID", "new_ID", "iteration", "City"])


#endregion

#region labels

#region create dictionary with start and end times of different activities; start_session and end_session should be datetime objects
dict_label_iteration02_Benedikt = {"01_standing": {"start_session": pd.Timestamp("2022-11-30 15:44:00"), "end_session": pd.Timestamp("2022-11-30 15:46:00")},
                "02_walking_mediumspeed": {"start_session": pd.Timestamp("2022-11-30 15:46:00"), "end_session": pd.Timestamp("2022-11-30 15:48:00")},
                "03_walking_lowspeed": {"start_session": pd.Timestamp("2022-11-30 15:48:00"), "end_session": pd.Timestamp("2022-11-30 15:50:00")},
              "04_sitting_onthecouch": {"start_session": pd.Timestamp("2022-11-30 15:50:00"), "end_session": pd.Timestamp("2022-11-30 15:52:00")},
                "05_walking_fastspeed": {"start_session": pd.Timestamp("2022-11-30 15:52:00"), "end_session": pd.Timestamp("2022-11-30 15:54:00")},
                "06_cycling": {"start_session": pd.Timestamp("2022-11-30 15:56:00"), "end_session": pd.Timestamp("2022-11-30 15:58:00")},
            "07_cycling_includingstops": {"start_session": pd.Timestamp("2022-11-30 15:58:00"), "end_session": pd.Timestamp("2022-11-30 16:01:00")},
                "08_running": {"start_session": pd.Timestamp("2022-11-30 16:03:00"), "end_session": pd.Timestamp("2022-11-30 16:05:00")},
                              "09_on_the_toilet": {"start_session": pd.Timestamp("2022-12-06 12:01:00"), "end_session": pd.Timestamp("2022-12-06 12:06:00")},
                                "10_lying_phoneinfront_onstomach": {"start_session": pd.Timestamp("2022-12-06 12:07:30"), "end_session": pd.Timestamp("2022-12-06 12:12:30")},
                                "11_lying_phoneoverhead": {"start_session": pd.Timestamp("2022-12-06 12:12:30"), "end_session": pd.Timestamp("2022-12-06 12:17:30")},
                                "12_lying_phoneonbed": {"start_session": pd.Timestamp("2022-12-06 12:17:30"), "end_session": pd.Timestamp("2022-12-06 12:22:30")},
                                "13_sitting_attable_phoneinhand": {"start_session": pd.Timestamp("2022-12-06 12:23:00"), "end_session": pd.Timestamp("2022-12-06 12:28:00")},
                                "14_sitting_attable_phoneontable": {"start_session": pd.Timestamp("2022-12-06 12:28:30"), "end_session": pd.Timestamp("2022-12-06 12:33:30")},
      "15_on_the_toilet": {
          "start_session": pd.Timestamp("2022-12-07 12:12:00"),
          "end_session": pd.Timestamp("2022-12-07 12:18:00")
      },
      "16_sitting_onthecouch": {
          "start_session": pd.Timestamp("2022-12-07 12:21:30"),
          "end_session": pd.Timestamp("2022-12-07 12:29:00")
      },
      "17_walking_mediumspeed": {
          "start_session": pd.Timestamp("2022-12-07 16:46:01"),
          "end_session": pd.Timestamp("2022-12-07 16:51:00")
      },
      "18_running": {
          "start_session": pd.Timestamp("2022-12-07 16:53:31"),
          "end_session": pd.Timestamp("2022-12-07 16:58:30")
      },
      "19_walking_mediumspeed": {
          "start_session": pd.Timestamp("2022-12-07 17:00:01"),
          "end_session": pd.Timestamp("2022-12-07 17:05:00")
      },
      "20_standing": {
          "start_session": pd.Timestamp("2022-12-07 17:06:01"),
          "end_session": pd.Timestamp("2022-12-07 17:12:00")
      },
      "21_walking_mediumspeed": {
          "start_session": pd.Timestamp("2022-12-07 17:13:00"),
          "end_session": pd.Timestamp("2022-12-07 17:17:30")
      },
      "22_on_the_toilet": {
          "start_session": pd.Timestamp("2022-12-07 17:23:01"),
          "end_session": pd.Timestamp("2022-12-07 17:28:00")
      },
      "23_sitting_onthecouch": {
          "start_session": pd.Timestamp("2022-12-07 17:32:01"),
          "end_session": pd.Timestamp("2022-12-07 17:37:00")
      },
      "24_lying_ontheside": {
          "start_session": pd.Timestamp("2022-12-07 17:38:31"),
          "end_session": pd.Timestamp("2022-12-07 17:43:30")
      },
      "25_standing": {
          "start_session": pd.Timestamp("2022-12-09 13:21:00"),
          "end_session": pd.Timestamp("2022-12-09 13:27:00")
      },
      "26_cycling": {
          "start_session": pd.Timestamp("2022-12-09 15:22:21"),
          "end_session": pd.Timestamp("2022-12-09 15:27:30")
      },
      "27_walking_lowspeed": {
          "start_session": pd.Timestamp("2022-12-09 15:30:01"),
          "end_session": pd.Timestamp("2022-12-09 15:34:00")
      },
      "28_walking": {
          "start_session": pd.Timestamp("2022-12-09 15:39:46"),
          "end_session": pd.Timestamp("2022-12-09 15:44:40")
      },
      "29_lying_phoneinfront_onstomach": {
          "start_session": pd.Timestamp("2022-12-09 17:10:01"),
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
    "03_on_the_toilet": {
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
    "10_on_the_toilet": {
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
"start_session": pd.Timestamp("2022-12-09 12:40:41"),
"end_session": pd.Timestamp("2022-12-09 12:45:30")
},
"02_standing": {
"start_session": pd.Timestamp("2022-12-09 12:46:01"),
"end_session": pd.Timestamp("2022-12-09 12:51:00")
},
"03_on_the_toilet": {
"start_session": pd.Timestamp("2022-12-09 12:52:01"),
"end_session": pd.Timestamp("2022-12-09 12:57:00")
},
"04_cycling": {
"start_session": pd.Timestamp("2022-12-09 13:09:31"),
"end_session": pd.Timestamp("2022-12-09 13:16:30")
},
"05_running": {
"start_session": pd.Timestamp("2022-12-09 13:18:31"),
"end_session": pd.Timestamp("2022-12-09 13:23:30")
},
"06_walking": {
"start_session": pd.Timestamp("2022-12-09 13:23:31"),
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

#region activity database: mapping user activities to activity classes
human_motion = {
    "sitting: at a table (in hand/s)": pd.DataFrame(
        data={"user activity": ["sitting_attable_phoneinhand"]}),
    "sitting: at a table (on flat surface)": pd.DataFrame(
        data={"user activity": ["sitting_attable_phoneontable"]}),

    "sitting: on the couch (in hand/s)": pd.DataFrame(
        data={"user activity": ["sitting_onthecouch"]}),

    "standing (in hand/s)": pd.DataFrame(
        data={"user activity": ["standing"]}),

    "lying (in hand/s)": pd.DataFrame(
        data={"user activity": ["lying_phoneinfront_onstomach", "lying_ontheside",
                                               "lying_phoneoverhead", "lying_phoneinfront_onback"]}),
    "lying (on flat surface)": pd.DataFrame(
        data={"user activity": ["lying_phoneonbed"]}),

    "walking (in hand/s)": pd.DataFrame(
        data={"user activity": ["walking_lowspeed", "walking_mediumspeed",
                                               "walking_fastspeed", "walking"]}),

    "running (in hand/s)": pd.DataFrame(
        data={"user activity": ["running"]}),

    "cycling (in hand/s)": pd.DataFrame(
        data={"user activity": ["cycling"]}),

}

on_the_toilet = {
    "on the toilet": pd.DataFrame(
        data={"user activity": ["on_the_toilet", "sitting_on_the_toilet", "on the toilet"]}),

    "sitting not on the toilet": pd.DataFrame(
        data={"user activity": ["sitting_attable_phoneinhand", "sitting_onthecouch"]}),
}

#endregion

#endregion











