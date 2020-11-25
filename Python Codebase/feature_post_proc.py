from datetime import date
import os
import pandas as pd

statefile = {
    'NewYork': 'lime_actual_120_featureImportance',
    'NewJersey': 'lime_actual_210_featureImportance',
    'California': 'lime_actual_210_featureImportance',
    'Florida': 'lime_actual_120_featureImportance',
    'Texas': 'lime_actual_180_featureImportance',
    'Alabama': 'lime_actual_150_featureImportance',
    'Alaska': 'lime_actual_150_featureImportance',
    'Arizona': 'lime_actual_210_featureImportance',
    'Arkansas': 'lime_actual_150_featureImportance',
    'Colorado': 'lime_actual_210_featureImportance',
    'Connecticut': 'lime_actual_150_featureImportance',
    'Delaware': 'lime_actual_210_featureImportance',
    'Georgia': 'lime_actual_180_featureImportance',
    'Hawaii': 'lime_actual_180_featureImportance',
    'Idaho': 'lime_actual_120_featureImportance',
    'Illinois': 'lime_actual_120_featureImportance',
    'Indiana': 'lime_actual_120_featureImportance',
    'Iowa': 'lime_actual_120_featureImportance',
    'Kansas': 'lime_actual_210_featureImportance',
    'Kentucky': 'lime_actual_180_featureImportance',
    'Louisiana': 'lime_actual_120_featureImportance',
    'Maine': 'lime_actual_150_featureImportance',
    'Maryland': 'lime_actual_120_featureImportance',
    'Massachusetts': 'lime_actual_120_featureImportance',
    'Michigan': 'lime_actual_150_featureImportance',
    'Minnesota': 'lime_actual_150_featureImportance',
    'Mississippi': 'lime_actual_210_featureImportance',
    'Missouri': 'lime_actual_120_featureImportance',
    'Montana': 'lime_actual_120_featureImportance',
    'Nebraska': 'lime_actual_150_featureImportance',
    'Nevada': 'lime_actual_120_featureImportance',
    'NewHampshire': 'lime_actual_210_featureImportance',
    'NewMexico': 'lime_actual_180_featureImportance',
    'NorthCarolina': 'lime_actual_210_featureImportance',
    'NorthDakota': 'lime_actual_120_featureImportance',
    'Ohio': 'lime_actual_120_featureImportance',
    'Oklahoma': 'lime_actual_120_featureImportance',
    'Oregon': 'lime_actual_180_featureImportance',
    'Pennsylvania': 'lime_actual_120_featureImportance',
    'RhodeIsland': 'lime_actual_120_featureImportance',
    'SouthCarolina': 'lime_actual_120_featureImportance',
    'SouthDakota': 'lime_actual_180_featureImportance',
    'Tennessee': 'lime_actual_120_featureImportance',
    'Utah': 'lime_actual_180_featureImportance',
    'Vermont': 'lime_actual_180_featureImportance',
    'Virginia': 'lime_actual_120_featureImportance',
    'Washington': 'lime_actual_120_featureImportance',
    'WestVirginia': 'lime_actual_150_featureImportance',
    'Wisconsin': 'lime_actual_120_featureImportance',
    'Wyoming': 'lime_actual_120_featureImportance'
}


# this function splits the comparator conditions from the features
def get_feature_split(x):
    s = x
    word_list = s.split()
    pattern = ["<=", ">=", "<", ">", "="]
    if set(word_list) & set(pattern):
        lisst = list(set(word_list) & set(pattern))
        for item in lisst:
            if len(lisst) == 1:
                word_list[word_list.index(item):] = ["".join(word_list[word_list.index(item):])]
            else:
                if word_list.index(item) == 1:
                    word_list[:word_list.index(item) + 1] = ["".join(word_list[:word_list.index(item) + 1])]
                else:
                    word_list[word_list.index(item):] = ["".join(word_list[word_list.index(item):])]
    if len(word_list) == 2:
        word_list.insert(0, "")
    return word_list


# this function splits the lag conditions from the features
def get_lag_split(x):
    pattern = "_Lag"
    if pattern in x:
        x = x.rstrip("d_Lag")
        word_list = x.rsplit('_', 1)
    else:
        word_list = [x, 0]
    return word_list


def extract_feature_imp(statefile):
    feature_importance = pd.DataFrame()
    for state, fileName in statefile.items():
        file_df = pd.read_csv(os.path.join("features", state + '_' + fileName + ".csv"))
        file_df = file_df[file_df["state"] == state]
        file_df = file_df[~file_df.column.str.contains("Visited", na=False)]
        if "Unnamed: 0" in file_df.columns:
            file_df.drop(["Unnamed: 0"], inplace=True, axis=1)
        file_df["scaledInfluence"] = (file_df["totalInfluence"] / file_df["absTotal"].max()) * 100
        file_df.sort_values("absTotal", ascending=False, inplace=True)
        temp = file_df["column"].apply(get_feature_split)
        file_df[["condition1", "columnName", "condition2"]] = pd.DataFrame.from_dict(
            dict(zip(temp.index, temp.values))).T
        del temp
        file_df = file_df[["state", "column", "totalInfluence", "absTotal", "scaledInfluence", "columnName",
                           "condition1", "condition2"]]
        temp = file_df["columnName"].apply(get_lag_split)
        file_df[["featureName", "effectsLagDays"]] = pd.DataFrame.from_dict(dict(zip(temp.index, temp.values))).T
        del temp
        file_df["featureValue"] = file_df["condition1"] + " & " + file_df["condition2"]
        file_df["featureValue"] = file_df["featureValue"].str.strip("& ")
        file_df.drop(["columnName", "condition1", "condition2"], inplace=True, axis=1)
        file_df["simuFlag"] = 0
        file_df["version"] = fileName.rstrip("_featureImportance")
        file_df["group"] = "Not controllable"

        file_df.loc[file_df.column.str.contains("OxCGRT_Policy", na=False), 'group'] = "Policymaker controllable"
        file_df.loc[file_df.column.str.contains("Tested", na=False), 'group'] = "Policymaker controllable"

        if feature_importance.empty:
            feature_importance = file_df.copy()
        else:
            feature_importance = feature_importance.append(file_df)
    return feature_importance


# this function checks the direction of policy and tested vars based on different back testing window
def check_feature_imp(statefile):
    for state, file_name in statefile.items():
        print("*_" + state)
        file_name = "lime_actual_"
        for ver in [120, 150, 180, 210]:
            df = pd.read_csv(file_name + str(ver) + "_featureImportance.csv")
            df = df[df["state"] == state]
            df = df[~df.column.str.contains("Visited", na=False)]
            df_policy = df[df['column'].str.contains("OxCGRT")]
            df_tested = df[df['column'].str.contains("_Tested_")]
            print(" " + str(ver) + "-Policy:" + str(round(df_policy["totalInfluence"].mean(), 2))
                  + " Tested " + str(round(df_tested["totalInfluence"].mean(), 2)))


feature_importance = extract_feature_imp(statefile)
feature_importance.loc[(feature_importance['column'].str.contains("OxCGRT"))
                       & (feature_importance['scaledInfluence'].abs() > 2), "simuFlag"] = 1
feature_importance.loc[(feature_importance['column'].str.contains("Tested"))
                       & (feature_importance['scaledInfluence'].abs() > 2), "simuFlag"] = 1

today = date.today()
f_name = "featureImportance_final_" + today.strftime("%Y-%m-%d") + ".csv"
feature_importance.to_csv(os.path.join("features", f_name))
