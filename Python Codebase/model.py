import os
from datetime import date
from math import sqrt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular

# the file is scheduled for fortnightly model refresh/retraining
today = date.today()
scheduled_date = today.strftime("%d")
datem = today.strftime("%Y-%m")

# load persisted data for training
masterdata = pd.read_csv(os.path.join("data", "MasterData_" + datem + "-" + scheduled_date + "_final.csv"))

# load model version
v_dict = load(os.path.join("data", "config.joblib"))
model_version = "v" + str(v_dict['version'] + 1)

# model explainability parameters
rank_dict = {"lime": ["actual", "weekNr", "default"], "permute": ["noWeights"]}
datasize_list = [120, 150, 180, 210]

# subset of metrics used from Masterdata
list_vars = [
    "date",
    "state",
    "JHU_ConfirmedCases",
    "CDS_Tested",
    "CovidTrackingProject_ConfirmedCases",
    "AverageDailyTemperature",
    "AverageDewPoint",
    "AverageRelativeHumidity",
    "AverageSurfaceAirPressure",
    "AverageWindSpeed",
    "AverageWindGustSpeed",
    "OxCGRT_Policy_C1_SchoolClosing",
    "OxCGRT_Policy_C2_WorkplaceClosing",
    "OxCGRT_Policy_C3_CancelPublicEvents",
    "OxCGRT_Policy_C4_RestrictionsOnGatherings",
    "OxCGRT_Policy_C5_ClosePublicTransport",
    "OxCGRT_Policy_C6_StayAtHomeRequirements",
    "OxCGRT_Policy_C7_RestrictionsOnInternalMovement",
    "OxCGRT_Policy_C8_InternationalTravelControls",
    "OxCGRT_Policy_H1_PublicInformationCampaigns",
    "OxCGRT_Policy_H2_TestingPolicy",
    "OxCGRT_Policy_H3_ContactTracing",
    "weekNr",
    "dayOfWeek",
    "Visited_Alabama",
    "Visited_Alaska",
    "Visited_Arizona",
    "Visited_Arkansas",
    "Visited_California",
    "Visited_Colorado",
    "Visited_Connecticut",
    "Visited_Delaware",
    "Visited_DistrictofColumbia",
    "Visited_Florida",
    "Visited_Georgia",
    "Visited_Hawaii",
    "Visited_Idaho",
    "Visited_Illinois",
    "Visited_Indiana",
    "Visited_Iowa",
    "Visited_Kansas",
    "Visited_Kentucky",
    "Visited_Louisiana",
    "Visited_Maine",
    "Visited_Maryland",
    "Visited_Massachusetts",
    "Visited_Michigan",
    "Visited_Minnesota",
    "Visited_Mississippi",
    "Visited_Missouri",
    "Visited_Montana",
    "Visited_Nebraska",
    "Visited_Nevada",
    "Visited_NewHampshire",
    "Visited_NewJersey",
    "Visited_NewMexico",
    "Visited_NewYork",
    "Visited_NorthCarolina",
    "Visited_NorthDakota",
    "Visited_Ohio",
    "Visited_Oklahoma",
    "Visited_Oregon",
    "Visited_Pennsylvania",
    "Visited_RhodeIsland",
    "Visited_SouthCarolina",
    "Visited_SouthDakota",
    "Visited_Tennessee",
    "Visited_Texas",
    "Visited_Utah",
    "Visited_Vermont",
    "Visited_Virginia",
    "Visited_Washington",
    "Visited_WestVirginia",
    "Visited_Wisconsin",
    "Visited_Wyoming"
]

# target variables for economic models
eco_target_vars = ["OIET_WomplyRevenue_RevenueAll"]

# metrics for which daily delta is calculated
calc_dailynew_metrics = [
    "JHU_ConfirmedCases",
    "CDS_Tested",
    "CovidTrackingProject_ConfirmedCases"]

# metrics for which lag values are calculated
calc_lag_metrics = [
    "AverageDailyTemperature",
    "AverageDewPoint",
    "AverageRelativeHumidity",
    "AverageSurfaceAirPressure",
    "AverageWindSpeed",
    "AverageWindGustSpeed",
    "OxCGRT_Policy_C1_SchoolClosing",
    "OxCGRT_Policy_C2_WorkplaceClosing",
    "OxCGRT_Policy_C3_CancelPublicEvents",
    "OxCGRT_Policy_C4_RestrictionsOnGatherings",
    "OxCGRT_Policy_C5_ClosePublicTransport",
    "OxCGRT_Policy_C6_StayAtHomeRequirements",
    "OxCGRT_Policy_C7_RestrictionsOnInternalMovement",
    "OxCGRT_Policy_C8_InternationalTravelControls",
    "OxCGRT_Policy_H1_PublicInformationCampaigns",
    "OxCGRT_Policy_H2_TestingPolicy",
    "OxCGRT_Policy_H3_ContactTracing"]

# Used for lex data processing
visited_full_list = [
    "Visited_Alabama",
    "Visited_Alaska",
    "Visited_Arizona",
    "Visited_Arkansas",
    "Visited_California",
    "Visited_Colorado",
    "Visited_Connecticut",
    "Visited_Delaware",
    "Visited_DistrictofColumbia",
    "Visited_Florida",
    "Visited_Georgia",
    "Visited_Hawaii",
    "Visited_Idaho",
    "Visited_Illinois",
    "Visited_Indiana",
    "Visited_Iowa",
    "Visited_Kansas",
    "Visited_Kentucky",
    "Visited_Louisiana",
    "Visited_Maine",
    "Visited_Maryland",
    "Visited_Massachusetts",
    "Visited_Michigan",
    "Visited_Minnesota",
    "Visited_Mississippi",
    "Visited_Missouri",
    "Visited_Montana",
    "Visited_Nebraska",
    "Visited_Nevada",
    "Visited_NewHampshire",
    "Visited_NewJersey",
    "Visited_NewMexico",
    "Visited_NewYork",
    "Visited_NorthCarolina",
    "Visited_NorthDakota",
    "Visited_Ohio",
    "Visited_Oklahoma",
    "Visited_Oregon",
    "Visited_Pennsylvania",
    "Visited_RhodeIsland",
    "Visited_SouthCarolina",
    "Visited_SouthDakota",
    "Visited_Tennessee",
    "Visited_Texas",
    "Visited_Utah",
    "Visited_Vermont",
    "Visited_Virginia",
    "Visited_Washington",
    "Visited_WestVirginia",
    "Visited_Wisconsin",
    "Visited_Wyoming"
    ]

# list of states
states = ["NewYork",
          "California",
          "Florida",
          "Texas",
          "NewJersey",
          "Illinois",
          "Nevada",
          "Mississippi",
          "Alabama",
          "Arizona",
          "Arkansas",
          "Colorado",
          "Connecticut",
          "Delaware",
          "Georgia",
          "Hawaii",
          "Idaho",
          "Indiana",
          "Iowa",
          "Kansas",
          "Kentucky",
          "Louisiana",
          "Maine",
          "Maryland",
          "Massachusetts",
          "Michigan",
          "Minnesota",
          "Missouri",
          "Montana",
          "Nebraska",
          "NewHampshire",
          "NewMexico",
          "NorthCarolina",
          "NorthDakota",
          "Ohio",
          "Oklahoma",
          "Oregon",
          "Pennsylvania",
          "RhodeIsland",
          "SouthCarolina",
          "SouthDakota",
          "Tennessee",
          "Utah",
          "Vermont",
          "Virginia",
          "Washington",
          "WestVirginia",
          "Wisconsin",
          "Wyoming"]


# Generates lags for metrics
def lag_variables(lag, data, metrics):
    for col in data[metrics]:
        for lagval in lag:
            data.loc[:, col+"_"+str(lagval)+"d_Lag"] = data[col].shift(lagval)
    return data


def norm_variables(my_data, metrics, method):
    for col in metrics:
        if method == "min-max":
            scaler = MinMaxScaler()
            my_data[col + "_minMax"] = scaler.fit_transform(my_data[col].values.reshape(-1, 1))
    return my_data


# this function gives column-wise count of missing values with and % of missing values
def check_missing_value(data):
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns,
                                     'missing_count': data.isnull().sum(),
                                     'percent_missing': percent_missing})
    return (missing_value_df.loc[missing_value_df["missing_count"] != 0]).sort_values("missing_count")


# get empty and same valued cols
def get_empty_same_valued_cols(dataset):
    empty_list, single_value_list = [], []
    for col in list(dataset):
        if dataset[col].nunique() == 0:
            empty_list.append(col)
        if dataset[col].nunique() == 1:
            single_value_list.append(col)
    return empty_list, single_value_list


# function gives instance-based explanation
def get_interpretation_by_instance(instance_id, explainer, data, feature_names, model_obj):
    """
    get index value from id or date
    define the explainer
    pass id to get explanation
    """
    i = instance_id
    exp = explainer.explain_instance(data.loc[i, :].values, model_obj.predict, num_features=len(feature_names))
    result = pd.DataFrame(exp.as_list(), columns=["column", "value"])
    result.sort_values("column", inplace=True, ascending=True)
    return result


# averaging out all the observation output from LIME for each observation
def get_global_interpretation(data, model_obj, actual_predicted, week_nr, weight_option):
    index_list = data.index.tolist()
    feature_names = data.columns.tolist()
    explainer = lime.lime_tabular.LimeTabularExplainer(data[feature_names].astype(int).values,
                                                       mode='regression',
                                                       feature_names=feature_names,
                                                       class_names=["JHU_ConfirmedCases_dailyNewCases"])
    total_influence = pd.DataFrame()
    actual = None
    itr = 0

    for i in index_list:
        result = get_interpretation_by_instance(instance_id=i,
                                                explainer=explainer,
                                                data=data,
                                                feature_names=feature_names,
                                                model_obj=model_obj)

        # weekNr | actual --> input
        if weight_option == "actual":
            actual = (actual_predicted.loc[i, ]["actual"]) / actual_predicted["actual"].sum()

        elif weight_option == "weekNr":
            actual = (week_nr.loc[i, ]["weekNr"]) / 100

        elif weight_option == "default":
            actual = 1

        if itr == 0:
            total_influence = result.copy()
            # persist feature importance
            total_influence["value"] = total_influence["value"] * actual
            total_influence.rename(columns={"value": i}, inplace=True)
        else:
            total_influence[i] = result["value"] * actual
        itr = itr + 1

    return total_influence


def data_preprocessing(master_data, state, list_vars, eco_target_vars, calc_dailynew_metrics, visited_full_list):
    # subset data
    mdl_input = master_data[list_vars + eco_target_vars]
    mdl_input["date"] = pd.to_datetime(mdl_input["date"])
    mdl_input = mdl_input.groupby('state', sort=True).apply(pd.DataFrame.sort_values, 'date').reset_index(drop=True)
    mdl_input = mdl_input[mdl_input["state"] == state]
    visited = visited_full_list

    # treating outliers from CDS_Tested if present
    zero_val_indx = mdl_input[mdl_input["CDS_Tested"] == 0]["CDS_Tested"].tail(10).index.tolist()
    if len(zero_val_indx) != 0:
        mdl_input.loc[zero_val_indx, "CDS_Tested"] = np.nan
        mdl_input["CDS_Tested"] = ((mdl_input["CDS_Tested"].ffill() + mdl_input["CDS_Tested"].bfill()) / 2).ffill()

    # compute dailynew
    for metric in calc_dailynew_metrics:
        mdl_input[metric + "_dailyNewCases"] = mdl_input.groupby("state")[metric].diff()
    mdl_input = mdl_input.drop(columns=calc_dailynew_metrics)

    # checking for negative values in dailyNew imputed by CDS
    try:
        if (mdl_input["JHU_ConfirmedCases_dailyNewCases"] <= 0).any():
            mdl_input.loc[(mdl_input["JHU_ConfirmedCases_dailyNewCases"] <= 0) & (mdl_input[
                        "CovidTrackingProject_ConfirmedCases_dailyNewCases"] > 0), "JHU_ConfirmedCases_dailyNewCases"] \
                = mdl_input[mdl_input["JHU_ConfirmedCases_dailyNewCases"] <= 0][
                "CovidTrackingProject_ConfirmedCases_dailyNewCases"]
    except:
        if (mdl_input["JHU_ConfirmedCases_dailyNewCases"] <= 0).any():
            mdl_input.loc[(mdl_input["JHU_ConfirmedCases_dailyNewCases"] <= 0) & (mdl_input[
                "CovidTrackingProject_ConfirmedCases_dailyNewCases"] > 0), "JHU_ConfirmedCases_dailyNewCases"] = \
                mdl_input[mdl_input["JHU_ConfirmedCases_dailyNewCases"] < 0][
                    "CovidTrackingProject_ConfirmedCases_dailyNewCases"]

    backtest = mdl_input[
        ["date", "JHU_ConfirmedCases_dailyNewCases", "CovidTrackingProject_ConfirmedCases_dailyNewCases"]]
    mdl_input.drop("CovidTrackingProject_ConfirmedCases_dailyNewCases", inplace=True, axis=1)

    # treating neg or zero from CDS_Tested_dailyNewCases if present
    zero_val_indx = mdl_input[mdl_input["CDS_Tested_dailyNewCases"] <= 0]["CDS_Tested_dailyNewCases"].tail(
        10).index.tolist()
    if len(zero_val_indx) != 0:
        mdl_input.loc[zero_val_indx, "CDS_Tested_dailyNewCases"] = np.nan
        mdl_input["CDS_Tested_dailyNewCases"] = ((mdl_input["CDS_Tested_dailyNewCases"].ffill() + mdl_input[
            "CDS_Tested_dailyNewCases"].bfill()) / 2).ffill()

    # replace infinite values by nan
    mdl_input = mdl_input.replace(np.inf, np.nan)
    mdl_input = mdl_input.replace(-np.inf, np.nan)

    # check missing values followed by imputation
    chck_misn_df = check_missing_value(mdl_input)
    cols = chck_misn_df[(chck_misn_df["missing_count"] == 1) &
                        (chck_misn_df["column_name"].str.endswith('_dailyNewCases'))]["column_name"].tolist()
    mdl_input[cols] = mdl_input[cols].bfill()
    cols = chck_misn_df[(chck_misn_df["column_name"].str.startswith('Visited_'))]["column_name"].tolist()
    mdl_input[cols] = ((mdl_input[cols].ffill() + mdl_input[cols].bfill()) / 2).ffill()
    calc_lag = calc_lag_metrics

    # check for same and empty valued col presence
    empty_list, single_value_list = get_empty_same_valued_cols(mdl_input)
    if len(empty_list) != 0:
        mdl_input.drop(empty_list, axis=1, inplace=True)
        calc_lag = list(set(calc_lag) - set(empty_list))
        visited = list(set(visited) - set(empty_list))
        eco_target_vars = list(set(eco_target_vars) - set(empty_list))

    if len(single_value_list) != 0:
        mdl_input.drop(single_value_list, axis=1, inplace=True)
        calc_lag = list(set(calc_lag) - set(single_value_list))
        visited = list(set(visited) - set(single_value_list))
        eco_target_vars = list(set(eco_target_vars) - set(single_value_list))

    # calculating lag values
    mdl_input = lag_variables([14, 21, 28, 35], mdl_input, calc_lag)
    mdl_input = mdl_input.drop(columns=calc_lag)

    # treating lex variables
    visited_to_take = mdl_input[visited].mean().sort_values(ascending=False)
    visited_to_take = visited_to_take.where(visited_to_take >= 0.01).dropna()
    visited_to_take = visited_to_take.where(visited_to_take <= 0.90).dropna().index.values.tolist()
    visted_to_drop = list(set(visited) - set(visited_to_take))

    # dropping not req visited columns
    mdl_input = mdl_input.drop(columns=visted_to_drop)

    # impute missing values for lex vars
    for lex_var in visited_to_take:
        mdl_input[lex_var] = ((mdl_input[lex_var].ffill() + mdl_input[lex_var].bfill()) / 2).ffill()

    # get lags for lex vars
    mdl_input = lag_variables([14, 21, 28, 35], mdl_input, visited_to_take)
    mdl_input = mdl_input.drop(columns=visited_to_take)

    # get lags for JHU_ConfirmedCases_dailyNewCases
    mdl_input = lag_variables([1, 7, 14, 21, 28], mdl_input, ["JHU_ConfirmedCases_dailyNewCases"])
    mdl_input = mdl_input.fillna(0)

    # removing records with neg case
    mdl_input = mdl_input[mdl_input["JHU_ConfirmedCases_dailyNewCases"] >= 0]
    mdl_input = mdl_input[mdl_input["CDS_Tested_dailyNewCases"] >= 0]

    # create input data for eco models and clean the main mdl input
    eco_input = mdl_input
    mdl_input = mdl_input.drop(columns=eco_target_vars)
    mdl_input = mdl_input.drop(columns=['JHU_ConfirmedCases_dailyNewCases_7d_Lag',
                                        'JHU_ConfirmedCases_dailyNewCases_14d_Lag',
                                        'JHU_ConfirmedCases_dailyNewCases_21d_Lag',
                                        'JHU_ConfirmedCases_dailyNewCases_28d_Lag'])

    # truncate until earliest lag date
    mdl_input = mdl_input[(mdl_input['date'] >= '2020-02-01')]
    return mdl_input, backtest, eco_input


# backtest the results
def backtesting(pipeline, backtest, oot_size, ds_x, ds_y, state):
    acc_metric_rmse = {}
    acc_metric_r2 = {}

    if oot_size != 0:
        predictions = pipeline.predict(ds_x)
        pred = pd.DataFrame(data=predictions, index=ds_x.index, columns=["predictions"])

        r2 = r2_score(predictions, ds_y)
        rmse = sqrt(mean_squared_error(predictions, ds_y))
        acc_metric_r2[state] = r2
        acc_metric_rmse[state] = rmse

        pred["split_flag"] = 0
        pred.tail(oot_size)["split_flag"] = 1
        backtest['index'] = backtest.index.astype(int)
        pred['index'] = pred.index.astype(int)
        display_back_test = pd.merge(backtest, pred, on="index")
        display_back_test.drop("index", inplace=True, axis=1)
        display_back_test['state'] = state
    return display_back_test, acc_metric_r2, acc_metric_rmse


# train economic impact models
def economic_factor_models(oot_size, eco_target_vars, eco_input, state, model_version):
    eco_feature_dictionary = {}
    eco_model_dictionary = {}

    if oot_size == 0:
        eco_models = []
        for eco_target in eco_target_vars:
            eco_dup_drop = eco_input.copy()
            eco_dup_drop.drop_duplicates(subset=eco_target, keep="first", inplace=True)

            learner = LinearRegression()
            eco_y = eco_dup_drop[eco_target]
            eco_x = eco_dup_drop.drop(columns=eco_target_vars)
            eco_x = eco_x.drop(columns=["date"])
            eco_x = eco_x[['JHU_ConfirmedCases_dailyNewCases_14d_Lag',
                           'JHU_ConfirmedCases_dailyNewCases_21d_Lag',
                           'JHU_ConfirmedCases_dailyNewCases_28d_Lag',
                           'OxCGRT_Policy_C6_StayAtHomeRequirements_28d_Lag']]

            eco_learner = Pipeline([('step1', StandardScaler()), ('step2', learner)])

            # training the  model
            eco_learner.fit(eco_x, eco_y)
            eco_models.append(state + '_' + model_version + "_" + eco_target + '.joblib')
            dump(eco_learner, os.path.join('models', state + '_' + model_version + "_" + eco_target + '.joblib'))
        eco_feature_dictionary[state] = eco_x.columns
        eco_model_dictionary[state] = eco_models
    return eco_feature_dictionary, eco_model_dictionary


def get_feature_importance(rank_method, weight_option, state, datasize, ds_x, ds_y, predictions, pipeline):
    if rank_method == "permute":
        rank_x = ds_x.tail(datasize).values
        rank_y = ds_y.tail(datasize).values.flatten()
        rank = permutation_importance(pipeline, rank_x, rank_y, n_repeats=30, random_state=0)
        feature_influence = pd.DataFrame()
        for i in rank.importances_mean.argsort()[::-1]:
            feature_influence = feature_influence.append({'state': state, "feature": ds_x.columns[i],
                                                         "permute_importance": rank.importances_mean[i]},
                                                         ignore_index=True)

    elif rank_method == "lime":
        dat_for_imp = ds_x
        actual_predicted = pd.DataFrame(data=predictions, index=ds_x.index, columns=["predicted"])
        actual_predicted["actual"] = ds_y
        if weight_option == "actual":
            tot_influ = get_global_interpretation(data=dat_for_imp.tail(datasize),
                                                  model_obj=pipeline,
                                                  actual_predicted=actual_predicted,
                                                  week_nr=None,
                                                  weight_option=weight_option)
        elif weight_option == "weekNr":
            week_nr = pd.DataFrame()
            week_nr["weekNr"] = dat_for_imp["weekNr"]
            tot_influ = get_global_interpretation(data=dat_for_imp.tail(datasize),
                                                  model_obj=pipeline,
                                                  actual_predicted=None,
                                                  week_nr=week_nr,
                                                  weight_option="weekNr")
        elif weight_option == "default":
            tot_influ = get_global_interpretation(data=dat_for_imp.tail(datasize),
                                                  model_obj=pipeline,
                                                  actual_predicted=None,
                                                  week_nr=None,
                                                  weight_option="default")

        feature_influence = pd.DataFrame(index=tot_influ.index)
        feature_influence["state"] = state
        feature_influence["column"] = tot_influ["column"]
        feature_influence["negInfluence"] = tot_influ.iloc[:, 1:].where(tot_influ.iloc[:, 1:] < 0).sum(axis=1)
        feature_influence["posInfluence"] = tot_influ.iloc[:, 1:].where(tot_influ.iloc[:, 1:] >= 0).sum(axis=1)
        feature_influence["totalInfluence"] = (feature_influence["negInfluence"] + feature_influence[
            "posInfluence"]) / datasize
        feature_influence["absTotal"] = abs(feature_influence["totalInfluence"])
        feature_influence.drop(["negInfluence", "posInfluence"], inplace=True, axis=1)
        feature_influence.sort_values("absTotal", ascending=False, inplace=True)
        rank_method + "_" + weight_option + "_" + str(datasize) + "_" + "featureImportance.csv"
        feat_file_name = state + "_" + rank_method + "_" + weight_option + "_" + str(
           datasize) + "_" + "featureImportance.csv"
        feature_influence.to_csv(os.path.join('models', feat_file_name))

    return feature_influence


def model_building(mdl_input, backtest, oot_size, state, model_version, rank_method, weight_option, datasize):
    feature_dictionary = {}

    mdl_input.drop('date', axis=1, inplace=True)
    rows_cnt = mdl_input["JHU_ConfirmedCases_dailyNewCases"].count()
    df_train = mdl_input.head(rows_cnt - oot_size)

    numeric_features = mdl_input.columns.to_list()
    numeric_features.remove("JHU_ConfirmedCases_dailyNewCases")

    x_train = df_train[numeric_features]
    y_train = df_train[["JHU_ConfirmedCases_dailyNewCases"]]
    ds_x = mdl_input[numeric_features]
    ds_y = mdl_input[["JHU_ConfirmedCases_dailyNewCases"]]

    feature_dictionary[state] = numeric_features
    pipeline.fit(x_train, y_train.values.flatten())

    # - voting regressor not available error thrown on C3
    # - Not able to autotune due to numpy version issue
    # - Ticket raised

    feature_influence = None
    display_back_test = None
    acc_metric_r2 = None
    acc_metric_rmse = None
    if oot_size != 0:
        display_back_test, acc_metric_r2, acc_metric_rmse = backtesting(pipeline=pipeline,
                                                                        backtest=backtest,
                                                                        oot_size=oot_size,
                                                                        ds_x=ds_x,
                                                                        ds_y=ds_y,
                                                                        state=state)
    elif oot_size == 0:
        predictions = pipeline.predict(ds_x)
        if rank_method is not None:
            feature_influence = get_feature_importance(rank_method=rank_method,
                                                       weight_option=weight_option,
                                                       state=state,
                                                       datasize=datasize,
                                                       ds_x=ds_x,
                                                       ds_y=ds_y,
                                                       predictions=predictions,
                                                       pipeline=pipeline)

        # - Upsert the model (switched from C3 way to a pkl dump)
        # - Not able to retrieve a C3 pipeline model due to error of module 'numpy.core'
        #   has no attribute '_multiarray_umath'
        # - Ticket raised
        dump(pipeline,  os.path.join('models', state + '_' + model_version + '.joblib'))

    return display_back_test, feature_influence, feature_dictionary, acc_metric_r2, acc_metric_rmse


def execute_all(oot_size, model_version, rank_method, weight_option, states, datasize, masterdata, list_vars,
                calc_dailynew_metrics, visited_full_list, eco_target_vars):
    # instantiation
    df_back_test_all = pd.DataFrame()
    feature_importance = pd.DataFrame()
    acc_metric_r2 = {}
    acc_metric_rmse = {}
    feature_dictionary = {}
    eco_feature_dictionary = {}
    eco_model_dictionary = {}

    # looping all states
    for state in states:
        # preprocessing
        mdl_input, back_test, eco_input = data_preprocessing(master_data=masterdata.copy(),
                                                             state=state,
                                                             list_vars=list_vars,
                                                             eco_target_vars=eco_target_vars,
                                                             calc_dailynew_metrics=calc_dailynew_metrics,
                                                             visited_full_list=visited_full_list)

        # whole data training or backtesting
        display_back_test, feature_influence, feat_dict_temp, acc_metric_r2_temp, acc_metric_rmse_temp = model_building(
            mdl_input=mdl_input,
            backtest=back_test,
            oot_size=oot_size,
            state=state,
            model_version=model_version,
            rank_method=rank_method,
            weight_option=weight_option,
            datasize=datasize)

        # appending all feature influence
        if rank_method is not None:
            feature_importance = feature_importance.append(feature_influence)

        # appending all backTested results
        if oot_size != 0:
            df_back_test_all = df_back_test_all.append(display_back_test)
            acc_metric_r2.update(acc_metric_r2_temp)
            acc_metric_rmse.update(acc_metric_rmse_temp)

        # appending feature dictionary
        feature_dictionary.update(feat_dict_temp)

        # economic modelling
        eco_feature_dict_temp, eco_model_dict_temp = economic_factor_models(oot_size=oot_size,
                                                                            eco_target_vars=eco_target_vars,
                                                                            eco_input=eco_input,
                                                                            state=state,
                                                                            model_version=model_version)

        # update all dictionary values
        eco_feature_dictionary.update(eco_feature_dict_temp)
        eco_model_dictionary.update(eco_model_dict_temp)

    # Persisting dataframes, dictionaries and model objects
        if rank_method is not None:
            feature_file_name = state + "_" + rank_method + "_" + weight_option + "_" + str(datasize) \
                                + "_" + "featureImportance.csv"
            feature_importance.to_csv(os.path.join('models', feature_file_name))

    if oot_size != 0:
        back_test_file = "backTest_" + str(oot_size) + "days_on_" + datem + "-" + scheduled_date + ".csv"
        df_back_test_all.to_csv(os.path.join('models', back_test_file))

    if oot_size == 0:
        dump(feature_dictionary,  os.path.join('models', 'feature_dictionary_' + model_version + '.joblib'))
        dump(eco_feature_dictionary,  os.path.join('models', 'eco_feature_dictionary' + model_version + '.joblib'))
        dump(eco_model_dictionary,  os.path.join('models', 'eco_model_dictionary' + model_version + '.joblib'))

# - Not able to create an ensemble model on the C3 platform. Using Vanilla scikit-learn for now
# - Ticket raised


""""
The c3 code for ensemble model of Voting Regressor throws an error - 
C3RuntimeException: 500 - NotClassified - c3.love.util.OsUtil_err2 [2423.541268]
message: "Error executing command: /usr/local/share/c3/condaEnvs/synechron/dev5/py-sklearn_1_0_0/bin/python 
                                    /tmp/pythonActionSourceCache1951794049330979434/SklearnPipe_isTrainable.py
File "/tmp/pythonActionSourceCache1951794049330979434/SklearnPipe_isTrainable.py", line 1618, in str_to_moduleitem
    return reduce(getattr, input_str.split("."), module)
AttributeError: module 'sklearn.ensemble' has no attribute 'VotingRegressor'.
"""
# # Define the SklearnPipe for the StandardScaler
# standard_scaler = c3.SklearnPipe(
#                     name="standardScaler",
#                     technique=c3.SklearnTechnique(
#                         name="preprocessing.StandardScaler",
#                         processingFunctionName="transform"))
#
# # Define the models
# rf = RandomForestRegressor(n_estimators=80, max_depth=5,max_features="sqrt")
# gbm = GradientBoostingRegressor(n_estimators=100, max_depth=5,learning_rate=0.03)
# lr = LinearRegression()
#
# # Define SklearnPipe from the models
# rf_c3pipe = c3.SklearnPipe.fromEstimator(rf)
# gbm_c3pipe = c3.SklearnPipe.fromEstimator(gbm)
# lr_c3pipe = c3.SklearnPipe.fromEstimator(lr)
#
# # Create the ensembler
# ensembler = c3.SklearnPipe(
#                     name="VotingRegressor",
#                     technique=c3.SklearnTechnique(
#                         name="ensemble.VotingRegressor",
#                         processingFunctionName="predict",
#                         hyperParameters={'estimators':[("m1",rf_c3pipe),("m2",gbm_c3pipe),("m3",lr_c3pipe)]}))
#
# # Define the whole pipeline
# pipeline = c3.MLSerialPipeline(
#                         name="pipeline",
#                         steps=[c3.MLStep(name="step1", pipe=standard_scaler),
#                               c3.MLStep(name="step2", pipe=ensembler)])

# instantiate vanilla scikit-learn model
rf = RandomForestRegressor(n_estimators=80, max_depth=5, max_features="sqrt")
gbm = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.03)
lr = LinearRegression()
ensembler = VotingRegressor([("m1", rf), ("m2", gbm), ("m3", lr)])
pipeline = Pipeline([('step1', StandardScaler()), ('step2', ensembler)])

# model execution with/without Feature Importance
# define backtesting period. 0 indicates the entire dataset is used for training
oot_size = 0

# persist TS_name data
ts_plot = masterdata[list_vars + eco_target_vars]
ts_name = "TS_" + datem + "-" + scheduled_date + ".csv"
ts_plot.to_csv(os.path.join('models', ts_name))

# uncomment for running Feature Importance metrics. To be enhanced later
# for rank_method, weights in rank_dict.items():
#     for weight_option in weights:
#         for datasize in datasize_list:
#             execute_all(oot_size, model_version, rank_method, weight_option,
#                         states, datasize, masterdata, list_vars, calc_dailynew_metrics,
#                         visited_full_list, eco_target_vars)

# execute core function for training without Feature Importance calculations
rank_method = None
weight_option = None
datasize = 0

execute_all(oot_size, model_version, rank_method, weight_option, states, datasize, masterdata, list_vars,
            calc_dailynew_metrics, visited_full_list, eco_target_vars)

# backtesting
# define backtesting period
oot_size = 14

# backtesting models not used for determining feature importance
rank_method = None
weight_option = None
datasize = 0

# execute core function for backtesting
execute_all(oot_size, model_version, rank_method, weight_option, states, datasize, masterdata, list_vars,
            calc_dailynew_metrics, visited_full_list, eco_target_vars)

# update model version in config file
v_dict['version'] += 1
dump(v_dict, os.path.join("data", "config.joblib"))
