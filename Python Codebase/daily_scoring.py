import os
from datetime import timedelta, date
import pandas as pd
import numpy as np
from joblib import load

today = date.today()
persisted_filename = "MasterData_"+today.strftime("%Y-%m-%d")+"_final.csv"
masterdata = pd.read_csv(os.path.join('data', persisted_filename))

state_list = ["Alabama",
              "Arizona",
              "Arkansas",
              "California",
              "Colorado",
              "Connecticut",
              "Delaware",
              "Florida",
              "Georgia",
              "Hawaii",
              "Idaho",
              "Illinois",
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
              "Mississippi",
              "Missouri",
              "Montana",
              "Nebraska",
              "Nevada",
              "NewHampshire",
              "NewJersey",
              "NewMexico",
              "NewYork",
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
              "Texas",
              "Utah",
              "Vermont",
              "Virginia",
              "Washington",
              "WestVirginia",
              "Wisconsin",
              "Wyoming"]

# global configuration and data structures
pred_window_size = 36
lag_vals = [14, 21, 28, 35]
no_simul_vars = 12
# daily new cases averaged over how many days
average_over_days = 5
# load model version
v_dict = load(os.path.join("data", "config.joblib"))
model_version = "v" + str(v_dict['version'])

# persist all scored information in this dictionary state-wise
state_calc_dict = {}

# lag metrics not in lex data
lag_metrics_novisit = ["AverageDailyTemperature",
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
# lex data lag metrics
visited = ["Visited_Alabama",
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
           "Visited_Wyoming"]

# combined lag metrics
lag_metrics = lag_metrics_novisit + visited

non_lag_metrics = ["weekNr", "dayOfWeek", "JHU_ConfirmedCases_dailyNewCases", "CDS_Tested_dailyNewCases"]

# metrics for which daily delta is calculated
calc_dailynew_metrics = ["JHU_ConfirmedCases", "CDS_Tested", "CovidTrackingProject_ConfirmedCases"]

# lag metrics inflated for each lag value
lag_metrics_expanded = []
for metric in lag_metrics:
    for val in lag_vals:
        lag_metrics_expanded.append(metric + "_" + str(val) + "d_Lag")

# column list for lagged datagrid used for scoring
list_metrics_expanded = non_lag_metrics + lag_metrics_expanded

# for iterating through metrics while creating the lagged datagrid
list_metrics = non_lag_metrics + lag_metrics

# row index list for lagged datagrid used for scoring
date_list = []
for forward in range(pred_window_size):
    date_list.append("day_" + str(forward))

# used by gen_jhu_lag_data
calc_jhu_dailynew_metrics = ["JHU_ConfirmedCases", "CovidTrackingProject_ConfirmedCases"]
jhu_lag_vals = [14, 21, 28]
jhu_lag_cols = ['JHU_ConfirmedCases_dailyNewCases_' + str(x) + 'd_Lag' for x in jhu_lag_vals]

# used by gen_pol_lag_data
policy_lag_vals = [28]
lag_eco_policies = ['OxCGRT_Policy_C6_StayAtHomeRequirements']
lag_eco_policies_cols = []
for lag in policy_lag_vals:
    for policy in lag_eco_policies:
        lag_eco_policies_cols.append(policy + '_' + str(lag) + 'd_Lag')

# Dictionary containing state-wise features
feature_dict = load(os.path.join('models', 'feature_dictionary_' + model_version + '.joblib'))

# Dictionary containing state-wise list of economic model filenames
eco_model_dict = load(os.path.join('models', 'eco_model_dictionary' + model_version + '.joblib'))

# Vaccine attributes
vaccine_dict = {'VaccineEfficacy': [0, 50, 80],
                'CoverageSpeed_DaysPer10pctPopulation': [5, 10]}

# Load scenario permutation dictionaries
policy_dict = load(os.path.join('data', 'policy_dict.joblib'))
policy_vac_dict = load(os.path.join('data', 'policy_vac_dict.joblib'))

# Load vaccine mulitiplier dictionary
vac_mul_dict = load(os.path.join('data', 'vac_mul_dict.joblib'))


# extracts statewise relevant data for gen_jhu_lag_data
def extract_hist_data(state):
    state_hist_data = masterdata.loc[masterdata['state'] == state].copy()

    # compute dailynew metric values
    for metric in calc_jhu_dailynew_metrics:
        state_hist_data[metric+"_dailyNewCases"] = state_hist_data.groupby("state")[metric].diff()

    state_hist_data = state_hist_data.drop(columns=calc_jhu_dailynew_metrics)

    # checking for negative values in dailyNew imputed by CDS
    if (state_hist_data["JHU_ConfirmedCases_dailyNewCases"] <= 0).any():
        state_hist_data.loc[(state_hist_data["JHU_ConfirmedCases_dailyNewCases"] <= 0) &
                            (state_hist_data["CovidTrackingProject_ConfirmedCases_dailyNewCases"] > 0),
                            "JHU_ConfirmedCases_dailyNewCases"] = state_hist_data[state_hist_data[
                                                                "JHU_ConfirmedCases_dailyNewCases"] <= 0][
                                                                    "CovidTrackingProject_ConfirmedCases_dailyNewCases"]
    state_hist_data = state_hist_data[['date', 'JHU_ConfirmedCases_dailyNewCases',
                                       'OxCGRT_Policy_C6_StayAtHomeRequirements']]
    return state_hist_data


# used for generating daily new cases lag data for economic policies
def gen_jhu_lag_data(state, datagrid_list):
    state_hist_data = extract_hist_data(state)
    eco_lag_list = []
    for dg in datagrid_list:
        simul_data = pd.DataFrame(columns=list(state_hist_data.columns))
        simul_data['date'] = (pd.date_range(start=today,
                                            periods=pred_window_size,
                                            freq='1D'))
        simul_data['date'] = simul_data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        simul_data['JHU_ConfirmedCases_dailyNewCases'] = dg
        combined_data = state_hist_data.append(simul_data)
        lag_df = pd.DataFrame(index=jhu_lag_cols, columns=date_list)
        for forward in range(pred_window_size):
            date_val = today + timedelta(days=forward)
            column_name = "day_" + str(forward)
            for lag in jhu_lag_vals:
                lag_date = date_val - timedelta(days=lag)
                lag_date_string = lag_date.strftime("%Y-%m-%d")
                try:
                    metric_name = "JHU_ConfirmedCases_dailyNewCases_"+str(lag)+"d_Lag"
                    lag_df.at[metric_name, column_name] = combined_data[combined_data['date'] == lag_date_string][
                                                                        'JHU_ConfirmedCases_dailyNewCases'].values[0]
                except:
                    pass
        lag_df = lag_df.ffill(axis=1)
        eco_lag_list.append(lag_df.T)
    return eco_lag_list


# used for generating lag data for economic policies. scenario_flag is false for real situation and true for simulations
def gen_pol_lag_data(state, datagrid_list, scenario_flag, dict_list):
    state_hist_data = extract_hist_data(state)
    eco_lag_list = []
    sim_var_list = list(dict_list[0].keys())
    lag_policy_list = []
    for policy in lag_eco_policies:
        if policy in sim_var_list:
            lag_policy_list.append(policy)
    idx_val = 0
    for dg in datagrid_list:
        lag_df = pd.DataFrame(index=lag_eco_policies_cols, columns=date_list)
        for forward in range(pred_window_size):
            date_val = today + timedelta(days=forward)
            column_name = "day_" + str(forward)
            for metric in lag_eco_policies:
                for lag in policy_lag_vals:
                    lag_date = date_val - timedelta(days=lag)
                    lag_date_string = lag_date.strftime("%Y-%m-%d")
                    try:
                        metric_name = metric+"_"+str(lag)+"d_Lag"
                        lag_df.at[metric_name, column_name] = state_hist_data[state_hist_data['date'] ==
                                                                              lag_date_string][metric].values[0]
                    except:
                        pass
        lag_df = lag_df.ffill(axis=1)
        if scenario_flag is True and len(lag_policy_list) > 0:
            for policy in lag_policy_list:
                for lag in policy_lag_vals:
                    lag_df.loc[policy+"_"+str(lag)+"d_Lag", 'day_'+str(lag):] = dict_list[idx_val][policy]
        eco_lag_list.append(lag_df.T)
        idx_val += 1
    return eco_lag_list


# creates dataframe used to predict daily new cases based on current scenario
def create_orig_datagrid(state):
    state_data = masterdata.loc[masterdata['state'] == state].copy()
    # Fill missing visited values
    for state_to in visited:
        state_data[state_to] = ((state_data[state_to].ffill() + state_data[state_to].bfill()) / 2).ffill()
        state_data[state_to] = ((state_data[state_to].ffill() + state_data[state_to].bfill()) / 2).bfill()

    # Preprocess similar to pipeline
    # treating outliers from CDS_Tested if present
    zero_val_indx = state_data[state_data["CDS_Tested"] == 0]["CDS_Tested"].tail(10).index.tolist()
    if len(zero_val_indx) != 0:
        state_data.loc[zero_val_indx, "CDS_Tested"] = np.nan
        state_data["CDS_Tested"] = ((state_data["CDS_Tested"].ffill() + state_data["CDS_Tested"].bfill()) / 2).ffill()

    # compute dailynew
    for metric in calc_dailynew_metrics:
        state_data[metric + "_dailyNewCases"] = state_data.groupby("state")[metric].diff()
    state_data = state_data.drop(columns=calc_dailynew_metrics)

    # checking for negative values in dailyNew inmputed by CDS
    if (state_data["JHU_ConfirmedCases_dailyNewCases"] <= 0).any():
        state_data.loc[(state_data["JHU_ConfirmedCases_dailyNewCases"] <= 0) & (state_data[
                "CovidTrackingProject_ConfirmedCases_dailyNewCases"] > 0),
                "JHU_ConfirmedCases_dailyNewCases"] = state_data[state_data["JHU_ConfirmedCases_dailyNewCases"] <= 0][
                "CovidTrackingProject_ConfirmedCases_dailyNewCases"]

    # treating neg or zero from CDS_Tested_dailyNew if present
    zero_val_indx = state_data[state_data["CDS_Tested_dailyNewCases"] <= 0]["CDS_Tested_dailyNewCases"].index.tolist()
    if len(zero_val_indx) != 0:
        state_data.loc[zero_val_indx, "CDS_Tested_dailyNewCases"] = np.nan
        state_data["CDS_Tested_dailyNewCases"] = ((state_data["CDS_Tested_dailyNewCases"].ffill() + state_data[
            "CDS_Tested_dailyNewCases"].bfill()) / 2).ffill()

    datagrid = pd.DataFrame(index=list_metrics_expanded, columns=date_list)
    for forward in range(pred_window_size):
        date_val = today + timedelta(days=forward)
        date_val_string = date_val.strftime("%Y-%m-%d")
        column_name = "day_" + str(forward)
        for metric in list_metrics:
            if metric in lag_metrics:
                for lag in lag_vals:
                    lag_date = date_val - timedelta(days=lag)
                    lag_date_string = lag_date.strftime("%Y-%m-%d")
                    try:
                        metric_name = metric + "_" + str(lag) + "d_Lag"
                        datagrid.at[metric_name, column_name] = \
                            state_data[state_data['date'] == lag_date_string][metric].values[0]
                    except:
                        pass
            else:
                if metric == 'weekNr':
                    datagrid.at[metric, column_name] = pd.to_datetime(date_val_string).week
                elif metric == 'dayOfWeek':
                    datagrid.at[metric, column_name] = pd.to_datetime(date_val_string).dayofweek
                elif metric == 'JHU_ConfirmedCases_dailyNewCases':
                    try:
                        cases = 0
                        count = 0
                        for sub in range(1, average_over_days + 1):
                            lag_date = today - timedelta(days=sub)
                            lag_date_string = lag_date.strftime("%Y-%m-%d")
                            cases += (state_data[state_data['date'] == lag_date_string][metric].values[0])
                            count += 1
                        datagrid.at[metric, column_name] = int(cases/count)
                    except:
                        if count == 0:
                            count = 1
                        datagrid.at[metric, column_name] = int(cases/count)
                else:
                    lag_date = today - timedelta(days=1)
                    lag_date_string = lag_date.strftime("%Y-%m-%d")
                    try:
                        datagrid.at[metric, column_name] = \
                            state_data[state_data['date'] == lag_date_string][metric].values[0]
                    except:
                        pass
    datagrid.rename(index={"JHU_ConfirmedCases_dailyNewCases": "JHU_ConfirmedCases_dailyNewCases_1d_Lag"}, inplace=True)
    features_keep = feature_dict[state]
    datagrid = datagrid.loc[features_keep, :]
    datagrid = datagrid.ffill(axis=1)
    return datagrid


# generates daily new case predictions via bootstrapping. For simulated scenarios
# simuflag is true, vaccine efficacy is incorporated
def generate_predictions(datagrid_list, state, simuflag):
    pred_list = []
    daily_pred_model = load(os.path.join('models', state+'_'+model_version+'.joblib'))
    for dg in datagrid_list:
        dg_t = dg.T
        predictions = []
        for forward in range(pred_window_size):
            x = daily_pred_model.predict(dg_t.iloc[[forward]])[0]
            if x < 0:
                x = 0
            if forward != (pred_window_size - 1):
                dg_t.loc['day_'+str(forward + 1), 'JHU_ConfirmedCases_dailyNewCases_1d_Lag'] = x
            predictions.append(x)
        pred_list.append(predictions)
    if simuflag:
        vaccine_pred_list = []
        for pred in pred_list:
            for efficacy in vaccine_dict['VaccineEfficacy']:
                if efficacy == 0:
                    for speed in vaccine_dict['CoverageSpeed_DaysPer10pctPopulation']:
                        vaccine_pred_list.append(pred)
                else:
                    mult_dict = vac_mul_dict[state][efficacy]
                    for speed in vaccine_dict['CoverageSpeed_DaysPer10pctPopulation']:
                        multiplier_key = 10
                        start_index = 0
                        vaccine_pred = []
                        stop_index = speed
                        steps = int(pred_window_size / speed) + 1
                        for step in range(steps):
                            vaccine_pred.extend([x * mult_dict[multiplier_key] for x in pred[start_index:stop_index]])
                            start_index += speed
                            stop_index += speed
                            multiplier_key += 10
                        vaccine_pred_list.append(vaccine_pred)
        return vaccine_pred_list
    return pred_list


# creates list of dataframes used to predict daily new cases based on simulated scenarios
def simulate_datagrid(variable_list, original_grid):
    simulated_list = []
    for policy_dict in variable_list:
        inter_grid = original_grid.copy()
        for policy in policy_dict:
            if policy == "CDS_Tested_dailyNewCases":
                inter_grid.loc[policy, :] = inter_grid.loc[policy, 'day_0'] + policy_dict[policy]
            else:
                for lag in lag_vals:
                    inter_grid.loc[policy+"_"+str(lag)+"d_Lag", 'day_'+str(lag):] = policy_dict[policy]
        simulated_list.append(inter_grid)
    return simulated_list


# generates predictions for economic models
def eco_predictions(datagrid_list, state):
    eco_models = eco_model_dict[state]
    pred_list = []
    for dg in datagrid_list:
        pred_dict = {}
        for model in eco_models:
            loaded_model = load(os.path.join('models', model))
            prediction = loaded_model.predict(dg)
            pred_dict[model] = prediction.tolist()
        pred_list.append(pred_dict)
    return pred_list


# Scoring for all states
for state in state_list:
    state_dict = {'orig_dg': create_orig_datagrid(state)}

    # Store current situation prediction array here
    state_dict['actual_pred'] = generate_predictions([state_dict['orig_dg'].copy()], state, False)[0]

    # Precalculated list of all possible scenarios loaded from saved dictionary
    state_dict['policy_list'] = policy_dict[state]
    state_dict['pol_vac_list'] = policy_vac_dict[state]

    # Simulated dataframes with all possible user inputs. Index mapped to state_dict['policy_list']
    state_dict['simul_dg'] = simulate_datagrid(state_dict['policy_list'], state_dict['orig_dg'])

    # Store list of simulated prediction arrays here. Index mapped to state_dict['policy_list']
    state_dict['simul_pred_list'] = generate_predictions(state_dict['simul_dg'], state, True)

    # Dataframes for economic predictions
    state_dict['jhu_actual_data_list'] = gen_jhu_lag_data(state, [state_dict['actual_pred']])
    state_dict['jhu_simul_data_list'] = gen_jhu_lag_data(state, state_dict['simul_pred_list'])
    state_dict['pol_actual_data_list'] = gen_pol_lag_data(state, [state_dict['actual_pred']], False,
                                                          state_dict['pol_vac_list'])
    state_dict['pol_simul_data_list'] = gen_pol_lag_data(state, state_dict['simul_pred_list'], True,
                                                         state_dict['pol_vac_list'])

    combined_actual_df_list = []
    for i in range(len(state_dict['jhu_actual_data_list'])):
        combined_actual_df_list.append(pd.concat([state_dict['jhu_actual_data_list'][i],
                                                  state_dict['pol_actual_data_list'][i]], axis=1))
    combined_simul_df_list = []
    for i in range(len(state_dict['jhu_simul_data_list'])):
        combined_simul_df_list.append(pd.concat([state_dict['jhu_simul_data_list'][i],
                                                 state_dict['pol_simul_data_list'][i]], axis=1))

    state_dict['combined_actual_df_list'] = combined_actual_df_list
    state_dict['combined_simul_df_list'] = combined_simul_df_list

    # Economic predictions
    state_dict['eco_actual_pred_list'] = eco_predictions(state_dict['combined_actual_df_list'], state)
    state_dict['eco_simul_pred_list'] = eco_predictions(state_dict['combined_simul_df_list'], state)

    # Dataframes for frontend

    # Actual Scenario
    df_cols = ['simu' + str(x + 1) for x in range(no_simul_vars)]
    df_cols.extend(["SimuFlag", "State", "Date", "Future_day", "JHU_ConfirmedCases_dailyNewCase",
                    "OIET_WomplyRevenue_RevenueAll"])
    tab_dg_actual = pd.DataFrame(index=date_list, columns=df_cols)
    state_data = masterdata.loc[masterdata['state'] == state]
    lag_date = today - timedelta(days=1)
    lag_date_string = lag_date.strftime("%Y-%m-%d")
    for locn, metric in enumerate(list(state_dict['policy_list'][0].keys())):
        if metric == "CDS_Tested_dailyNewCases":
            tab_dg_actual['simu' + str(locn + 1)] = metric + "_increment=0"
        else:
            tab_dg_actual['simu' + str(locn + 1)] = metric + "=" + str(int(state_data[state_data['date']
                                                                                      == lag_date_string][
                                                                               metric].values[0]))
    # Populate daily new case predictions
    tab_dg_actual["JHU_ConfirmedCases_dailyNewCase"] = state_dict['actual_pred']
    # Populate economic predictions
    for eco_model in state_dict['eco_actual_pred_list'][0]:
        col_name = eco_model.split(state + '_' + model_version + '_')[1]
        col_name = col_name.split('.joblib')[0]
        tab_dg_actual[col_name] = state_dict['eco_actual_pred_list'][0][eco_model]
    tab_dg_actual['Date'] = (pd.date_range(start=today,
                                           periods=pred_window_size,
                                           freq='1D'))
    tab_dg_actual['Future_day'] = date_list
    tab_dg_actual['State'] = state
    tab_dg_actual['SimuFlag'] = 'Actual'
    tab_dg_actual.reset_index(drop=True, inplace=True)

    # Simulated Scenarios
    scenario_count = len(state_dict['pol_vac_list'])
    for i in range(scenario_count):
        tab_dg = pd.DataFrame(index=date_list, columns=df_cols)
        for locn, policy in enumerate(state_dict['pol_vac_list'][i]):
            if policy == "CDS_Tested_dailyNewCases":
                tab_dg['simu' + str(locn + 1)] = policy + "_increment=" + str(state_dict['pol_vac_list'][i][policy])
            else:
                tab_dg['simu' + str(locn + 1)] = policy + "=" + str(state_dict['pol_vac_list'][i][policy])
        # Populate daily new cases
        tab_dg["JHU_ConfirmedCases_dailyNewCase"] = state_dict['simul_pred_list'][i]
        # Populate economic predictions
        for eco_model in state_dict['eco_simul_pred_list'][i]:
            col_name = eco_model.split(state + '_' + model_version + '_')[1]
            col_name = col_name.split('.joblib')[0]
            tab_dg[col_name] = state_dict['eco_simul_pred_list'][i][eco_model]
        tab_dg['Future_day'] = date_list
        tab_dg['State'] = state
        tab_dg['SimuFlag'] = 'Scenario'
        tab_dg['Date'] = (pd.date_range(start=today,
                                        periods=pred_window_size,
                                        freq='1D'))
        tab_dg.reset_index(drop=True, inplace=True)
        tab_dg_actual = pd.concat([tab_dg_actual, tab_dg])

    state_dict['ui_dg'] = tab_dg_actual
    state_calc_dict[state] = state_dict

# Generate collated dataframe
scored_datagrid = pd.DataFrame()
for state in state_list:
    scored_datagrid = scored_datagrid.append(state_calc_dict[state]['ui_dg'])
collated_dg = "DataGrid_"+today.strftime("%Y-%m-%d")+"_Scored.csv"
scored_datagrid.to_csv(os.path.join('data', collated_dg))
