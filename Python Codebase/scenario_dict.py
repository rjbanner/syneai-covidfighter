import os
from itertools import product
import pandas as pd
from joblib import dump

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

# Dictionaries with all possible simulation variables for states and their ranges
simul_range_dict = {'OxCGRT_Policy_C6_StayAtHomeRequirements': [0, 1, 2, 3],
                    'OxCGRT_Policy_C2_WorkplaceClosing': [0, 1, 2, 3],
                    'OxCGRT_Policy_C4_RestrictionsOnGatherings': [0, 1, 2, 3, 4],
                    'OxCGRT_Policy_C7_RestrictionsOnInternalMovement': [0, 1, 2],
                    'OxCGRT_Policy_C3_CancelPublicEvents': [0, 1, 2],
                    'OxCGRT_Policy_H3_ContactTracing': [0, 1, 2],
                    'OxCGRT_Policy_C5_ClosePublicTransport': [0, 1, 2],
                    'CDS_Tested_dailyNewCases': [1000, 2000, 3000],
                    'OxCGRT_Policy_H2_TestingPolicy': [0, 1, 2, 3],
                    'OxCGRT_Policy_C1_SchoolClosing': [0, 1, 2, 3]}

sim_combined_dict = {'OxCGRT_Policy_C6_StayAtHomeRequirements': [0, 1, 2, 3],
                     'OxCGRT_Policy_C2_WorkplaceClosing': [0, 1, 2, 3],
                     'OxCGRT_Policy_C4_RestrictionsOnGatherings': [0, 1, 2, 3, 4],
                     'OxCGRT_Policy_C7_RestrictionsOnInternalMovement': [0, 1, 2],
                     'OxCGRT_Policy_C3_CancelPublicEvents': [0, 1, 2],
                     'OxCGRT_Policy_H3_ContactTracing': [0, 1, 2],
                     'OxCGRT_Policy_C5_ClosePublicTransport': [0, 1, 2],
                     'CDS_Tested_dailyNewCases': [1000, 2000, 3000],
                     'OxCGRT_Policy_H2_TestingPolicy': [0, 1, 2, 3],
                     'OxCGRT_Policy_C1_SchoolClosing': [0, 1, 2, 3],
                     'VaccineEfficacy': [0, 50, 80],
                     'CoverageSpeed_DaysPer10pctPopulation': [5, 10]}


def generate_scenarios(state, var_dict, range_dict, vaccine_flag):
    simul_list_of_lists = []
    policies = var_dict[state].copy()
    if vaccine_flag:
        policies.extend(['VaccineEfficacy', 'CoverageSpeed_DaysPer10pctPopulation'])
    for policy in policies:
        simul_list_of_lists.append(range_dict[policy])
    comb_list = list(product(*simul_list_of_lists))
    key_count = len(policies)
    scenario_list = []
    for combi in comb_list:
        scenario = {}
        for i in range(key_count):
            scenario[policies[i]] = combi[i]
        scenario_list.append(scenario)
    return scenario_list


policy_dict = {}
policy_vac_dict = {}

# gets the simulation variables for each state
simul_dict_file = pd.read_csv(os.path.join('models', 'Sim_Vars_State.csv'))
simul_dict_grouped = simul_dict_file.groupby('state').agg({'featureName': list})
simul_dict_grouped['state'] = simul_dict_grouped.index
simul_dict = pd.Series(simul_dict_grouped.featureName.values, index=simul_dict_grouped.state).to_dict()

# generate state-wise scenario dictionaries
for state in state_list:
    policy_dict[state] = generate_scenarios(state, simul_dict, simul_range_dict, False)
    policy_vac_dict[state] = generate_scenarios(state, simul_dict, sim_combined_dict, True)

# persist dictionaries
dump(policy_dict, os.path.join('data', 'policy_dict.joblib'))
dump(policy_vac_dict, os.path.join('data', 'policy_vac_dict.joblib'))
