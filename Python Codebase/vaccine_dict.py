import os
import pandas as pd
from joblib import dump

state_abbvtn = {'Alabama': 'AL',
                'Alaska': 'AK',
                'Arizona': 'AZ',
                'Arkansas': 'AR',
                'California': 'CA',
                'Colorado': 'CO',
                'Connecticut': 'CT',
                'DistrictofColumbia': 'DC',
                'Delaware': 'DE',
                'Florida': 'FL',
                'Georgia': 'GA',
                'Hawaii': 'HI',
                'Idaho': 'ID',
                'Illinois': 'IL',
                'Indiana': 'IN',
                'Iowa': 'IA',
                'Kansas': 'KS',
                'Kentucky': 'KY',
                'Louisiana': 'LA',
                'Maine': 'ME',
                'Maryland': 'MD',
                'Massachusetts': 'MA',
                'Michigan': 'MI',
                'Minnesota': 'MN',
                'Mississippi': 'MS',
                'Missouri': 'MO',
                'Montana': 'MT',
                'Nebraska': 'NE',
                'Nevada': 'NV',
                'NewHampshire': 'NH',
                'NewJersey': 'NJ',
                'NewMexico': 'NM',
                'NewYork': 'NY',
                'NorthCarolina': 'NC',
                'NorthDakota': 'ND',
                'Ohio': 'OH',
                'Oklahoma': 'OK',
                'Oregon': 'OR',
                'Pennsylvania': 'PA',
                'RhodeIsland': 'RI',
                'SouthCarolina': 'SC',
                'SouthDakota': 'SD',
                'Tennessee': 'TN',
                'Texas': 'TX',
                'Utah': 'UT',
                'Vermont': 'VT',
                'Virginia': 'VA',
                'Washington': 'WA',
                'WestVirginia': 'WV',
                'Wisconsin': 'WI',
                'Wyoming': 'WY'}

pos_rate_50 = {
    10: 1,
    20: 3,
    30: 5,
    40: 9,
    50: 17,
    60: 19,
    70: 20,
    80: 20,
    90: 20,
    100: 20
}

pos_rate_80 = {
    10: 1,
    20: 3,
    30: 6,
    40: 11,
    50: 12,
    60: 12,
    70: 12,
    80: 12,
    90: 12,
    100: 12
}


# calculate proportion of immune populace
def get_prop_of_immune(R_0, vaccine_efficacy):
    P = (1 - (1 / R_0)) / vaccine_efficacy
    return P


# function returns R_e value
def get_efficient_transmsn_rate(P, S, R_os, Q, H, E, q, R_oc):
    R_e = (P * S + (1 - P)) * (R_os * (P * (Q * H + (1 - Q) * E) + (1 - P) * (q + (1 - q) * E)) + R_oc)
    return R_e


# function returns R_oc and R_os split from R_0
def get_roc_ros(R_0):
    R_oc = (2 * R_0) / 3
    return round(R_oc, 2), round((R_oc / 2), 2)


def get_vacc_pop(R_e, vaccine_efficacy):
    R_0 = R_e
    P = (1 - (1 / R_0)) / vaccine_efficacy
    return P


def vaccine_simulation(R0, vaccine_efficacy_list):
    vacc_cov = pd.DataFrame(columns=["immune Population in %"])
    protect_rate = {}
    R_0 = R0
    N_0 = R0
    immune_percentage_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    for vaccine_efficacy in vaccine_efficacy_list:
        col_name = str(vaccine_efficacy) + " % Efficacy"
        vacc_cov[col_name] = 0
        vaccine_efficacy = vaccine_efficacy / 100
        s = 1 - vaccine_efficacy
        H = s
        R_oc, R_os = get_roc_ros(R_0=R_0)
        E = 0.9
        Q = 0.04
        q = 0.4
        i = 0
        for immune_percentage in immune_percentage_list:
            vacc_cov.loc[i, "immune Population in %"] = immune_percentage
            immune_percentage = immune_percentage / 100
            p = immune_percentage
            R_e = round(get_efficient_transmsn_rate(P=p, H=H, R_os=R_os, Q=Q, S=s, E=E, q=q, R_oc=R_oc), 1)
            vacc_cov.loc[i, col_name] = R_e

            # method 2
            N_i = (N_0 / R_0) * R_e
            vacc_cov.loc[i, "Protectibility Index for " + col_name] = round((N_0 - N_i) / N_0, 2)

            i = i + 1

        for VE in vaccine_efficacy_list:
            if VE == 80:
                for s in pos_rate_80:
                    p = vacc_cov["Protectibility Index for 80 % Efficacy"][pos_rate_80[s] - 1]
                    protect_rate[s] = round(1 - p, 2)
            elif VE == 50:
                for s in pos_rate_50:
                    p = vacc_cov["Protectibility Index for 50 % Efficacy"][pos_rate_50[s] - 1]
                    protect_rate[s] = round(1 - p, 2)
            else:
                print("Not supported VE level")

    return protect_rate


def get_vaccine_multiplier(state, vaccine_efficacy):
    rt_data = pd.read_csv(os.path.join('data', 'rt.csv'))
    rt_data = rt_data[rt_data["region"] == state_abbvtn[state]]
    rt_data["date"] = pd.to_datetime(rt_data["date"])
    rt_data['date'] = rt_data['date'].dt.strftime('%d/%m/%Y')
    rt_data.sort_values("date", ascending=True, inplace=True)
    R_0 = rt_data["upper_80"].iat[-1]
    protect_rate = vaccine_simulation(R_0, [vaccine_efficacy])
    return protect_rate


# list of supported efficacy values
eff_list = [50, 80]

vac_mul_dict = {}

# generate state-wise efficacy multiplier dictionary
for state in list(state_abbvtn.keys()):
    state_dict = {}
    for eff in eff_list:
        state_dict[eff] = get_vaccine_multiplier(state, eff)
    vac_mul_dict[state] = state_dict

# persist dictionary
dump(vac_mul_dict, os.path.join('data', 'vac_mul_dict.joblib'))
