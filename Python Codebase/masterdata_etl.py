import re
import os
from datetime import timedelta
import requests
import pandas as pd
import numpy as np

dayfirst = False
today = pd.Timestamp.now() - timedelta(days=0)

# Data from the day before is fetched since today's data is updated later on in the day depending on timezone
start = today - timedelta(days=1)
today = today.strftime("%Y-%m-%d")
start = start.strftime("%Y-%m-%d")

# list of states to fetch data for
states = ['Alabama_UnitedStates', 'Alaska_UnitedStates', 'Arizona_UnitedStates',
          'Arkansas_UnitedStates', 'California_UnitedStates', 'Colorado_UnitedStates',
          'Connecticut_UnitedStates', 'Delaware_UnitedStates', 'DistrictofColumbia_UnitedStates',
          'Florida_UnitedStates', 'Georgia_UnitedStates', 'Hawaii_UnitedStates',
          'Idaho_UnitedStates', 'Illinois_UnitedStates', 'Indiana_UnitedStates',
          'Iowa_UnitedStates', 'Kansas_UnitedStates', 'Kentucky_UnitedStates',
          'Louisiana_UnitedStates', 'Maine_UnitedStates', 'Maryland_UnitedStates',
          'Massachusetts_UnitedStates', 'Michigan_UnitedStates', 'Minnesota_UnitedStates',
          'Mississippi_UnitedStates', 'Missouri_UnitedStates', 'Montana_UnitedStates',
          'Nebraska_UnitedStates', 'Nevada_UnitedStates', 'NewHampshire_UnitedStates',
          'NewJersey_UnitedStates', 'NewMexico_UnitedStates', 'NewYork_UnitedStates',
          'NorthCarolina_UnitedStates', 'NorthDakota_UnitedStates', 'Ohio_UnitedStates',
          'Oklahoma_UnitedStates', 'Oregon_UnitedStates', 'Pennsylvania_UnitedStates',
          'PuertoRico_UnitedStates', 'RhodeIsland_UnitedStates', 'SouthCarolina_UnitedStates',
          'SouthDakota_UnitedStates', 'Tennessee_UnitedStates', 'Texas_UnitedStates',
          'Utah_UnitedStates', 'Vermont_UnitedStates', 'Virginia_UnitedStates',
          'Washington_UnitedStates', 'WestVirginia_UnitedStates', 'Wisconsin_UnitedStates',
          'Wyoming_UnitedStates'
          ]

# Used for lex data fetch
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

# list of metrics to fetch
metrics = ["JHU_ConfirmedDeaths",
           "JHU_ConfirmedRecoveries",
           "CovidTrackingProject_ConfirmedCases",
           "CovidTrackingProject_ConfirmedDeaths",
           "CovidTrackingProject_ConfirmedHospitalizations",
           "CovidTrackingProject_NegativeTests",
           "CovidTrackingProject_PendingTests",
           "ECDC_ConfirmedCases",
           "ECDC_ConfirmedDeaths",
           "NYT_ConfirmedCases",
           "NYT_ConfirmedDeaths",
           "NYT_AllCausesDeathsWeekly_Deaths_AllCauses",
           "NYT_AllCausesDeathsWeekly_Excess_Deaths",
           "NYT_AllCausesDeathsWeekly_Expected_Deaths_AllCauses",
           "CDS_Cases",
           "CDS_Deaths",
           "CDS_Discharged",
           "CDS_GrowthFactor",
           "CDS_Hospitalized",
           "CDS_ICU",
           "CDS_Recovered",
           "CDS_Tested",
           "UniversityOfWashington_AdmisMean",
           "UniversityOfWashington_AdmisLower",
           "UniversityOfWashington_AdmisUpper",
           "UniversityOfWashington_AllbedMean",
           "UniversityOfWashington_AllbedLower",
           "UniversityOfWashington_AllbedUpper",
           "OIET_Affinity_SpendAcf",
           "OIET_Affinity_SpendAer",
           "OIET_Affinity_SpendAll",
           "OIET_Affinity_SpendApg",
           "OIET_Affinity_SpendGrf",
           "OIET_Affinity_SpendHcs",
           "OIET_Affinity_SpendTws",
           "OIET_Affinity_SpendAllInchigh",
           "OIET_Affinity_SpendAllInclow",
           "OIET_Affinity_SpendAllIncmiddle",
           "OIET_BurningGlass_BgPosts",
           "OIET_BurningGlass_BgPostsSs30",
           "OIET_BurningGlass_BgPostsSs55",
           "OIET_BurningGlass_BgPostsSs60",
           "OIET_BurningGlass_BgPostsSs65",
           "OIET_BurningGlass_BgPostsSs70",
           "OIET_BurningGlass_BgPostsJz1",
           "OIET_BurningGlass_BgPostsJz2",
           "OIET_BurningGlass_BgPostsJz3",
           "OIET_BurningGlass_BgPostsJz4",
           "OIET_BurningGlass_BgPostsJz5",
           "OIET_UIClaims_InitialClaims",
           "OIET_UIClaims_InitialClaimsRate",
           "OIET_UIClaims_TotalClaims",
           "OIET_UIClaims_TotalClaimsRate",
           "OIET_Employment_All",
           "OIET_Employment_IncLow",
           "OIET_Employment_IncMiddle",
           "OIET_Employment_IncHigh",
           "OIET_Employment_ss40",
           "OIET_Employment_ss60",
           "OIET_Employment_ss65",
           "OIET_Employment_ss70",
           "OIET_WomplyMerchants_MerchantsAll",
           "OIET_WomplyMerchants_MerchantsInchigh",
           "OIET_WomplyMerchants_MerchantsInclow",
           "OIET_WomplyMerchants_MerchantsIncmiddle",
           "OIET_WomplyMerchants_MerchantsSs60",
           "OIET_WomplyMerchants_MerchantsSs65",
           "OIET_WomplyMerchants_MerchantsSs70",
           "OIET_WomplyRevenue_RevenueAll",
           "OIET_WomplyRevenue_RevenueInchigh",
           "OIET_WomplyRevenue_RevenueInclow",
           "OIET_WomplyRevenue_RevenueIncmiddle",
           "OIET_WomplyRevenue_RevenueSs40",
           "OIET_WomplyRevenue_RevenueSs60",
           "OIET_WomplyRevenue_RevenueSs65",
           "OIET_WomplyRevenue_RevenueSs70",
           "OIET_LowIncEarningsSmallBusinesses_Pay",
           "OIET_LowIncEarningsSmallBusinesses_Pay31To33",
           "OIET_LowIncEarningsSmallBusinesses_Pay44To45",
           "OIET_LowIncEarningsSmallBusinesses_Pay48To49",
           "OIET_LowIncEarningsSmallBusinesses_Pay62",
           "OIET_LowIncEarningsSmallBusinesses_Pay72",
           "OIET_LowIncEarningsSmallBusinesses_PayInclow",
           "OIET_LowIncEarningsSmallBusinesses_PayIncmiddle",
           "OIET_LowIncEarningsSmallBusinesses_PayInchigh",
           "OIET_LowIncEarningsAllBusinesses_Pay",
           "OIET_LowIncEarningsAllBusinesses_Pay31To33",
           "OIET_LowIncEarningsAllBusinesses_Pay44To45",
           "OIET_LowIncEarningsAllBusinesses_Pay48To49",
           "OIET_LowIncEarningsAllBusinesses_Pay62",
           "OIET_LowIncEarningsAllBusinesses_Pay72",
           "OIET_LowIncEarningsAllBusinesses_PayInclow",
           "OIET_LowIncEarningsAllBusinesses_PayIncmiddle",
           "OIET_LowIncEarningsAllBusinesses_PayInchigh",
           "OIET_LowIncEmpAllBusinesses_Emp",
           "OIET_LowIncEmpAllBusinesses_Emp31To33",
           "OIET_LowIncEmpAllBusinesses_Emp44To45",
           "OIET_LowIncEmpAllBusinesses_Emp48To49",
           "OIET_LowIncEmpAllBusinesses_Emp62",
           "OIET_LowIncEmpAllBusinesses_Emp72",
           "OIET_LowIncEmpAllBusinesses_EmpInclow",
           "OIET_LowIncEmpAllBusinesses_EmpIncmiddle",
           "OIET_LowIncEmpAllBusinesses_EmpInchigh",
           "OIET_LowIncEmpSmallBusinesses_Emp",
           "OIET_LowIncEmpSmallBusinesses_Emp31To33",
           "OIET_LowIncEmpSmallBusinesses_Emp44To45",
           "OIET_LowIncEmpSmallBusinesses_Emp48To49",
           "OIET_LowIncEmpSmallBusinesses_Emp62",
           "OIET_LowIncEmpSmallBusinesses_Emp72",
           "OIET_LowIncEmpSmallBusinesses_EmpInclow",
           "OIET_LowIncEmpSmallBusinesses_EmpIncmiddle",
           "OIET_LowIncEmpSmallBusinesses_EmpInchigh",
           "Realtor_AvgMedianListingPrice",
           "Realtor_AvgMedianListingPricePerSquareFeet",
           "Realtor_ActiveListingCount",
           "Realtor_NewListingCount",
           "Realtor_PriceIncreasedCount",
           "Realtor_PriceReducedCount",
           "Realtor_PendingListingCount",
           "Realtor_TotalListingCount",
           "Realtor_AvgMedianDaysOnMarket",
           "Realtor_AvgMedianSquareFeet",
           "Realtor_AvgPendingRatio",
           "Realtor_AverageListingPrice",
           "Realtor_AvgPercentChangeMedianListingPriceMm",
           "Realtor_AvgPercentChangeMedianListingPriceYy",
           "Apple_DrivingMobility",
           "Apple_WalkingMobility",
           "Apple_TransitMobility",
           "Google_GroceryMobility",
           "Google_ParksMobility",
           "Google_TransitStationsMobility",
           "Google_RetailMobility",
           "Google_ResidentialMobility",
           "Google_WorkplacesMobility",
           "PlaceIQ_DeviceCount",
           "PlaceIQ_DeviceExposure",
           "PlaceIQ_DeviceCount_Adjusted",
           "PlaceIQ_DeviceExposure_Adjusted",
           "PlaceIQ_DeviceCount_Education1_Adjusted",
           "PlaceIQ_DeviceExposure_Education1_Adjusted",
           "PlaceIQ_DeviceCount_Education2_Adjusted",
           "PlaceIQ_DeviceExposure_Education2_Adjusted",
           "PlaceIQ_DeviceCount_Education3_Adjusted",
           "PlaceIQ_DeviceExposure_Education3_Adjusted",
           "PlaceIQ_DeviceCount_Education4_Adjusted",
           "PlaceIQ_DeviceExposure_Education4_Adjusted",
           "PlaceIQ_DeviceCount_Income1_Adjusted",
           "PlaceIQ_DeviceExposure_Income1_Adjusted",
           "PlaceIQ_DeviceCount_Income2_Adjusted",
           "PlaceIQ_DeviceExposure_Income2_Adjusted",
           "PlaceIQ_DeviceCount_Income3_Adjusted",
           "PlaceIQ_DeviceExposure_Income3_Adjusted",
           "PlaceIQ_DeviceCount_Income4_Adjusted",
           "PlaceIQ_DeviceExposure_Income4_Adjusted",
           "PlaceIQ_DeviceCount_RaceAsian_Adjusted",
           "PlaceIQ_DeviceExposure_RaceAsian_Adjusted",
           "PlaceIQ_DeviceCount_RaceBlack_Adjusted",
           "PlaceIQ_DeviceExposure_RaceBlack_Adjusted",
           "PlaceIQ_DeviceCount_RaceHispanic_Adjusted",
           "PlaceIQ_DeviceExposure_RaceHispanic_Adjusted",
           "PlaceIQ_DeviceCount_RaceWhite_Adjusted",
           "PlaceIQ_DeviceExposure_RaceWhite_Adjusted",
           "PlaceIQ_DeviceCount_Income3",
           "PlaceIQ_DeviceExposure_Income3",
           "PlaceIQ_DeviceCount_Income4",
           "PlaceIQ_DeviceExposure_Income4",
           "PlaceIQ_DeviceCount_RaceAsian",
           "PlaceIQ_DeviceExposure_RaceAsian",
           "PlaceIQ_DeviceCount_RaceBlack",
           "PlaceIQ_DeviceExposure_RaceBlack",
           "PlaceIQ_DeviceCount_RaceHispanic",
           "PlaceIQ_DeviceExposure_RaceHispanic",
           "PlaceIQ_DeviceCount_RaceWhite",
           "PlaceIQ_DeviceExposure_RaceWhite",
           "AverageDailyTemperature",
           "AverageDewPoint",
           "AverageRelativeHumidity",
           "AverageSurfaceAirPressure",
           "AveragePrecipitation",
           "AverageWindSpeed",
           "AverageWindDirection",
           "AverageHorizontalVisibility",
           "AverageWindGustSpeed",
           "AverageSnow",
           "AveragePrecipitationTotal",
           "AveragePressureTendency",
           "OxCGRT_Policy_C1_SchoolClosing",
           "OxCGRT_Policy_C2_WorkplaceClosing",
           "OxCGRT_Policy_C3_CancelPublicEvents",
           "OxCGRT_Policy_C4_RestrictionsOnGatherings",
           "OxCGRT_Policy_C5_ClosePublicTransport",
           "OxCGRT_Policy_C6_StayAtHomeRequirements",
           "OxCGRT_Policy_C7_RestrictionsOnInternalMovement",
           "OxCGRT_Policy_C8_InternationalTravelControls",
           "OxCGRT_Policy_E1_IncomeSupport",
           "OxCGRT_Policy_E2_DebtContractRelief",
           "OxCGRT_Policy_E3_FiscalMeasures",
           "OxCGRT_Policy_E4_InternationalSupport",
           "OxCGRT_Policy_H1_PublicInformationCampaigns",
           "OxCGRT_Policy_H2_TestingPolicy",
           "OxCGRT_Policy_H3_ContactTracing",
           "OxCGRT_Policy_H4_EmergencyInvestmentInHealthcare",
           "OxCGRT_Policy_H5_InvestmentInVaccines",
           "OxCGRT_Policy_M1_Wildcard",
           "OxCGRT_Policy_ConfirmedCases",
           "OxCGRT_Policy_ConfirmedDeaths",
           "OxCGRT_Policy_StringencyIndexForDisplay",
           "OxCGRT_Policy_StringencyLegacyIndexForDisplay",
           "OxCGRT_Policy_GovernmentResponseIndexForDisplay",
           "OxCGRT_Policy_ContainmentHealthIndexForDisplay",
           "OxCGRT_Policy_EconomicSupportIndexForDisplay"
           ]


def read_data_json(typename, api, body):
    """
    read_data_json directly accesses the C3.ai COVID-19 Data Lake APIs using the requests library,
    and returns the response as a JSON, raising an error if the call fails for any reason.
    ------
    typename: The type you want to access, i.e. 'OutbreakLocation', 'LineListRecord', 'BiblioEntry', etc.
    api: The API you want to access, either 'fetch' or 'evalmetrics'.
    body: The spec you want to pass. For examples, see the API documentation.
    """
    response = requests.post(
        "https://api.c3.ai/covid/api/1/" + typename + "/" + api,
        json=body,
        timeout=10000,
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    )

    # if request failed, show exception
    if response.status_code != 200:
        raise Exception(response.json()["message"])

    return response.json()


def evalmetrics(typename, body, get_all=False, remove_meta=True):
    """
    evalmetrics accesses the C3.ai COVID-19 Data Lake using read_data_json, and converts the response into a Pandas dataframe.
    evalmetrics is used for all timeseries data in the C3.ai COVID-19 Data Lake.
    ------
    typename: The type you want to access, i.e. 'OutbreakLocation', 'LineListRecord', 'BiblioEntry', etc.
    body: The spec you want to pass. For examples, see the API documentation.
    get_all: If True, get all metrics and ignore limits on number of expressions and ids. If False, consider expressions and ids limits. The default is False.
    remove_meta: If True, remove metadata about each record. If False, include it. The default is True.
    """
    if get_all:
        expressions = body['spec']['expressions']
        ids = body['spec']['ids']
        df = pd.DataFrame()

        for ids_start in range(0, len(ids), 10):
            for expressions_start in range(0, len(expressions), 4):
                body['spec'].update(
                    ids=ids[ids_start: ids_start + 10],
                    expressions=expressions[expressions_start: expressions_start + 4]
                )
                response_json = read_data_json(typename, 'evalmetrics', body)
                new_df = pd.json_normalize(response_json['result'])
                new_df = new_df.apply(pd.Series.explode)
                df = pd.concat([df, new_df], axis=1)

    else:
        response_json = read_data_json(typename, 'evalmetrics', body)
        df = pd.json_normalize(response_json['result'])
        df = df.apply(pd.Series.explode)

    # get the useful data out
    if remove_meta:
        df = df.filter(regex='dates|data|missing')

    # only keep one date column
    date_cols = [col for col in df.columns if 'dates' in col]
    keep_cols = date_cols[:1] + [col for col in df.columns if 'dates' not in col]
    df = df.filter(items=keep_cols).rename(columns={date_cols[0]: "dates"})
    df["dates"] = pd.to_datetime(df["dates"])

    return df


def reshape_timeseries(timeseries_df):
    state_from_location = lambda x: "_".join(x.split('_')[-2:]).replace("_UnitedStates", "")
    reshaped_ts = pd.melt(
        timeseries_df,
        id_vars=['dates'],
        value_vars=[x for x in timeseries_df.columns if re.match('.*\.(?:data|missing)', x)]
    ).rename(columns={"value": "data", "dates": "date"})

    reshaped_ts["state"] = (
        reshaped_ts["variable"]
        .str.replace("\..*", "")
        .apply(state_from_location)
    )

    reshaped_ts["metric"] = (
        reshaped_ts["variable"]
        .str.replace(".*UnitedStates\.", "")
        .str.replace(".data", "")
    )

    reshaped_ts = reshaped_ts.pivot_table(index=['date', 'state'], columns='metric', aggfunc={'data': np.sum})

    return reshaped_ts


def query_lex(home_state, visited_state, start, today):
    exposure = read_data_json(
        "locationexposure",
        "getlocationexposures",
        {
            "spec":
                {
                    "locationTarget": home_state,
                    "locationVisited": visited_state,
                    "start": start,
                    "end": today,
                }
        }

    )
    lex = pd.json_normalize(exposure["locationExposures"]["value"])
    lex["state"] = lex["locationTarget"].str.replace("_UnitedStates", "")
    lex = lex.rename(columns={"timestamp": "date", "value": "Visited_" + visited_state.replace("_UnitedStates", "")})
    lex["date"] = pd.to_datetime(lex.date)
    lex['date'] = lex['date'].dt.strftime('%Y-%m-%d')
    lex = lex.drop(columns=['locationTarget', 'locationVisited'])
    return lex


# Fetch data for ConfirmedCases and creates initial dataframe
metrics_ts = evalmetrics(
    "outbreaklocation",
    {
        "spec": {
            "ids": states,
            "expressions": ["JHU_ConfirmedCases"],
            "start": start,
            "end": today,
            "interval": "DAY",
        }
    },
    get_all=True
)
masterdata = reshape_timeseries(metrics_ts)
masterdata = pd.DataFrame(masterdata.to_records())
masterdata.rename(columns={masterdata.columns[2]: masterdata.columns[2].split("'")[3]}, inplace=True)
masterdata.rename(columns={masterdata.columns[3]: masterdata.columns[3].split("'")[3]}, inplace=True)

# fetch data for remaining metrics
for metric in metrics:
    try:
        metric_ts = evalmetrics(
            "outbreaklocation",
            {
                "spec": {
                    "ids": states,
                    "expressions": [metric],
                    "start": start,
                    "end": today,
                    "interval": "DAY",
                }
            },
            get_all=True
        )
        metric_data = reshape_timeseries(metric_ts)
        metric_data = pd.DataFrame(metric_data.to_records())
        metric_data.rename(columns={metric_data.columns[2]: metric_data.columns[2].split("'")[3]}, inplace=True)
        metric_data.rename(columns={metric_data.columns[3]: metric_data.columns[3].split("'")[3]}, inplace=True)
        masterdata = pd.merge(masterdata, metric_data, on=['date', 'state'])
    except:
        print("**TimeOut or Other Errors :" + metric)

# fetch lexical data
lex_df = pd.DataFrame()
for hstate in states:
    lex_hdf = pd.DataFrame({'date': [], 'state': []})
    for vstate in states:
        home_state = hstate
        visited_state = vstate
        try:
            pair = query_lex(home_state, visited_state, start, today)
            lex_hdf = pd.merge(lex_hdf, pair, on=['date', 'state'], how='outer', left_index=True)
        except:
            print("**Errors :" + hstate + ":" + vstate)
    lex_df = lex_df.append(lex_hdf)

# Missing data causing empty dataframe handled
if len(lex_df.index) == 0:
    lex_df = lex_df.reindex(columns=lex_df.columns.tolist()
                            + visited)
    state_names = [x.replace("_UnitedStates", "") for x in states]
    lex_df['state'] = state_names
    lex_df['date'] = start

masterdata["weekNr"] = pd.to_datetime(masterdata['date'], dayfirst=True)
masterdata["weekNr"] = pd.to_datetime(masterdata.weekNr).dt.week
masterdata["dayOfWeek"] = pd.to_datetime(masterdata.date).dt.dayofweek
masterdata["year"] = pd.to_datetime(masterdata.date).dt.strftime('%Y')

masterdata['date'] = pd.to_datetime(masterdata['date'], dayfirst=True)
lex_df['date'] = pd.to_datetime(lex_df['date'], dayfirst=True)
masterdata = pd.merge(masterdata, lex_df, on=['date', 'state'], how='left')
masterdata['date'] = masterdata['date'].dt.strftime('%Y-%m-%d')

# Persist data delta
final_file = os.path.join("data", "DailyData_" + today + "_final.csv")
masterdata.to_csv(final_file)

# Collate data delta with previously persisted data
persisted_filename = os.path.join("data", "MasterData_" + start + "_final.csv")
persisted_data = pd.read_csv(persisted_filename)
collated_data = pd.concat([persisted_data, masterdata])

collated_file = os.path.join("data", "MasterData_" + today + "_final.csv")
collated_data.to_csv(collated_file)
