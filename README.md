# CovidFighter
Synechron submission for C3.ai COVID-19 Grand Challenge

CovidFighter offers the following screens for Daily New Cases and Economic Impact predictions (actual current scenario as well as simulations), Backtesting of predictions to gauge model accuracy and feature importance analysis to determine the most influential features

## Daily Predictions (https://synechron-dev9.c3iot.ai/home/scoring)

![Alt text](/Screens/predA.png?raw=true "Current Setting")
![Alt text](/Screens/predB.png?raw=true "Scenario Planning")

* This screen displays the predicted Daily New Cases and Small Business Revenue Change (in percentage)
* Current/Scenario Setting: 'Current Setting' refers to the current actual value of the factors. 'Scenario Planning' is for simulated scenarios consisting of various permutations of the factor values based on their ranges
* Factors 1-4 are arranged in decreasing order of importance with Factor 1 being the most influential for prediction
* Vaccine Efficacy (Factor 5): Efficacy 80 indicates a high quality vaccine, 50 indicates a medium quality vaccine and 0 indicates the vaccine is not in use
* Coverage Speed (Factor 6): This indicates the number of days it takes to cover 10 percent of the population in increments. The less time it takes, the faster the population can be vaccinated
* Current Setting values cannot be combined with factors 5 and 6 filters(Vaccine Efficacy and Coverage Speed) since these are restrcited to Scenario planning. Hence selecting any of the values for factor 5 and 6 filters will return no results


## Backtesting Results (https://synechron-dev9.c3iot.ai/home/backtest)

![Alt text](/Screens/backtest.png?raw=true "Backtesting Results")

* This screen displays the predicted Daily New Cases in comparison to actual Daily New Cases from Johns Hopkins University and COVID Tracking Project
* Out of Time Value: If set to 1, it indicates this datapoint was in the 14 day window not used during traning the model. If it is 0, then it was part of the training data

## Feature Importance (https://synechron-dev9.c3iot.ai/home/features)

![Alt text](/Screens/feature.png?raw=true "Feature Importance")

* This screen displays the Feature Importance in decreasing order of Influence