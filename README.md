# Tigergraph Globalwarming


# Stage-1 : Started with 4 different timeseries datasets related to global warming to understand the impact of Carbon Emission on Temperature Anomaly , Ocean Heat and Arctic sea ice extent 

CO2 : Global Carbon Emission
Temperature Anomaly : Change in global surface temperature relative to 1951-1980 average temperature
Ocean Heat : Ocean heat content change since 1992
Arctic sea ice extent : Annual Arctic sea ice minimum since 1979, based on satellite observations

Currently using monthly datapoints but same graph an be extended to use hourly or daily datapoints as well


# Stage-2 : Used tigergrpah as data source to Regression model post understanding relationship between different datapoints mentioned in stage-1
Model-1: Timeseries forecast to predict Carbon Emission based on historical datapoints 
Model-2: Built Regression model with Carbon Emission as independent variable and Temperature Anomaly , Ocean Heat and Arctic sea ice extent as dependent variables
Used Model-1 forecasted Carbon Emission to forecast future Temperature Anomaly , Ocean Heat and Arctic sea ice extent

Output values from Model-2 can be use to understand any anomalies by comparing against actual captured values . Critical to understand if we heading in right direction or not
This model currently using monthly datapoints but this same model can be used for hourly or daily data samples as well.
