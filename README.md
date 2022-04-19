# Tigergraph Globalwarming


# Stage-1 : Started with 4 different timeseries datasets related to global warming to understand the impact of Carbon Emission on Temperature Anomaly , Ocean Heat and Arctic sea ice extent 

CO2 : Global Carbon Emission

Temperature Anomaly : Change in global surface temperature relative to 1951-1980 average temperature

Ocean Heat : Ocean heat content change since 1992

Arctic sea ice extent : Annual Arctic sea ice minimum since 1979, based on satellite observations

Currently using monthly datapoints but same graph an be extended to use hourly or daily datapoints as well



![image](https://user-images.githubusercontent.com/11903851/164027803-72a04ad8-dfb6-4506-9598-d0e4a1c044c7.png)


# Stage-2 : Used tigergrpah as data source to Regression model post understanding relationship between different datapoints mentioned in stage-1
Model-1: Timeseries forecast to predict Carbon Emission based on historical datapoints 

![image](https://user-images.githubusercontent.com/11903851/164051587-e6bf6666-a8a0-4037-90ce-969e87f22ebd.png)


Model-2: Built Regression model with Carbon Emission as independent variable and Temperature Anomaly , Ocean Heat and Arctic sea ice extent as dependent variables

Used Model-1 forecasted Carbon Emission to forecast future Temperature Anomaly , Ocean Heat and Arctic sea ice extent

Output values from Model-2 can be use to understand any anomalies by comparing against actual captured values . Critical to understand if we heading in right direction or not

This model currently using monthly datapoints but this same model can be used for hourly or daily data samples as well.

# Execution Flow

Run TigerGraph_Runner.ipynb to build graph and upload data

Run MachineLearning_Model.py to build model and forecast data

    graphName = "Global_Climate_Change" ----- Name of Graph
    Query_Name = "getDataForMLModelv6" ---- Name of Query to extract data to start building Machine Learning Model 
    forecast_co2_timeseries = False    ----- Flag True or False to create timeseries forecast for Carbon Emission
    forcast_Arctic_Sea_Ice_Extent = False  ----- Flag True or False to create model and forecast for Arctic Sea Ice Extent
    forcast_Ocean_Heat = False    ----- Flag True or False to create model and forecast for Ocean Heat
    forcast_Temperature_Anomaly = False   ----- Flag True or False to create model and forecast for Temperature Anomaly
    build_model_flag = False   ----- Flag True or False to build the model or just execute
    delete_flag = True    ----- Flag True or False to delete the forecasted data points from Graph
