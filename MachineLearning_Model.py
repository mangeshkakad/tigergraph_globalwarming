


import pyTigerGraph as tg
import json
import pandas as pd
import json

import requests
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt
from skforecast.ForecasterAutoreg import ForecasterAutoreg


from sklearn.ensemble import RandomForestRegressor
from skforecast.model_selection import grid_search_forecaster
import pickle


'''Connection to tgcloud.io'''
def tg_connection():
    # Connection parameters
    hostName = "https://e58ef4c275ea43059948c09089d35d23.i.tgcloud.io/"
    userName = "tigergraph"
    password = "Nescafe90"

    conn = tg.TigerGraphConnection(host=hostName, username=userName, password=password)

    print("Connected")
    return conn

'''Connection token to tgcloud.io to interact with API'''

def tg_gettoken(conn,graphName):
    print(graphName)
    conn.graphname = graphName
    secret = conn.createSecret()
    token = conn.getToken(secret, setToken=True)
    print(token[0])
    conn.apiToken = token
    print(token[0])

    return token[0]


'''Call http request to execute the query'''

def tg_executequery(conn,token,sYear,fYear,graphName,Query_Name):
    param = ""
    for year in range(sYear,fYear):
        param += 'mYear='+str(year) +"&"
    print (param)
    request_url = "https://e58ef4c275ea43059948c09089d35d23.i.tgcloud.io:9000/query/"+graphName+"/"+Query_Name+"?"+param[:-1]
    res = requests.get(request_url, headers={"Authorization": "Bearer "+token})
    res = json.loads(res.content)['results']
    if res:
        temp = conn.vertexSetToDataFrame(res[0]["temp"])
        co2 = conn.vertexSetToDataFrame(res[1]["co2"])
        extent = conn.vertexSetToDataFrame(res[2]["extent"])
        heat = conn.vertexSetToDataFrame(res[3]["heat"])

    model_dataframe = (co2.set_index('id').join(heat.set_index('id'), lsuffix='_in', rsuffix='_other').join(temp.set_index('id'), lsuffix='_caller', rsuffix='_other').join(extent.set_index('id'), lsuffix='_caller', rsuffix='_other'))
    return model_dataframe


'''Save the model post learning to re-use for next iteration'''

def save_model(model_name,model):
    with open("./Model/"+model_name,"wb") as file:
        pickle.dump(model,file)
    return True

'''Load the Machine Learning model '''

def load_mode(model_name):
    with open("./Model/"+model_name,"rb") as file:
        model = pickle.load(file)
    return model

'''Create a model '''

def build_model(model_dataframe,forecast_type):
    if forecast_type == "Ice_Extent":
        model_df = model_dataframe[['co_value','Extent']].dropna()
        y = model_df['Extent'].values.reshape(-1, 1)
    if forecast_type == "Ocean_Heat":
        model_df = model_dataframe[['co_value', 'Ocean_heat']].dropna()
        y = model_df['Ocean_heat'].values.reshape(-1, 1)
    if forecast_type == "Temperature_Anomaly":
        model_df = model_dataframe[['co_value', 'Temp_Anomaly']].dropna()
        y = model_df['Temp_Anomaly'].values.reshape(-1, 1)

    X = model_df['co_value'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)  # training the algorithm

    y_pred = regressor.predict(X_test)
    #print(y_test)
    #print(y_pred)

    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    df.plot(kind='line', figsize=(16, 10),marker='o')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Error for :'+forecast_type+"  ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Variance score: %.2f' % regressor.score(X_test, y_test))

    save_model(forecast_type,regressor)



'''Build CO2 timeseries forecast model '''

def build_co2_model(model_dataframe,graphName,token):
    model_df = model_dataframe[['co_value']].dropna()
    steps = 12
    data_train = model_df[:-steps]
    data_test = model_df[-steps:]

    #hyperparameter_tune_co2_timeseries(data_train)
    # Create and train forecaster
    # ==============================================================================
    forecaster = ForecasterAutoreg(
        regressor=RandomForestRegressor(max_depth=10,n_estimators=500,random_state=123),
        lags=20
    )

    forecaster.fit(y=data_train['co_value'])
    print(forecaster)
    # Predictions
    # ==============================================================================
    steps = 15
    validation_predictions = forecaster.predict(steps=steps)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(data_test['co_value'], validation_predictions[:-3])))
    y = [1,2,3,4,5,6,7,8,9,10,11,12]
    plt.plot(y,data_test['co_value'],label = "Actual",marker = 'o')
    plt.plot(y,validation_predictions[:-3],label = "Predicted",marker = 'o')
    plt.ylim(200,500)
    plt.legend()
    plt.show()
    forecast_df = validation_predictions[-3:]
    last_year = int(model_dataframe[-1:]['Year_in'][0])
    last_month = int(model_dataframe[-1:]['Month_in'][0])
    print(forecast_df)
    for i in range(1,4):
        if last_month < 12:
            last_month+=1
        else:
            last_month=1
            last_year+=1
        extent = forecast_model("Ice_Extent",forecast_df.iloc[i-1])
        Ocean_heat = forecast_model("Ocean_Heat",forecast_df.iloc[i-1])
        temp_anomaly = forecast_model("Temperature_Anomaly",forecast_df.iloc[i-1])
        print(f'Year {str(last_year)} Month {str(last_month)}')
        print (f'Forecast for CO2 is {str(forecast_df.iloc[i-1])}')
        print (f'Forecast for Ice Extent is {str(extent)}')
        print (f'Forecast for Ocean Heat is {str(Ocean_heat)}')
        print (f'Forecast for Temp Anomaly is {str(temp_anomaly)}')
        insert_records(last_year,last_month,forecast_df.iloc[i-1],graphName,token,extent,Ocean_heat,temp_anomaly)

'''Insert forecasted records '''

def insert_records(year,month,co_value,graphName,token,extent,Ocean_heat,temp_anomaly):
    param = "year="+str(year)+"&month="+str(month)+"&co_value="+str(co_value)+"&extent="+str(extent)+"&temp_anomaly="+str(temp_anomaly)+"&Ocean_heat="+str(Ocean_heat)
    request_url = "https://e58ef4c275ea43059948c09089d35d23.i.tgcloud.io:9000/query/"+graphName+"/"+"Insert_Records"+"?"+param
    res = requests.get(request_url, headers={"Authorization": "Bearer "+token})
    if res.status_code == 200:
        print ("Done....")

'''Hyperparameter Tuning for CO2 timeseries forecasting model '''

def hyperparameter_tune_co2_timeseries(data_train):
    # Hyperparameter Grid search
    # ==============================================================================
    steps = 36
    forecaster = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=123),
        lags=12  # This value will be replaced in the grid search
    )

    # Lags used as predictors
    lags_grid = [10, 20]

    # Regressor's hyperparameters
    param_grid = {'n_estimators': [100, 500],
                  'max_depth': [3, 5, 10]}

    results_grid = grid_search_forecaster(
        forecaster=forecaster,
        y=data_train['co_value'],
        param_grid=param_grid,
        lags_grid=lags_grid,
        steps=steps,
        refit=True,
        metric='mean_squared_error',
        initial_train_size=int(len(data_train) * 0.5),
        fixed_train_size=False,
        return_best=True,
        verbose=False
    )


'''Forecast value using save model '''
def forecast_model(model_name,co_value):
    model = load_mode(model_name)
    pred = model.predict(np.array([[co_value]]))[0][0]
    return (pred)

'''Delete record from Graph '''

def delete_records(year,months,graphName,token):
    param = ""
    for month in months:
        param += "&month=" + str(month)
    param = "year="+str(year)+param
    request_url = "https://e58ef4c275ea43059948c09089d35d23.i.tgcloud.io:9000/query/"+graphName+"/"+"Delete_Records"+"?"+param
    res = requests.get(request_url, headers={"Authorization": "Bearer "+token})
    if res.status_code == 200:
        print ("Done....")


'''Main runner from wherre the execution starts '''

def main_runner():
    graphName = "Global_Climate_Change"
    Query_Name = "getDataForMLModelv6"
    forecast_co2_timeseries = False
    forcast_Arctic_Sea_Ice_Extent = True
    forcast_Ocean_Heat = False
    forcast_Temperature_Anomaly = False
    build_model_flag = True
    delete_flag = True
    conn = tg_connection()
    if conn:
        token = tg_gettoken(conn,graphName)
        model_dataframe = tg_executequery(conn,token,1984,2023,graphName,Query_Name)
        if delete_flag == True:
            delete_records(2022,[4,5,6,7,8,9,10,11,12],graphName,token)
        if build_model_flag:
            if forecast_co2_timeseries:
                build_co2_model(model_dataframe,graphName,token)
            if forcast_Arctic_Sea_Ice_Extent:
                build_model(model_dataframe,"Ice_Extent")
            if forcast_Ocean_Heat:
                build_model(model_dataframe,"Ocean_Heat")
            if forcast_Temperature_Anomaly:
                build_model(model_dataframe,"Temperature_Anomaly")
        else:
            if forcast_Arctic_Sea_Ice_Extent:
                forecast_model("Ice_Extent")
            if forcast_Ocean_Heat:
                forecast_model("Ocean_Heat")
            if forcast_Temperature_Anomaly:
                forecast_model("Temperature_Anomaly")



if __name__ == "__main__":
    main_runner()