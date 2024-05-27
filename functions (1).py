def get_data_from_source(path_to_csv: str):
    
    import logging
    logging.info("Importing required packages ")
    import os
    import pandas as pd
    from pmdarima.model_selection import train_test_split
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f"user selected to read from a csv file.. reading csv file as input from {path_to_csv}")
    df = pd.read_csv(path_to_csv, delimiter=",",index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = [*df.columns[:-1], 'Target']
    
    test_size = int(len(df)*0.2)
    train, test = train_test_split(df, test_size=test_size)
    
    train.to_csv(path_to_csv.replace(".csv", "_train.csv"), index = True, encoding='utf-8-sig')
    test.to_csv(path_to_csv.replace(".csv", "_test.csv"), index = True, encoding='utf-8-sig')
    
    
def create_model(path_to_csv: str, frequency: str, seasonality: str, duration: str):
    
    import logging
    logging.info("Importing required packages ")
    import os
    import pandas as pd
    import datetime
    import pmdarima as pm
    from pmdarima.pipeline import Pipeline
    from pmdarima.preprocessing import LogEndogTransformer
    from pmdarima.preprocessing import BoxCoxEndogTransformer
    import pickle
    import gcsfs
    fs = gcsfs.GCSFileSystem(project = 'acs-is-agai-dev')
    logging.info("Reading the preprocessed training dataset.. ")
    
    duration = int(duration)
    seasonality = int(seasonality)
    
    if frequency == 'week':
        freq = 'W'
    elif frequency == 'month':
        freq = 'M'
    elif frequency == 'year':
        freq = 'A'
    elif frequency == 'quarter':
        freq = 'Q'
    else:
        freq = 'D'
        
    train = pd.read_csv(path_to_csv.replace(".csv", "_train.csv")) 
    test = pd.read_csv(path_to_csv.replace(".csv", "_test.csv")) 
    
    pipe = Pipeline([
        #("boxcox", BoxCoxEndogTransformer(lmbda2=1e-6)),
        ("logend",LogEndogTransformer(lmbda=1e-6)),
        ("arima", pm.AutoARIMA(start_p=0,
                               D=0,
                               start_q=0,
                               max_p=5,
                               max_D=5,
                               max_q=5,
                               test="adf",
                               seasonal=True,trace=True,stepwise=True,m=seasonality))])

    
    logging.info("Fitting the model on training dataset")
    
    model =pipe.fit(train['Target'])
    
    train_pred=model.predict_in_sample()
    train_pred_list = pd.Series(train_pred,index =train['Date'])
    length=len(test)+duration
    forecast=pipe.predict(n_periods=length)
    forecast_range=pd.date_range(start=test['Date'].iloc[0], periods=length,freq=freq)
    forecast_list = pd.Series(forecast, index=forecast_range)
    
    
    
    output_df =pd.concat([train_pred_list, forecast_list])
    output_df = pd.DataFrame(output_df,columns=['Value'])
    output_df=output_df[1:]
    output_df.index = pd.to_datetime(output_df.index)
    decimals = 2    
    output_df['Value'] = output_df['Value'].apply(lambda x: round(x, decimals))

    output_df.to_csv(path_to_csv.replace(".csv", "_prediction.csv"), index = True, encoding='utf-8-sig')
    
    path = path_to_csv.replace(".csv","-model.pkl")
    logging.info("Saving the model as pkl file")
    with fs.open(path, 'wb') as model_file:
            pickle.dump(model, model_file)
    
    
def model_evaluation(path_to_csv: str):
    import logging
    logging.info("Importing required packages ")
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error as mse
    import pickle
    import gcsfs
    import datetime
    
    fs = gcsfs.GCSFileSystem(project = 'acs-is-agai-dev')
    logging.info("Reading test dataset.. ")
    test = pd.read_csv(path_to_csv.replace(".csv", "_test.csv"))
    
    path_to_model = path_to_csv.replace(".csv","-model.pkl")
    logging.info("Loading the Model")
    with fs.open(path_to_model, 'rb') as model_file:
        model = pickle.load(model_file)
    
    logging.info("Predict using the trained ARIMA model for model evaluation.. ")
    test_pred=model.predict(n_periods=len(test))
    
    logging.info("Calculating the Performance Metrics & Storing in file")
    RMSE=round(np.sqrt(mse(test['Target'], test_pred)),2)
    path = path_to_csv.replace(".csv",  f"_{datetime.datetime.now().date()}_kpi.txt")
    with fs.open(path, 'w') as performance_metric_file:
        performance_metric_file.write(str(RMSE))
    
    