def get_data_from_source(path_to_csv: str):
    import logging
    logging.info("Importing required packages ")
    import os
    from sklearn.model_selection import train_test_split as tts
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f"user selected to read from a csv file.. reading csv file as input from {path_to_csv}")
    df = pd.read_csv(path_to_csv, delimiter=",")
    df.columns = [*df.columns[:-1], 'target']
                
    train, test = tts(df, test_size=0.2)
    train.to_csv(path_to_csv.replace(".csv", "_train.csv"), index = False, encoding='utf-8-sig')
    test.to_csv(path_to_csv.replace(".csv", "_test.csv"), index = False, encoding='utf-8-sig')

def preprocess(path_to_csv: str):
    import logging
    logging.info("Importing required packages ")
    import pandas as pd
    import numpy as np
    import pickle
    import gcsfs
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from pandas.api.types import is_numeric_dtype
    from sklearn.ensemble import IsolationForest
    
    # Read the Data
    logging.info(f"Reading the Input Dataframe..reading csv file as input from {path_to_csv}")
    df = pd.read_csv(path_to_csv, delimiter=",")
    df.columns = [*df.columns[:-1], 'target']
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    logging.info(f"Reading the Train Dataframe.. reading csv file as input from {path_to_csv.replace('.csv', '_train.csv')}")
    df_train = pd.read_csv(path_to_csv.replace(".csv", "_train.csv"))
    X_train = df_train.iloc[:,:-1]
    y_train = df_train.iloc[:,-1]
    
    logging.info(f"Reading the Test Dataframe.. reading csv file as input from {path_to_csv.replace('.csv', '_test.csv')}")
    df_test = pd.read_csv(path_to_csv.replace(".csv", "_test.csv"))
    X_test = df_test.iloc[:,:-1]
    y_test = df_test.iloc[:,-1]
    
    logging.info("Converting Numberic Signals into Categorical that are intrinsically categorical in nature e.g., Maturity Level")
    for col in X_train:
            if(is_numeric_dtype(X_train[col])):
                unique = X_train[col].nunique()
                total_count = X_train[col].count()
                contvscat = unique/total_count * 100

                if(contvscat <= 5):
                    X_train[col]=pd.Categorical(X_train[col])
                    
    
    logging.info("Recording Categorical & Numeric Column Names")
    numeric_columns=list(X_train.select_dtypes(include='number').columns)
    categorical_columns=list(X_train.select_dtypes(exclude='number').columns)
    
    logging.info("Remove Outliers from Numeric Columns")
    if numeric_columns:
        # identify outliers in the training dataset
          
        iso = IsolationForest(contamination=0.1)
        yhat = iso.fit_predict(X_train[numeric_columns].values)
        # select all rows that are not outliers
        mask = yhat != -1
        X_train, y_train = X_train.iloc[mask, :], y_train.iloc[mask]
    
    logging.info("Processing Categorical Signals by imputing missing values with most_frequent & encoding (OHT) these Signals ")
    # If Numeric, then Impute with Median, StandardScaler
    # Define categorical pipeline
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    logging.info("Processing Numeric Signals by imputing missing values with mean & encoding (OHT) these Signals ")
    # Define numerical pipeline
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combine categorical and numerical pipelines
    preprocessor = ColumnTransformer([
        ('cat', cat_pipe, categorical_columns),
        ('num', num_pipe, numeric_columns)
    ])
    
    #Training
    logging.info("Fiting the transformers on train data set and transforming the test data set ")
    preprocessor.set_output(transform="pandas")
    df_preprocessed_train = preprocessor.fit_transform(X_train)
    df_preprocessed_train = pd.concat([df_preprocessed_train, y_train], axis = 1)
    
    #Transforming
    df_preprocessed_test = preprocessor.transform(X_test)
    df_preprocessed_test = pd.concat([df_preprocessed_test, y_test], axis = 1)
    
    df_preprocessed_input = preprocessor.transform(X)
    df_preprocessed_input = pd.concat([df_preprocessed_input, y], axis = 1)
    
    logging.info("Saving the files in GCP Buckets")
    df_preprocessed_train.to_csv(path_to_csv.replace(".csv", "_train_preprocessed.csv"), index = False, encoding='utf-8-sig')
    df_preprocessed_test.to_csv(path_to_csv.replace(".csv", "_test_preprocessed.csv"), index = False, encoding='utf-8-sig')
    df_preprocessed_input.to_csv(path_to_csv.replace(".csv", "_input_preprocessed.csv"), index = False, encoding='utf-8-sig')
    
    logging.info("Saving the Processor as a pkl file in GCP bucket")
    fs = gcsfs.GCSFileSystem(project = 'acs-is-agai-dev')
    path = path_to_csv.replace(".csv","-rf-preprocessor.pkl")
    with fs.open(path, 'wb') as preprocessor_file:
        pickle.dump(preprocessor, preprocessor_file)


def create_model(path_to_csv: str, model_type: str):
    import logging
    logging.info("Importing required packages ")
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    import pandas as pd
    from sklearn.metrics import r2_score
    import pickle
    import gcsfs
    fs = gcsfs.GCSFileSystem(project = 'acs-is-agai-dev')
    logging.info("Reading the preprocessed training dataset.. ")
    data = pd.read_csv(path_to_csv.replace(".csv", "_train_preprocessed.csv")) 
    
    path = path_to_csv.replace(".csv","-model.pkl")
    
    logging.info("Fitting the model on training dataset")
    if model_type == 'random forest':
        model = RandomForestRegressor(n_estimators=10)
        model.fit(
        data.drop(columns=["target"]),
        data.target)
        logging.info("Saving the model as pkl file")
        with fs.open(path, 'wb') as model_file:
            pickle.dump(model, model_file)
            
    elif model_type == 'decision tree':
        model = DecisionTreeRegressor()
        model.fit(
        data.drop(columns=["target"]),
        data.target)
        logging.info("Saving the model as pkl file")
        with fs.open(path, 'wb') as model_file:
            pickle.dump(model, model_file)
            
    elif model_type == 'adaboost':
        model = AdaBoostRegressor()
        model.fit(
        data.drop(columns=["target"]),
        data.target)
        logging.info("Saving the model as pkl file")
        with fs.open(path, 'wb') as model_file:
            pickle.dump(model, model_file)
            
            
    elif model_type == 'best of all':
        rf_model = RandomForestRegressor(n_estimators=10)
        dt_model = DecisionTreeRegressor()
        ada_model = AdaBoostRegressor()
        rf_model.fit(data.drop(columns=["target"]),data.target)
        dt_model.fit(data.drop(columns=["target"]),data.target)
        ada_model.fit(data.drop(columns=["target"]),data.target)
        
        data_test = pd.read_csv(path_to_csv.replace(".csv", "_test_preprocessed.csv"))
        y_test = data_test.drop(columns=["target"])
        y_target=data_test.target
        r2_scores = {}
        r2_scores['rf']  = r2_score(y_target, rf_model.predict(y_test))
        r2_scores['dt']  = r2_score(y_target, dt_model.predict(y_test))
        r2_scores['ada']  = r2_score(y_target, ada_model.predict(y_test))
        best_model = max(r2_scores, key=r2_scores.get)
        
        if best_model == 'dt':
            logging.info("Saving the model as pkl file")
            with fs.open(path, 'wb') as model_file:
                pickle.dump(dt_model, model_file)
       
        elif best_model == 'rf':
            logging.info("Saving the model as pkl file")
            with fs.open(path, 'wb') as model_file:
                pickle.dump(rf_model, model_file)

        else:
            logging.info("Saving the model as pkl file")
            with fs.open(path, 'wb') as model_file:
                pickle.dump(ada_model, model_file)

    
    else:    
        logging.exception("Please choose from the right model 1) decision tree, 2) random forest 3) adaboost 4) best of all!")

def model_evaluation(path_to_csv: str):
    import logging
    logging.info("Importing required packages ")
    from sklearn.metrics import r2_score
    import pandas as pd
    import pickle
    import json
    import typing
    import gcsfs
    import datetime
    import numpy as np
    fs = gcsfs.GCSFileSystem(project = 'acs-is-agai-dev')
    logging.info("Reading the preprocessed test dataset.. ")
    data_test = pd.read_csv(path_to_csv.replace(".csv", "_test_preprocessed.csv"))
    
    path_to_model = path_to_csv.replace(".csv","-model.pkl")
    logging.info("Loading the Model")
    with fs.open(path_to_model, 'rb') as model_file:
        model = pickle.load(model_file)
     
    logging.info("Calculating the Performance Metrics & Storing in file")
    y_test = data_test.drop(columns=["target"])
    y_target=data_test.target
    y_test_pred = model.predict(y_test)
    
    r2 = r2_score(y_target, y_test_pred)
    r2 = np.round_(r2, decimals = 2)
  
    path = path_to_csv.replace(".csv",  f"_{datetime.datetime.now()}_kpi.txt")
    with fs.open(path, 'w') as performance_metric_file:
        performance_metric_file.write(str(r2))
        
    logging.info("Reading the preprocessed full input dataset.. ")    
    data_preprocessed = pd.read_csv(path_to_csv.replace(".csv", "_input_preprocessed.csv"))   
    
    logging.info("Predicting Values of Target & Storing in file")
    y_preprocessed = data_preprocessed.drop(columns=["target"])
    y_target=data_preprocessed.target
    y_pred = model.predict(y_preprocessed)
    path_pred = path_to_csv.replace(".csv",  f"_{datetime.datetime.now()}_prediction.txt")
    with fs.open(path_pred, 'w') as y_pred_file:
        for pred in y_pred:
            y_pred_file.write(str(pred)+"\n")

        

