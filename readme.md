# 🚀 Time Series Forecasting App (for Salesquota) 🚀

Welcome to our Time Series Forecasting App repository(for Salesquota)! This repository contains all the necessary files to run a FastAPI application that trains a time series model and visualizes its predictions. Let's dive in! 🏊‍♂️

## 📁 Files and their Roles 📁

### 🐍 app.py 🐍

This is the heart of our application. It provides an endpoint for training a time series model and visualizing its predictions. It also includes a root path for checking the connection status.

#### Functions/Methods:

1. `train_model(data_input, date_column, value_column)`: An asynchronous function that loads and preprocesses the data, trains the model, and logs any errors that occur during the process. It also generates and visualizes predictions, exporting them to a CSV file. 📈

2. `read_root()`: An asynchronous function that returns a JSON object indicating a successful connection status. 🌐

#### Dependencies:

This file relies on the following external libraries:
- uvicorn: A lightning-fast ASGI server. ⚡
- fastapi: A modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. 🚀
- a2wsgi: A WSGI to ASGI converter. 🔄
- main: A custom module that presumably contains the DataLoader, DataProcessor, AutoTSModel, and PredictionVisualizer classes. 🛠
- logging: A standard Python library for generating logging messages. 📝

#### Usage Examples:

- To train a model, make a POST request to the "/train_model" endpoint with the appropriate parameters.
- To check the connection status, make a GET request to the "/" endpoint.

#### Notes:

- The FastAPI app is wrapped in ASGIMiddleware.
- The app is set to run on localhost (127.0.0.1) at port 8000.
- There are several commented-out lines of code that may be intended for future use or improvements.

### 📜 logfile.log 📜

This file is used to record events or actions that occur while a software or application is running. It helps in tracking errors, user activities, and system behavior for debugging and auditing purposes. 🕵️‍♂️

### 🐍 main.py 🐍

This file is a Python script that is used for loading, preprocessing, and cleaning data, training a time series forecasting model using AutoTS, making predictions, and visualizing and exporting the predictions.

#### Functions/Methods:

1. `DataLoader.load_data_from_json(filepath)`: This method loads data from a JSON file and returns a pandas DataFrame. 📂
2. `DataProcessor.dynamic_preprocess_and_clean(df)`: This method preprocesses and cleans the input DataFrame by handling missing values and removing duplicates. ��
3. `AutoTSModel.train_model(df, date_col, value_col)`: This method trains an AutoTS model on the input DataFrame using the specified date and value columns. 🚂   
4. `AutoTSModel.make_prediction(df)`: This method makes predictions using the trained AutoTS model. 🔮
5. `PredictionVisualizer.visualize_and_export_predictions(df, autots_model, export_csv_path="predictions.csv")`: This method visualizes the predictions made by the AutoTS model and exports them to a CSV file. 📊

#### Dependencies:

This file relies on the following external libraries: pandas, json, AutoTS, joblib, os, logging, matplotlib, and base64.

#### Usage Examples:

- To load data from a JSON file: `data_df = DataLoader.load_data_from_json(filepath)`
- To preprocess and clean a DataFrame: `clean_df = DataProcessor.dynamic_preprocess_and_clean(df)`
- To train an AutoTS model: `autots_model.train_model(df, date_col, value_col)`
- To make predictions using the trained model: `forecast = autots_model.make_prediction(df)`
- To visualize and export the predictions: `predictions = PredictionVisualizer.visualize_and_export_predictions(df, autots_model, export_csv_path="predictions.csv")`

#### Notes:

- The AutoTSModel class requires the AutoTS library to be installed.
- The DataLoader.load_data_from_json method currently loads data from an Excel file, not a JSON file. This might be a mistake or a placeholder for future implementation.
- The PredictionVisualizer.visualize_and_export_predictions method has some commented out code for plotting the predictions and saving the plot as a PNG file. This might be intended for future use.

## 🎉 That's it! 🎉

We hope you find this repository useful for your time series forecasting needs. Happy coding! 🚀👩‍�👨‍‍💻🚀