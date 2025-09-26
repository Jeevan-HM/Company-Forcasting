import pandas as pd
import json
from autots import AutoTS
import joblib
import os
import logging
import matplotlib.pyplot as plt
import base64

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("logfile.log"),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)


class DataLoader:
    @staticmethod
    def load_data(filepath):
        try:
            # data_dict = json.loads(json_string)
            # data_df = pd.DataFrame(data_dict["results"]["Result"])
            if isinstance(filepath, dict):
                data_dict = filepath
            elif isinstance(filepath, str):
                filepath = filepath
                if filepath.endswith(".json"):
                    with open(filepath, "r") as json_file:
                        data_dict = json.load(json_file)
                elif filepath.endswith(".csv"):
                    data_dict = pd.read_csv(filepath)
                elif filepath.endswith((".xls", ".xlsx")):
                    data_dict = pd.read_excel(filepath)
                else:
                    raise ValueError(
                        "Unsupported file format. Only JSON, CSV, and Excel files are supported."
                    )
            else:
                raise ValueError("Input must be either a dictionary or a file path.")

            if "results" in data_dict:
                data_df = pd.DataFrame(data_dict["results"]["Result"])
            else:
                data_df = pd.DataFrame(data_dict)

            return data_df

        except Exception as e:
            logging.error(f"Error loading data from JSON: {e}")
            raise


class DataProcessor:
    @staticmethod
    def dynamic_preprocess_and_clean(df):
        try:
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].mean(), inplace=True)
                    # Additional step to remove outliers from integer columns
                    if pd.api.types.is_integer_dtype(df[column]):
                        Q1 = df[column].quantile(0.25)
                        Q3 = df[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[
                            (df[column] >= lower_bound) & (df[column] <= upper_bound)
                        ]
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    # Replace inplace operation with assignment
                    df[column] = df[column].fillna(method="ffill")
                elif pd.api.types.is_string_dtype(df[column]):
                    # Replace inplace operation with assignment
                    df[column] = df[column].fillna("Unknown")
                elif pd.api.types.is_categorical_dtype(df[column]):
                    # Replace inplace operation with assignment
                    df[column] = df[column].fillna(df[column].mode()[0])
            df.drop_duplicates(inplace=True)

            return df
        except Exception as e:
            logging.error(f"Error preprocessing and cleaning data: {e}")
            raise


class AutoTSModel:
    def __init__(self):
        self.model = None

    def train_model(self, df, date_col, value_col, forecast_length):
        try:
            frequency = "D"
            prediction_interval = 0.9
            max_generations = 0
            num_validations = 0

            self.model = AutoTS(
                frequency=frequency,
                prediction_interval=prediction_interval,
                forecast_length=forecast_length,
                ensemble="simple",
                model_list="probabilistic",
                transformer_list="all",
                max_generations=max_generations,
                num_validations=num_validations,
            )
            self.model.fit(df, date_col=date_col, value_col=value_col, id_col=None)
        except Exception as e:
            logging.error(f"Error training AutoTS model: {e}")
            raise

    def make_prediction(self, df, forecast_length):
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            forecast = self.model.predict(forecast_length=forecast_length)
            return forecast
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise


class PredictionVisualizer:
    @staticmethod
    def visualize_and_export_predictions(
        df,
        autots_model,
        forecast_length,
        export_csv_path="predictions.csv",
    ):
        try:
            forecast = autots_model.make_prediction(df, forecast_length)
            predictions = forecast.forecast
            print(predictions)
            predictions.to_csv(export_csv_path, index=True)
            return predictions
        except Exception as e:
            logging.error(f"Error visualizing and exporting predictions: {e}")
            raise
