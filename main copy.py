import pandas as pd
import json
from autots import AutoTS
import joblib
import os
import logging
import matplotlib.pyplot as plt
import base64


# Configure logging
logging.basicConfig(
    filename="logfile.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class DataLoader:
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

            return data_df
        except Exception as e:
            logging.error(f"Error loading data from JSON: {e}")
            raise


class DataProcessor:
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
                    df[column].fillna(method="ffill", inplace=True)
                elif pd.api.types.is_string_dtype(df[column]):
                    df[column].fillna("Unknown", inplace=True)
                elif pd.api.types.is_categorical_dtype(df[column]):
                    df[column].fillna(df[column].mode()[0], inplace=True)
            df.drop_duplicates(inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error preprocessing and cleaning data: {e}")
            raise


class AutoTSModel:
    def __init__(self):
        self.model = None

    def train_model(self, df, date_col, value_col):
        try:
            frequency = "D"
            prediction_interval = 0.9
            max_generations = 0
            num_validations = 0

            self.model = AutoTS(
                frequency=frequency,
                prediction_interval=prediction_interval,
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

    def make_prediction(self, df):
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            forecast = self.model.predict(forecast_length=df["Length"].iloc[-1])
            return forecast
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise


class PredictionVisualizer:
    def visualize_and_export_predictions(
        df, autots_model, export_csv_path="predictions.csv"
    ):
        try:
            forecast = autots_model.make_prediction(df)
            predictions = forecast.forecast
            print(predictions)
            # plt.figure(figsize=(10, 6))
            # predictions.plot()
            # plt.title("AutoTS Forecast")
            # plt.xlabel("Date")
            # plt.ylabel("Predicted Values")
            # plt.grid(True)
            # plt.show(block = False)
            # plt.savefig("forecast_graph.png")
            # with open("C:\\Users\\HP\\Desktop\\Forecasting Production\\forecast_graph.png", "rb") as image_file:
            #     encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            predictions.to_csv(export_csv_path, index=True)
            return predictions
        except Exception as e:
            logging.error(f"Error visualizing and exporting predictions: {e}")
            raise
