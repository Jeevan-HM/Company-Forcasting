import json
import logging
from pathlib import Path

import pandas as pd
from autots import AutoTS

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DataLoader:
    @staticmethod
    def load_data(filepath):
        try:
            if isinstance(filepath, dict):
                data_dict = filepath
            elif isinstance(filepath, str):
                if filepath.endswith(".json"):
                    with Path(filepath).open() as json_file:
                        data_dict = json.load(json_file)
                elif filepath.endswith(".csv"):
                    data_dict = pd.read_csv(filepath)
                elif filepath.endswith((".xls", ".xlsx")):
                    data_dict = pd.read_excel(filepath)
                else:
                    raise ValueError(
                        "Unsupported file format. Only JSON, CSV, and Excel files are supported.",
                    )
            else:
                raise TypeError("Input must be either a dictionary or a file path.")

            if "results" in data_dict:
                data_df = pd.DataFrame(data_dict["results"]["Result"])
            else:
                data_df = pd.DataFrame(data_dict)

        except Exception:
            logger.exception("Error loading data from JSON")
            raise
        else:
            return data_df


class DataProcessor:
    @staticmethod
    def dynamic_preprocess_and_clean(df):
        try:
            df_cleaned = df.copy()
            for column in df_cleaned.columns:
                if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    df_cleaned[column] = df_cleaned[column].fillna(
                        df_cleaned[column].mean()
                    )
                    if pd.api.types.is_integer_dtype(df_cleaned[column]):
                        q1 = df_cleaned[column].quantile(0.25)
                        q3 = df_cleaned[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        df_cleaned = df_cleaned[
                            (df_cleaned[column] >= lower_bound)
                            & (df_cleaned[column] <= upper_bound)
                        ]
                elif pd.api.types.is_datetime64_any_dtype(df_cleaned[column]):
                    df_cleaned[column] = df_cleaned[column].ffill()
                elif pd.api.types.is_string_dtype(df_cleaned[column]):
                    df_cleaned[column] = df_cleaned[column].fillna("Unknown")
                elif pd.api.types.is_categorical_dtype(df_cleaned[column]):
                    df_cleaned[column] = df_cleaned[column].fillna(
                        df_cleaned[column].mode()[0]
                    )
            df_cleaned = df_cleaned.drop_duplicates()

        except Exception:
            logger.exception("Error preprocessing and cleaning data")
            raise
        else:
            return df_cleaned


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
        except Exception:
            logger.exception("Error training AutoTS model")
            raise

    def make_prediction(self, forecast_length):
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            forecast = self.model.predict(forecast_length=forecast_length)
        except Exception:
            logger.exception("Error making predictions")
            raise
        else:
            return forecast


class PredictionVisualizer:
    @staticmethod
    def visualize_and_export_predictions(
        autots_model,
        forecast_length,
        export_csv_path="predictions.csv",
    ):
        try:
            forecast = autots_model.make_prediction(forecast_length)
            predictions = forecast.forecast
            print(predictions)
            predictions.to_csv(export_csv_path, index=True)
        except Exception:
            logger.exception("Error visualizing and exporting predictions")
            raise
        else:
            return predictions
