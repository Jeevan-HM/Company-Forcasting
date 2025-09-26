import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from a2wsgi import ASGIMiddleware
import main
import logging
import json
from pydantic import BaseModel


class TrainModelRequest(BaseModel):
    data_input: str
    date_column: str
    value_column: str
    forecast_length: int


# Create a FastAPI app
app = FastAPI(
    title="Forecasting",
    docs_url="/forecasting/api/docs",
    redoc_url=None,
    openapi_url="/forecasting/api/openapi.json",
)

# CORS settings
origins = ["*"]

# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Existing endpoint for training model
@app.post("/forecasting/api/train_model")
async def train_model(request: TrainModelRequest):
    try:
        data_input = request.data_input
        date_column = request.date_column
        value_column = request.value_column
        forecast_length = request.forecast_length
        # Load and preprocess data
        data_loader = main.DataLoader()
        try:
            print(data_input)
            data_input_dict = json.loads(data_input)
            print(data_input)
            df = data_loader.load_data(data_input_dict)
        except:
            df = data_loader.load_data(data_input)
        data_processor = main.DataProcessor()
        df = data_processor.dynamic_preprocess_and_clean(df)

        # Train model
        auto_ts_model = main.AutoTSModel()

        auto_ts_model.train_model(df, date_column, value_column, forecast_length)
        logging.info("Model trained successfully")
    except Exception as e:
        logging.error(f"Model training error: {e}")
        return {"message": "failed", "result": e}

    try:
        # Generate and visualize predictions
        prediction_visualizer = main.PredictionVisualizer()
        forecast_result = prediction_visualizer.visualize_and_export_predictions(
            df, auto_ts_model, forecast_length, export_csv_path="predictions.csv"
        )

        return {"message": "success", "result": forecast_result}
    except Exception as e:
        logging.error(f"Visualization error: {e}")
        return {"message": "failed", "result": e}


# Wrap the FastAPI app in ASGIMiddleware
wsgi_app = ASGIMiddleware(app)
logging.info("App created successfully")


# Define the Root path
@app.get("/")
async def read_root():
    return {"status": "Connection Successful"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
