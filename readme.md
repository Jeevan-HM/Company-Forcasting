# 🚀 Time Series Forecasting API 🚀

This repository contains a FastAPI application for time series forecasting. The application provides an API to train a forecasting model and get predictions.

## 📁 Files and their Roles

*   **`app.py`**: The main FastAPI application file. It defines the API endpoints for training the model and making predictions.
*   **`main.py`**: This file contains the core logic for data loading, preprocessing, model training, and prediction. It includes the following classes:
    *   `DataLoader`: Handles loading data from various formats (JSON, CSV, Excel).
    *   `DataProcessor`: Cleans and preprocesses the data.
    *   `AutoTSModel`: A wrapper for the `AutoTS` library to train the forecasting model.
    *   `PredictionVisualizer`: Handles generating and exporting predictions.
*   **`requirements.txt`**: A list of all the Python packages required to run the application.
*   **`pyproject.toml`**: Project metadata and configuration file.
*   **`Synthetic_Data_60_Rows.csv`**: Sample data file.

## ⚙️ How to Run the Application

### 1. Install Dependencies

First, you need to install the required Python packages. Open a terminal and run the following command:

```bash
pip install -r requirements.txt
```

### 2. Run the API Server

Once the dependencies are installed, you can start the FastAPI application by running the following command in your terminal:

```bash
python app.py
```

This will start a local server, and you can access the application at `http://127.0.0.1:8000` in your web browser.

### 3. Access the API Documentation

The API documentation is automatically generated and available at `http://127.0.0.1:8000/forecasting/api/docs`. You can use this documentation to test the API endpoints.

## ☁️ API Endpoints

### `POST /forecasting/api/train_model`

This endpoint trains the time series forecasting model. You need to provide the data, date column, value column, and forecast length in the request body.

**Request Body:**

```json
{
  "data_input": "path/to/your/data.csv",
  "date_column": "your_date_column",
  "value_column": "your_value_column",
  "forecast_length": 30
}
```

The `data_input` can be a file path or a JSON string containing the data.

**Response:**

The API will return a JSON object with the forecast results.

### `GET /`

This is a root endpoint to check if the API is running. It will return a simple status message.

## 🎉 Enjoy! 🎉

We hope you find this repository useful for your time series forecasting needs. Happy coding! 🚀👩‍💻👨‍💻🚀