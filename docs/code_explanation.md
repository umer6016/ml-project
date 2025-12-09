# Codebase Explanation & Walkthrough

This document explains how the Stock Prediction System works under the hood. It is designed to help you understand the code so you can explain it during your presentation.

## 1. The Big Picture
The system is a pipeline that moves data through these stages:
1.  **Ingestion**: Fetch raw data from the internet (Alpha Vantage).
2.  **Processing**: Clean data and calculate math features (SMA, RSI).
3.  **Training**: Teach the AI models using the processed data.
4.  **Serving**: Make the models available via an API for predictions.

## 2. Key Files Explained

### A. `src/orchestration/flows.py` (The Conductor)
This is the "brain" of the training pipeline. It uses **Prefect** to organize tasks.
-   **`@task`**: Decorators that turn python functions into managed tasks (with retries/logging).
-   **`main_pipeline`**: The main function that calls everything in order:
    1.  `fetch_daily_data`: Downloads CSVs.
    2.  `process_data`: Adds technical indicators.
    3.  `train_and_evaluate`: Trains the models and saves them.

### B. `src/api/main.py` (The Web Server)
This is the **FastAPI** application that serves the models.
-   **`@app.on_event("startup")`**: When the server starts, it looks into the `models/` folder and loads the `.pkl` files into memory.
-   **`/predict/price`**: An endpoint that takes features (SMA, RSI, etc.) and uses the loaded `regression_model` to predict the next closing price.

### C. `src/processing/features.py` (The Math)
This file contains the logic for financial indicators.
-   **`calculate_sma`**: A simple rolling average.
-   **`calculate_rsi`**: A momentum indicator measuring the speed of price changes.
-   **`process_data`**: Combines these functions to transform raw "Close" prices into a dataset ready for ML.

### D. `docker-compose.yml` (The Infrastructure)
This file tells Docker how to run the system.
-   **`api`**: Builds your code and runs the FastAPI server.
-   **`prefect-server`**: Runs the dashboard where you see your pipelines.
-   **`postgres`**: A database used by Prefect to store flow history.

## 3. How the AI Works
We use **Scikit-Learn** for the machine learning models (defined in `src/models/train.py`).

1.  **Regression (LinearRegression)**:
    -   **Goal**: Predict the exact price (e.g., $150.25).
    -   **How**: Draws a straight line through the data points to minimize error.

2.  **Classification (RandomForest)**:
    -   **Goal**: Predict direction (UP or DOWN).
    -   **How**: Uses multiple "decision trees" (like a flowchart of yes/no questions) to vote on the outcome.

## 4. Common Questions & Answers

**Q: Why do we need Docker?**
A: It ensures the code runs exactly the same on your computer, my computer, and the cloud, by packaging all dependencies (Python, Pandas, etc.) into a "container".

**Q: Why Prefect?**
A: If the API fails or data is missing, Prefect handles retries and alerts. It turns a simple script into a robust pipeline.

**Q: What is Deepchecks?**
A: It's a testing tool that looks at our data to make sure it's not "drifting" (changing significantly) from what the model expects, ensuring our predictions remain accurate.
