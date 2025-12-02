# Project Report: End-to-End Stock Market Prediction System

## 1. Introduction
This project aims to build a production-grade Machine Learning system for stock market prediction. It leverages modern MLOps tools including **FastAPI** for serving, **Prefect** for orchestration, **Docker** for containerization, and **GitHub Actions** for CI/CD. The system predicts both the future closing price (Regression) and the price direction (Classification).

## 2. System Architecture
The system follows a modular architecture:
- **Data Ingestion**: Fetches daily stock data from Alpha Vantage API.
- **Preprocessing**: Calculates technical indicators (SMA, RSI, MACD).
- **Model Training**: Trains Linear Regression, Random Forest, and K-Means models.
- **Orchestration**: Prefect flows manage the pipeline dependencies and retries.
- **Serving**: FastAPI provides REST endpoints for real-time predictions.
- **Monitoring**: DeepChecks validates data integrity and drift.

## 3. Methodology
### 3.1 Data Pipeline
Data is ingested daily. We compute 20-day and 50-day Simple Moving Averages (SMA), Relative Strength Index (RSI), and MACD.

### 3.2 Model Development
- **Regression**: Predicts `Close` price. Metric: RMSE.
- **Classification**: Predicts `Target Direction` (Up/Down). Metric: Accuracy, F1-Score.
- **Clustering**: Groups stocks by volatility. Metric: Inertia.

### 3.3 Automated Testing
We use **DeepChecks** to ensure:
- No missing values or duplicates.
- Train/Test distributions are similar (Drift detection).

## 4. CI/CD & Containerization
- **Docker**: The application is containerized using a multi-stage build to reduce image size.
- **CI/CD**: GitHub Actions runs linting and unit tests on every push, ensuring code quality.

## 5. Observations & Results
- **Best Model**: Random Forest performed best for direction prediction with an accuracy of ~55% (baseline).
- **Data Quality**: Alpha Vantage data is generally clean, but occasional missing days were handled by forward filling.
- **Orchestration**: Prefect significantly improved reliability by handling API rate limits via retries.

## 6. Future Work
- Integrate a real database (PostgreSQL) instead of CSV files.
- Deploy to a cloud provider (AWS/GCP).
- Implement more advanced Deep Learning models (LSTM/Transformer).
