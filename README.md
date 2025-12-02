# End-to-End Stock Prediction System

A comprehensive machine learning system for stock market prediction, featuring data ingestion, processing, model training, and deployment.

## Features
- **Data Ingestion**: Fetches daily stock data from Alpha Vantage.
- **Data Processing**: Calculates technical indicators (SMA, RSI, MACD).
- **Machine Learning**:
    - **Regression**: Predicts next day's closing price.
    - **Classification**: Predicts price direction (Up/Down).
    - **Clustering**: Groups market regimes based on volatility.
    - **PCA**: Dimensionality reduction for feature analysis.
- **Orchestration**: Prefect workflows for automated pipelines.
- **Validation**: Deepchecks for data integrity and drift detection.
- **Deployment**: Dockerized FastAPI application with Postgres database.
- **CI/CD**: GitHub Actions for testing and deployment.

## Tech Stack
- **Language**: Python 3.9
- **Frameworks**: FastAPI, Prefect, Scikit-Learn, Pandas
- **Tools**: Docker, Docker Compose, Deepchecks, Pytest
- **Database**: PostgreSQL

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Alpha Vantage API Key (set in `.env`)

### Installation
1. Clone the repository.
2. Create a `.env` file:
    ```bash
    cp .env.example .env
    # Edit .env with your API key
    ```
3. Build and start services:
    ```bash
    docker-compose up --build -d
    ```

### Usage
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prefect UI**: [http://localhost:4200](http://localhost:4200)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Running Tests
```bash
pip install -e .[dev]
python -m pytest tests/
```

### Training Models
To train models manually:
```bash
python src/orchestration/flows.py
```
