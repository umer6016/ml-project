# Demonstration Video Plan (5-10 minutes)

## 1. Introduction (1 min)
- **Goal**: Introduce the Stock Market Prediction System.
- **Visual**: Slide with project title and architecture diagram.
- **Script**: "Welcome to the End-to-End Stock Market Prediction System. This project integrates FastAPI, Prefect, Docker, and ML models to predict stock prices and trends."

## 2. System Architecture & Code Walkthrough (2 mins)
- **Goal**: Show the code structure and key components.
- **Visual**: VS Code showing `src/` folder, `Dockerfile`, and `flows.py`.
- **Script**: "Here is the project structure. We have data ingestion using Alpha Vantage, feature engineering, and training pipelines orchestrated by Prefect."

## 3. Data Ingestion & Orchestration (2 mins)
- **Goal**: Demonstrate Prefect flow.
- **Visual**: Run `python src/orchestration/flows.py`. Show terminal output and Discord notification.
- **Script**: "I'm triggering the data ingestion flow. You can see it fetching data, processing it, and sending a notification to Discord upon completion."

## 4. Model Training & Validation (2 mins)
- **Goal**: Show DeepChecks and Model Artifacts.
- **Visual**: Open `reports/data_integrity.html` and `metrics.json`.
- **Script**: "We use DeepChecks to validate data integrity. Here is the generated report. We also log model metrics like RMSE and Accuracy."

## 5. Deployment & API Demo (2 mins)
- **Goal**: Show the running application.
- **Visual**: Run `docker-compose up`. Open Swagger UI (`localhost:8000/docs`). Make a prediction request.
- **Script**: "Now let's run the system with Docker. The API is up. I'll send a request to predict the price of AAPL based on recent indicators."

## 6. Conclusion (1 min)
- **Goal**: Wrap up.
- **Visual**: Summary slide.
- **Script**: "In summary, we've built a robust, containerized ML system with automated testing and CI/CD."
