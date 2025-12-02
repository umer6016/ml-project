# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Install dependencies
RUN pip install "numpy<2.0" pandas
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ src/
COPY tests/ tests/
COPY .env.example .env

# Create directories for data and models
RUN mkdir -p data/processed models reports

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
