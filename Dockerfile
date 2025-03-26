# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy required files
COPY main.py .
COPY xgboost_model.pkl .
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
