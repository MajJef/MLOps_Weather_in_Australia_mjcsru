# Use a lightweight Python image
FROM python:3.9-slim  

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (to leverage Docker caching)
COPY requirements.txt .  

# Install dependencies (with no cache to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt  

# Copy the rest of the application files
COPY . .

# Expose the FastAPI server port  
EXPOSE 8000  

# Start the FastAPI server when the container runs  
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]  

