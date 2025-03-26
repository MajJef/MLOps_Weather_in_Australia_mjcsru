from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load the saved model
model = joblib.load("xgboost_model.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to the Rain Prediction API!"}

@app.post("/predict")
def predict(data: list):
    """
    Send a list of feature values to get a prediction.
    Example:
    {
      "data": [0.1, 0.2, 0.3, 0.4, ...]  # Replace with real features
    }
    """
    prediction = model.predict([np.array(data)])
    return {"prediction": int(prediction[0])}
