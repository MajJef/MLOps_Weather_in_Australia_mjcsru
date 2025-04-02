from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load your model
try:
    model = joblib.load("xgboost_model.pkl")  # Or '/app/xgboost_model.pkl' inside Docker
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Define a data model for incoming requests
class InputData(BaseModel):
    data: list[float]  # Expecting a list of numbers

@app.post("/predict")
def predict(input_data: InputData):
    try:
        features = np.array(input_data.data).reshape(1, -1)  # Convert list to NumPy array
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

