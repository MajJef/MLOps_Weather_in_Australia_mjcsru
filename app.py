from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

# Load the saved model
try:
    model = joblib.load("xgboost_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

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
    try:
        # Ensure the input is a list of numbers
        if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data):
            raise HTTPException(status_code=400, detail="Input must be a list of numbers.")

        prediction = model.predict([np.array(data)])
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

