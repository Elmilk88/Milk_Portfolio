from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

app = FastAPI()

# Define the input schema
class CreditCardData(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.post("/predict")
def predict(data: CreditCardData):
    # Convert the input into a DataFrame
    input_data = [[
        data.V1, data.V2, data.V3, data.V4, data.V5, data.V6,
        data.V7, data.V8, data.V9, data.V10, data.V11, data.V12,
        data.V13, data.V14, data.V15, data.V16, data.V17, data.V18,
        data.V19, data.V20, data.V21, data.V22, data.V23,
        data.V24, data.V25, data.V26, data.V27, data.V28
    ]]

    columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
               'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
               'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

    df = pd.DataFrame(input_data, columns=columns)

    # Preprocessing (scaling) and prediction
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}
