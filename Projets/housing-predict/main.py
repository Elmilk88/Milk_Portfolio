from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle et le préprocesseur
model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

app = FastAPI()

# Définir le schéma d’entrée
class HouseData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

@app.post("/predict")
def predict(data: HouseData):
    # Convertir les données en DataFrame
    input_data = [[
        data.longitude, data.latitude, data.housing_median_age,
        data.total_rooms, data.total_bedrooms, data.population,
        data.households, data.median_income, data.ocean_proximity
    ]]
    
    import pandas as pd
    columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
               'total_bedrooms', 'population', 'households', 'median_income',
               'ocean_proximity']
    df = pd.DataFrame(input_data, columns=columns)

    # Prétraitement et prédiction
    X = preprocessor.transform(df)
    prediction = model.predict(X)
    
    return {"prediction": float(prediction[0])}
