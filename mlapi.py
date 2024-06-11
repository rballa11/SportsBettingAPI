from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    spread_favorite: int
    weather_temperature: float
    weather_humidity: float
    weather_windspeed: float
    weather_detail: str

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/")
async def scoring_endpoint(scoringitem: Item):
    scoring_df = pd.DataFrame([scoringitem.values()], columns=scoringitem.keys())
    yhat = model.predict(scoring_df)
    return {"predicted_result": yhat }