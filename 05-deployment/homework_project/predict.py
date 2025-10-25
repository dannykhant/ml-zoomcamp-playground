import pickle

import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel


class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


class PredictResponse(BaseModel):
    probability: float
    

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI(title="predict")


@app.post("/predict")
def predict(client: Client) -> PredictResponse:
    result = pipeline.predict_proba(client.model_dump())[0, 1]
    return {
        "probability": result
    }
