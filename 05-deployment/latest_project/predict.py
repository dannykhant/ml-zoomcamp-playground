import pickle

import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal


app = FastAPI(title="predict")

with open("model.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


class Customer(BaseModel):
    # Numeric
    tenure: float
    monthlycharges: float
    totalcharges: float

    # Categorical
    gender: Literal["male", "female"]
    seniorcitizen: Literal["0", "1"]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["fiber_optic", "dsl", "no"]
    onlinesecurity: Literal["yes", "no", "no_internet_service"]
    onlinebackup: Literal["yes", "no", "no_internet_service"]
    deviceprotection: Literal["yes", "no", "no_internet_service"]
    techsupport: Literal["yes", "no", "no_internet_service"]
    streamingtv: Literal["yes", "no", "no_internet_service"]
    streamingmovies: Literal["yes", "no", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)"
    ]


class ChurnResponse(BaseModel):
    churn_proba: float
    churn: bool


@app.post("/predict")
def predict(customer: Customer) -> ChurnResponse:
    y_pred = pipeline.predict_proba(customer.model_dump())[0, 1]
    result = {
        "churn_proba": y_pred,
        "churn": y_pred >= 0.5
    }
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port= 9696)
