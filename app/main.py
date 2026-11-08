"""FastAPI: validation, single + batch prediction, risk level, model info"""
import json
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Churn Prediction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])

bundle = joblib.load("model/churn_model.joblib")
model, THRESHOLD, COLUMNS = bundle["model"], bundle["threshold"], bundle["columns"]


class Customer(BaseModel):
    tenure: int = Field(ge=0, le=120, description="koto mash dhore customer")
    monthly_charges: float = Field(gt=0, le=500)
    total_charges: Optional[float] = Field(default=None, ge=0)  # missing hole pipeline impute kore
    age: Optional[int] = Field(default=None, ge=16, le=110)
    support_calls: int = Field(ge=0, le=100)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber", "None"]
    payment_method: Literal["Card", "Bank transfer", "Mobile banking", "Cash"]


def risk_level(p: float) -> str:
    if p >= max(THRESHOLD, 0.6):
        return "high"
    return "medium" if p >= THRESHOLD else "low"


def score(customers: list[Customer]) -> list[dict]:
    df = pd.DataFrame([c.model_dump() for c in customers])[COLUMNS]
    probs = model.predict_proba(df)[:, 1]
    return [{"churn_probability": round(float(p), 4),
             "will_churn": bool(p >= THRESHOLD),
             "risk_level": risk_level(float(p))} for p in probs]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return json.load(open("model/metrics.json"))


@app.post("/predict")
def predict(customer: Customer):
    return score([customer])[0]


@app.post("/predict-batch")
def predict_batch(customers: list[Customer]):
    return {"count": len(customers), "results": score(customers)}
