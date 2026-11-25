"""Real-time scoring API.

Run:  uvicorn api:app --reload
Then POST a payment to http://127.0.0.1:8000/score (docs at /docs).
"""
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from clearflag import ClearFlag

app = FastAPI(title="ClearFlag", description="Real-time UPI fraud alerts that explain themselves")
engine = ClearFlag("model")


class Payment(BaseModel):
    step: int = Field(..., description="Hour of the simulation (time of payment)", examples=[3])
    type: str = Field(..., examples=["TRANSFER"])
    amount: float = Field(..., examples=[48000])
    oldbalanceOrg: float = Field(..., examples=[50000])
    newbalanceOrig: float = Field(..., examples=[2000])
    oldbalanceDest: float = Field(..., examples=[0])
    newbalanceDest: float = Field(..., examples=[0])


@app.post("/score")
def score(p: Payment):
    return engine.score(pd.DataFrame([p.model_dump()]))[0]


@app.get("/health")
def health():
    return {"status": "ok"}
