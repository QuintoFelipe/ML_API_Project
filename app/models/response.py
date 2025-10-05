# app/models/response.py
from pydantic import BaseModel
from typing import Optional

class PredictionSchema(BaseModel):
    autoID: str
    class_pred: str
    class_prob: float 
    demand_pred: float
