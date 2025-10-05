# app/models/request.py
from pydantic import BaseModel, Field, condecimal
from typing import Literal

PositiveDecimal = condecimal(gt=0)

class RawInputSchema(BaseModel):
    autoID: str = Field(..., example="7590-VHVEG")

    SeniorCity: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    Service1: Literal["Yes", "No"]
    Service2: Literal["Yes", "No", "No phone service"]

    Security: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]

    PaperlessBilling: Literal["Yes", "No"]

    Contract: Literal["Month-to-month", "One year", "Two year"]

    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    Charges: PositiveDecimal = Field(..., example=56.95)
