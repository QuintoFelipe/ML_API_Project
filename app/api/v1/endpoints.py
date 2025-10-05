# app/api/v1/endpoints.py
from fastapi import APIRouter, Depends
from ...models.request import RawInputSchema
from ...models.response import PredictionSchema
from ...services.predictor import predict
from ...api.deps import get_settings

router = APIRouter(prefix="/v1", tags=["predictions"])

@router.post("/predict", response_model=PredictionSchema)
def make_prediction(
    data: RawInputSchema,
    _settings = Depends(get_settings),   
):
    result = predict(data.dict())
    result["autoID"] = data.autoID      
    return result
