# app/main.py
from fastapi import FastAPI

from .core.logging import configure_logging
from .api.v1.endpoints import router as v1_router
from .core.config import settings

configure_logging()           # inicializa logs antes de crear la app

app = FastAPI(
    title="Demand & Class Predictor API",
    version=settings.VERSION,
)

app.include_router(v1_router)
