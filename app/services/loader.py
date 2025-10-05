# app/services/loader.py
from functools import lru_cache
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]   # ml_api_project/

@lru_cache
def get_regression_model():
    return joblib.load(BASE_DIR / "models" / "catboost_reg_pipeline.pkl")

@lru_cache
def get_classification_model():
    return joblib.load(BASE_DIR / "models" / "logreg_clf_pipeline.pkl")
