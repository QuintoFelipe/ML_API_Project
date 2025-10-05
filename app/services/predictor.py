# app/services/predictor.py
from typing import Dict, Any

from .preprocessor import transform_regression, transform_classification
from .loader import get_regression_model, get_classification_model

# Diccionario que mapea la salida numérica del modelo (0/1) 
# a las etiquetas de texto que queremos exponer en la API
_CLASS_MAPPING = {
    0: "Alpha",
    1: "Betha",
}

def predict(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recibe el JSON crudo (raw_input) ya validado por Pydantic,
    aplica preprocesamiento para clasificación/regresión, 
    ejecuta ambos modelos y devuelve:
      {
        "class_pred": "Alpha" o "Betha",
        "class_prob": float | None,
        "demand_pred": float
      }
    """
    # ——— 1) PREDICCIÓN PARA CLASIFICACIÓN ———
    cls_df = transform_classification(raw_input)
    cls_model = get_classification_model()

    # El modelo devuelve 0 o 1:
    raw_class = cls_model.predict(cls_df)[0]            # 0 ó 1
    # Convertimos a “Alpha” / “Betha” usando el mapping
    class_pred = _CLASS_MAPPING.get(int(raw_class), str(raw_class))

    # Si el clasificador tiene predict_proba, sacamos la prob de la clase “1”
    class_prob = None
    if hasattr(cls_model, "predict_proba"):
        proba = cls_model.predict_proba(cls_df)[0]       # ej: [0.35, 0.65]
        class_prob = float(proba[1])                     # prob de “1” (Betha)

    # ——— 2) PREDICCIÓN PARA REGRESIÓN ———
    reg_df = transform_regression(raw_input)
    reg_model = get_regression_model()

    demand_pred = float(reg_model.predict(reg_df)[0])

    return {
        "class_pred": class_pred,
        "class_prob": class_prob,
        "demand_pred": demand_pred,
    }
