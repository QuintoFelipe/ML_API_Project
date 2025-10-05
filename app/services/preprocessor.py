
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd

# ──────────────────────────────────────────
# Constantes (igual que antes) …
# ──────────────────────────────────────────
_SERVICES_WITH_REDUNDANT_CAT = [
    "Security", "TechSupport", "OnlineBackup", "DeviceProtection"
]
_YES_NO_COLUMNS = [
    "Partner", "Service1", "TechSupport", "PaperlessBilling",
    "Security", "DeviceProtection", "Dependents",
    "OnlineBackup", "Service2"
]
_CONTRACT_TO_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}
_AUTOPAY_METHODS = {
    "Bank transfer (automatic)", "Credit card (automatic)"
}
_PAYMENT_SIMPLE_MAP = {
    "Bank transfer (automatic)": "Automatic",
    "Credit card (automatic)":  "Automatic",
    "Electronic check":         "Electronic check",
    "Mailed check":             "Mailed check",
}

_BASE_COL_ORDER = [
    "SeniorCity", "Partner", "Dependents",
    "Service1", "Service2", "Security",
    "OnlineBackup", "DeviceProtection",
    "TechSupport", "PaperlessBilling",
    "InternetService", "Charges",
    "Contract_months", "AutoPayment_flag",
    "PaymentMethod_simple",
]

# ──────────────────────────────────────────
# 1)  BLOQUE BASE  (común)
# ──────────────────────────────────────────
def _base_transform(raw: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([raw]).copy()

    # InternetService
    df["InternetService"] = df["Security"].apply(
        lambda x: 0 if x == "No internet service" else 1
    )
    # Uniformar categorías
    df[_SERVICES_WITH_REDUNDANT_CAT] = df[_SERVICES_WITH_REDUNDANT_CAT].replace(
        "No internet service", "No"
    )
    df["Service2"] = df["Service2"].replace("No phone service", "No")

    # Yes/No → 1/0
    df[_YES_NO_COLUMNS] = df[_YES_NO_COLUMNS].replace({"Yes": 1, "No": 0})

    # Contract → meses
    df["Contract_months"] = df["Contract"].map(_CONTRACT_TO_MONTHS)

    # Flag autopago
    df["AutoPayment_flag"] = df["PaymentMethod"].isin(_AUTOPAY_METHODS).astype(int)

    # Método simple
    df["PaymentMethod_simple"] = df["PaymentMethod"].replace(_PAYMENT_SIMPLE_MAP)

    return df

# ──────────────────────────────────────────
# 2)  TRANSFORMACIÓN PARA REGRESIÓN
# ──────────────────────────────────────────
def transform_regression(raw: Dict[str, Any]) -> pd.DataFrame:
    df = _base_transform(raw)
    df_final = df[_BASE_COL_ORDER]        # solo columnas base
    return df_final


# ──────────────────────────────────────────
# 3)  TRANSFORMACIÓN PARA CLASIFICACIÓN
# ──────────────────────────────────────────
def transform_classification(raw: Dict[str, Any]) -> pd.DataFrame:
    df = _base_transform(raw)

    # —— BLOQUE DE FEATURES DE INGENIERÍA ——
    df["TotalAddOns"] = (
        df["Security"] +
        df["OnlineBackup"] +
        df["DeviceProtection"] +
        df["TechSupport"]
    )

    # Evita división por 0
    df["Charges_per_AddOn"] = df["Charges"] / (df["TotalAddOns"] + 1)
    df["Contract_x_Charges"] = df["Contract_months"] * df["Charges"]

    cls_cols = _BASE_COL_ORDER + [
        "TotalAddOns",
        "Charges_per_AddOn",
        "Contract_x_Charges",
    ]

    df_final = df[cls_cols]
    return df_final

