# ML API Project  

This repository is an exercise that demonstrates the full life cycle of an ML project, covering everything from experimentation to model deployment.  

It is divided into two main components that support demand analysis and prediction:  

1. **API (FastAPI)** in `app/` which exposes classification and regression models.  
2. **Laboratories (LAB_ML)** in `LAB_ML/` containing notebooks, data, and results from the training processes.  

---

## API Introduction  

The API works as the **backend** for a frontend application through which the model is consumed. In this app, the user selects parameters (e.g., service type, “Yes/No” flags, payment method, etc.) from dropdown lists and sends this data in JSON format. The purpose of this service is to:  

- **Automatically preprocess** that data (encodings, flags, feature engineering).  
- **Classify** the case into one of two codes (`Alpha` or `Betha`).  
- **Record the probability** associated with the predicted label.  
- **Predict numeric demand** for the given feature set.  

Finally, the API returns a JSON response:  

```
{
  "autoID": "id_unico",
  "class_pred": "Alpha" | "Betha",
  "class_prob": 0.0 - 1.0,
  "demand_pred": 0.0 - …
}
```

This way, the frontend receives in real time both the **classification code (Alpha/Betha)** and the **estimated demand value**.  

## 1. General Repository Structure  

```
C:.
│   Dockerfile
│   README.md
│   requirements.txt
│   to_predict.csv
│
├───app/                       ← API source code
│   │   main.py
│   │   __init__.py
│   │
│   ├───api/
│   │   ├── deps.py
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── endpoints.py   ← POST /v1/predict
│   │       └── __init__.py
│   │
│   ├───core/
│   │   ├── config.py          ← Environment variables (Pydantic)
│   │   ├── logging.py         ← Log configuration
│   │   ├── version.py         ← API version
│   │   └── __init__.py
│   │
│   ├───models/
│   │   ├── request.py         ← Input Pydantic schema
│   │   └── response.py        ← Output Pydantic schema
│   │
│   └───services/
│       ├── loader.py          ← Loads pipelines (.pkl)
│       ├── predictor.py       ← Prediction logic (clas/ + reg/) 
│       ├── preprocessor.py    ← JSON-to-DataFrame transformations
│       └── __init__.py
│
├───models/                    ← Serialized pipelines used by the API
│       catboost_reg_pipeline.pkl  # Full regression pipeline
│       logreg_clf_pipeline.pkl    # Full classification pipeline
│
├───LAB_ML/       ← Training and forecasting lab
│   ├───Clasificacion_Regresion/
│   │   ├───models/
│   │   │       catboost_reg_pipeline.joblib
│   │   │       logistic_clf_pipeline.joblib
│   │   │       txt_info_modelos.json
│   │   │
│   │   ├───notebooks/
│   │   │       EDA_&_preprosesado.ipynb
│   │   │       ML01_experiment_class_regresion.ipynb
│   │   │
│   │   ├───raw_data/
│   │   │       dataset_alpha_betha.csv
│   │   │
│   │   └───transformed_data/
│   │           dataset_A_raw.csv
│   │           dataset_B_addon_sum.csv
│   │           dataset_C_contractx.csv
│   │           dataset_D_fulleng.csv

```

- **root/**  
  - `Dockerfile`: instructions to package the API in Docker.  
  - `requirements.txt`: Python dependencies required by the API.  
  - `to_predict.csv`: CSV file with records for which predictions were requested.  
- **app/**  
  - Contains all the FastAPI logic.  
- **models/**  
  - Serialized pipelines (`.pkl`) used by the API for inference.  
- **LAB_ML/**  
  - **Clasificacion_Regresion/:** notebooks, data, and artifacts showing how the regression and classification pipelines were developed.  

## 2. API (FastAPI)  
### **2.1. Overview**  
- **Main endpoint:** ``POST /v1/predict.``  

- **Input:** JSON with fields validated by Pydantic (see ``app/models/request.py``).  

- **Process:**  

    1. ``preprocessor.py`` converts the JSON into a ``DataFrame`` ready for the pipelines:  

        - Creates binary columns, replaces text (“No internet service” → “No”, “No phone service” → “No”), generates ``Contract_months``, ``InternetService``, ``AutoPayment_flag`` and ``PaymentMethod_simple``.  

        - For classification, it adds ``TotalAddOns``, ``Charges_per_AddOn`` and ``Contract_x_Charges``.  

    2. ``predictor.py`` applies:  

        - **Classification** (``logreg_clf_pipeline.pkl``): returns ``0`` or ``1`` → mapped to ``"Alpha"``/``"Betha"``. It also extracts the probability of the predicted class.  

        - **Regression** (``catboost_reg_pipeline.pkl``): returns a numeric demand value.  

    3. Returns a JSON like:  

```json
{
  "autoID": "7590-VHVEG",
  "class_pred": "Alpha",
  "class_prob": 0.81,
  "demand_pred": 55.2
}

```
### 2.2. Dependencies  
Listed in `requirements.txt`.  

---

## 3. API Deployment with Docker  

### 3.1. Build the image  
From the root (`C:\…\ML_API_PROJECT`), run:  

```bash
docker build -t ml_api_ml:latest .
```
- Uses ``python:3.11-slim`` as the base.
- Installs packages from ``requirements.txt.``
- Copies the ``app/`` folder and the``models/`` pipelines into the image.

### 3.2. Run the container
```bash
docker run -d \
  --name ml_api_container \
  -p 8000:8000 \
  ml_api_ml:latest
```
- ``-d`` runs the container in the background
- ``--name ml_api_container``: sets the container name.
- ``-p 8000:8000``: maps port 8000 of the container to port 8000 on the host.

### 3.3. Verify and test
 **1.** Check the running container:
```bash
docker ps
```
**2.** Access Swagger UI:
```bash
http://127.0.0.1:8000/docs
```
**3.** Example with ``curl``:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
        "autoID": "7590-VHVEG",
        "SeniorCity": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Service1": "Yes",
        "Service2": "No phone service",
        "Security": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "PaperlessBilling": "Yes",
        "Contract": "Two year",
        "PaymentMethod": "Electronic check",
        "Charges": 56.95
      }'
```
Expected response:
```json
{
  "autoID": "7590-VHVEG",
  "class_pred": "Alpha",
  "class_prob": 0.81,
  "demand_pred": 55.2
}
```
## 4. Local Execution (without Docker)

If you prefer not to use Docker, follow these steps:

**1.** Create and activate a Conda environment with Python 3.11:
```bash
conda create -n ml_api python=3.11 -y
conda activate ml_api
```

**2.** Install dependencies:
```bash
pip install -r requirements.txt
pip install scikit-learn==1.6.1 pydantic-settings
```
**3.** Start the server:
```bash
uvicorn app.main:app --reload
```
**4.** Open ``http://127.0.0.1:8000/docs`` to test the ``POST /v1/predict`` endpoint.

## 5. LAB_ML: Training
Install ``LAB_ML/`` you will find the original development notebooks and datasets:
- **Clasificacion_Regresion/**
    - ``models/``: ``.joblib`` pipelines and metadata in``txt_info_modelos.json``.
    - ``notebooks/:``
        - ``EDA_&_preprosesado.ipynb``: cleaning, EDA, and generation of transformed CSVs.

        - ``ML01_experiment_class_regresion.ipynb``: comparison of blocks A–D, final tuning of **CatBoost** and **LogisticRegression**, metrics extraction.

- ``raw_data/dataset_alpha_betha.csv``: original classification data.
- ``transformed_data/``: CSVs for each feature engineering block (A, B, C, D).

> **Note:** Everything in ``LAB_ML/`` is provided for reference and traceability of how the models now used by the API were created. The API only consumes the serialized files in ``models/``

## 6. Additional Root Files

- ``to_predict.csv``: CSV with records for which predictions were generated using the API (test values).

## 7. Author and Contact
**Juan Felipe Quinto Ríos**
| Data Scientist          quintoriosjuanfelip@gmail.com
 https://www.linkedin.com/in/juan-felipe-quinto-rios

 
