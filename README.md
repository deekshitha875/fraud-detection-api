# Fraud Detection API

A REST API built with Flask that uses an unsupervised machine learning model (Isolation Forest) to flag potentially fraudulent credit card transactions in real time.

---

## Overview

E-commerce and payment platforms process thousands of transactions per second, and only a tiny fraction are fraudulent — making it a highly imbalanced problem. This project tackles that with **unsupervised anomaly detection**, so the model can flag suspicious transactions without needing large amounts of labeled fraud data.

The model is trained on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 real anonymized European credit card transactions, of which only 492 (0.17%) are fraudulent.

---

## Tech Stack

- **Python 3**
- **Flask** — REST API framework
- **scikit-learn** — Isolation Forest model
- **pandas** — data handling
- **joblib** — model serialization

---

## Project Structure

```
fraud-detection-api/
├── app.py            # Flask application and /predict endpoint
├── fraud_model.pkl   # Pre-trained Isolation Forest model
└── README.md
```

---

## How It Works

1. Transactions are represented by 30 features: `Time`, `Amount`, and `V1`–`V28` (PCA-transformed, anonymized features from the original dataset).
2. An **Isolation Forest** model was trained on the full dataset with `contamination=0.01` (i.e., assuming ~1% of transactions are anomalous), without using the fraud labels — the model learns to isolate outliers based on how easily a point can be separated from the rest of the data.
3. The trained model is serialized with `joblib` and loaded by the Flask API at startup.
4. A client sends transaction data as JSON to `/predict`. The API wraps it into a DataFrame and runs inference.
5. Isolation Forest returns `-1` for anomalies and `1` for normal points — the API converts this into a simple `fraud: true/false` response.

**Model performance on the training set:**

| Metric | Normal (Class 0) | Fraud (Class 1) |
|---|---|---|
| Precision | 1.00 | 0.10 |
| Recall | 0.99 | 0.59 |
| F1-score | 1.00 | 0.17 |

Overall accuracy: 99%. Since fraud makes up only 0.17% of transactions, accuracy alone is misleading — recall on the fraud class (0.59) is the more meaningful metric here, and there's a known precision/recall tradeoff with unsupervised methods on this level of class imbalance. A natural next step would be comparing this against supervised models (e.g. Random Forest with SMOTE oversampling) trained directly on the fraud labels.

---

## Getting Started

### Prerequisites

```bash
python -m pip install flask scikit-learn pandas joblib
```

### Running the API

```bash
python app.py
```

Server starts at `http://127.0.0.1:5000`.

---

## API Reference

### `POST /predict`

**Request**

```http
POST /predict
Content-Type: application/json
```

```json
{
  "Time": 0,
  "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33,
  "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09,
  "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46,
  "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
  "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12,
  "V26": -0.18, "V27": 0.13, "V28": -0.02,
  "Amount": 149.62
}
```

**Response**

```json
{
  "fraud": false
}
```

| Field | Type | Description |
|---|---|---|
| `fraud` | boolean | `true` if the transaction is flagged as anomalous, `false` otherwise |

---

## Example Usage

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"Time": 0, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25, "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
```

**Python:**
```python
import requests

payload = {
    "Time": 0, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33,
    "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46,
    "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12,
    "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62
}
response = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(response.json())
```

---

## Notes

- `fraud_model.pkl` must be in the same directory as `app.py`.
- Input JSON must include all 30 fields (`Time`, `Amount`, `V1`–`V28`) in the schema the model was trained on.
- Not intended for production use as-is — `debug=True` should be disabled and a proper WSGI server (e.g. Gunicorn) used for deployment.

---

## Dataset Credit

Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB Machine Learning Group, hosted on Kaggle.
