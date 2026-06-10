import os
import time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder

from utils.feature_extractor import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "sample_data", "incidents.csv")

risk_model = None
isolation_model = None
risk_encoder = None
feature_columns = None


def train_prediction_models():
    global risk_model
    global isolation_model
    global risk_encoder
    global feature_columns

    df = pd.read_csv(CSV_PATH)

    feature_rows = [
        extract_features(log)
        for log in df["log"]
    ]

    X = pd.DataFrame(feature_rows)
    feature_columns = X.columns.tolist()

    risk_encoder = LabelEncoder()
    y_risk = risk_encoder.fit_transform(df["risk_level"])

    risk_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    risk_model.fit(X, y_risk)

    isolation_model = IsolationForest(
        n_estimators=150,
        contamination=0.25,
        random_state=42
    )

    isolation_model.fit(X)


def prediction_agent(log):
    global risk_model
    global isolation_model
    global risk_encoder
    global feature_columns

    start = time.time()

    if risk_model is None or isolation_model is None:
        train_prediction_models()

    features = extract_features(log)
    X = pd.DataFrame([features])
    X = X[feature_columns]

    risk_pred = risk_model.predict(X)[0]
    risk_level = risk_encoder.inverse_transform([risk_pred])[0]

    risk_confidence = max(risk_model.predict_proba(X)[0])

    anomaly_raw = isolation_model.predict(X)[0]
    anomaly_score = isolation_model.decision_function(X)[0]

    anomaly_detected = anomaly_raw == -1

    if risk_level in ["Critical", "High"]:
        likely_future_issue = "Service disruption or operational outage"
    elif risk_level == "Medium":
        likely_future_issue = "Service degradation if unresolved"
    else:
        likely_future_issue = "No major issue expected"

    latency = round(time.time() - start, 3)

    return {
        "response": {
            "anomaly_detected": bool(anomaly_detected),
            "risk_level": risk_level,
            "likely_future_issue": likely_future_issue,
            "confidence_score": round(float(risk_confidence), 2),
            "isolation_forest_score": round(float(anomaly_score), 4),
            "model_type": "RandomForestClassifier + IsolationForest"
        },
        "latency": latency,
        "input_tokens": 0,
        "output_tokens": 0
    }
