import joblib
import numpy as np

def predict_with_threshold(X, threshold=0.3):
    model = joblib.load("outputs/model.joblib")
    proba = model.predict_proba(X)[:,1]
    return (proba >= threshold).astype(int)
