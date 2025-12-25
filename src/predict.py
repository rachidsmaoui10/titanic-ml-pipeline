import joblib
import pandas as pd

from src.data_preprocessing import clean_data
from src.feature_engineering import add_features

MODEL_PATH = "outputs/model.joblib"

def predict_from_raw(df_raw, threshold=0.3):
    """
    Predict survival from raw passenger dataframe.
    """
    model = joblib.load(MODEL_PATH)

    df = clean_data(df_raw)
    X, _ = add_features(df)

    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)
