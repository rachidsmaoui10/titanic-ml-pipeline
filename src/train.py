from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import add_features
import joblib
import os

def main():
    # Load + clean
    df = load_and_clean_data()

    # Feature engineering
    X, y = add_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model in /outputs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    joblib.dump(model, os.path.join(outputs_dir, "model.joblib"))
    print("✅ Model saved to outputs/model.joblib")

if __name__ == "__main__":
    main()

