import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import add_features

# 1. Load and preprocess data
df = load_and_clean_data()
X, y = add_features(df)

# 2. Same train / test split as during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Load trained model
model = joblib.load("outputs/model.joblib")

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

