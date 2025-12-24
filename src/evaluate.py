import joblib
from sklearn.metrics import accuracy_score, classification_report
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import add_features
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = load_and_clean_data(URL)
X, y = add_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("outputs/model.joblib")
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
