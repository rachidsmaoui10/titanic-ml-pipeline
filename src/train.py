from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import add_features
import joblib

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = load_and_clean_data(URL)
X, y = add_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "outputs/model.joblib")
