import pandas as pd

def add_features(df):
    df = df.copy()

    df["IsChild"]  = (df["Age"] < 15).astype(int)
    df["IsMother"] = ((df["Sex"] == "female") & (df["Parch"] > 0)).astype(int)
    df["IsAlone"]  = ((df["SibSp"] == 0) & (df["Parch"] == 0)).astype(int)

    X = df[["Sex", "Pclass", "Age", "IsChild", "IsMother", "IsAlone"]]
    y = df["Survived"]

    X = pd.get_dummies(X, drop_first=True)
    return X, y


