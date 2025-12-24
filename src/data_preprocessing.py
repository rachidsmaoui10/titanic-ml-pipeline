import pandas as pd

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_data():
    return pd.read_csv(URL)

def clean_data(df):
    df = df.drop(columns=["cabin"])
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Pclass"] = df["Pclass"].astype("category")
    return df
