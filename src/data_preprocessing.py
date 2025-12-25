import pandas as pd
import os

def load_data(project_root):
    path = os.path.join(project_root, "data", "titanic.csv")
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop(columns=["Cabin"])
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Pclass"] = df["Pclass"].astype("category")
    return df

