import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv("../data/raw/train.csv")


def process_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.drop(columns=["date"])


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["street", "country"]
    return df.drop(columns=drop_cols)


def split_target(df: pd.DataFrame, target: str = "price"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def get_feature_types(X: pd.DataFrame):
    num_features = X.select_dtypes(include=np.number).columns
    cat_features = X.select_dtypes(exclude=np.number).columns
    return num_features, cat_features


def build_preprocessor(num_features, cat_features):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(data_path: str):
    df = load_data(data_path)
    df = process_date(df)
    df = drop_columns(df)

    X, y = split_target(df)
    num_features, cat_features = get_feature_types(X)

    preprocessor = build_preprocessor(num_features, cat_features)
    X_train, X_val, y_train, y_val = split_data(X, y)

    return X_train, X_val, y_train, y_val, preprocessor
