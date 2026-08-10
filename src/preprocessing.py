"""Data loading and preprocessing for the cardiovascular risk pipeline.

Dataset: UCI Heart Disease (Cleveland), 303 patients, 13 clinical features.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/heart_disease.csv"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def binarize_target(df: pd.DataFrame) -> pd.Series:
    """The raw 'num' column encodes disease severity 0-4.

    We collapse it to a binary "at risk" label (1 = any diagnosed
    narrowing, 0 = none) since the resume/business framing is risk
    classification, not severity staging.
    """
    return (df["num"] > 0).astype(int)


def build_preprocessing_pipeline():
    """Impute missing values, then scale.

    'ca' and 'thal' have a handful of missing values in the raw
    Cleveland data (angiography and thallium-scan results that
    weren't recorded for every patient). Median imputation is used
    instead of dropping rows, since dropping would cost ~2% of an
    already-small (303-row) dataset.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    return imputer, scaler


def load_and_split(path: str = RAW_PATH, test_size: float = 0.2, random_state: int = 42):
    df = load_raw(path)
    X = df[FEATURE_COLUMNS].copy()
    y = binarize_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    imputer, scaler = build_preprocessing_pipeline()
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), imputer, scaler
