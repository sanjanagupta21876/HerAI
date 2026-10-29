"""Model training, evaluation, pipeline construction, and persistence for HERAI."""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from herai.config import (
    MODEL_RIDGE_PATH,
    MODEL_RIDGE_37_PATH,
    TARGET_COLUMN,
    EXCLUDED_FROM_TRAINING,
    SUSPICIOUS_OR_LAB_FEATURES,
    FEATURES_37,
)


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Partition DataFrame columns into numerical and categorical feature names."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return num_cols, cat_cols


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    model_type: str = "ridge",
    alpha: float = 1.0,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Pipeline:
    """Build a leakage-free Scikit-learn Pipeline with preprocessing and regressor.
    
    Preprocessing:
    - Numerical: SimpleImputer(strategy='median') + StandardScaler (for Ridge)
    - Categorical: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
    """
    if model_type == "ridge":
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", num_pipeline, numeric_features),
        ("categorical", cat_pipeline, categorical_features),
    ])

    if model_type == "ridge":
        regressor = Ridge(alpha=alpha, random_state=random_state)
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def split_by_user(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split dataset by user_id so no user appears in both train and test splits."""
    clean_df = df.dropna(subset=[target_col]).copy()

    if feature_cols is None:
        feature_cols = [c for c in clean_df.columns if c not in EXCLUDED_FROM_TRAINING]

    X = clean_df[feature_cols]
    y = clean_df[target_col]
    groups = clean_df["user_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Fit pipeline and compute standard regression metrics (MAE, RMSE, R²)."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def save_model(pipeline: Pipeline, file_path: Path) -> None:
    """Save pipeline to disk using joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, file_path)


def load_model(file_path: Path) -> Pipeline:
    """Load pipeline from disk."""
    if not file_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {file_path}")
    return joblib.load(file_path)

