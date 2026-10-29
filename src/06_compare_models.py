"""HERAI Model Comparison Pipeline.

Evaluates candidate regression models (Ridge 42-feature, Random Forest, and Ridge 37-feature)
using user-grouped splitting, and saves the 37-feature production model to
models/herai_ridge_37_features.pkl.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from herai.config import (
    ENGINEERED_DATA_PATH,
    MODEL_RIDGE_37_PATH,
    TARGET_COLUMN,
    EXCLUDED_FROM_TRAINING,
    SUSPICIOUS_OR_LAB_FEATURES,
    FEATURES_37,
)
from herai.models import (
    split_by_user,
    get_feature_types,
    build_pipeline,
    evaluate_model,
    save_model,
)


def run_comparison() -> pd.DataFrame:
    """Train and compare Ridge, Random Forest, and Reduced 37-Feature Ridge.
    
    Returns:
        pd.DataFrame: Table comparing candidate models by MAE, RMSE, and R2.
    """
    print("\n========== HERAI MODEL COMPARISON ==========")
    print(f"Loading data from: {ENGINEERED_DATA_PATH}")

    if not ENGINEERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {ENGINEERED_DATA_PATH}. Run src/03_create_features.py first."
        )

    df = pd.read_csv(ENGINEERED_DATA_PATH)
    clean_df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # All candidate features (42 features)
    features_42 = [c for c in clean_df.columns if c not in EXCLUDED_FROM_TRAINING]
    # Reduced features (37 features)
    features_37 = [c for c in features_42 if c not in SUSPICIOUS_OR_LAB_FEATURES]

    # Split dataset by user (80/20)
    X_train_42, X_test_42, y_train, y_test, g_train, g_test = split_by_user(
        clean_df, feature_cols=features_42, test_size=0.2, random_state=42
    )
    X_train_37 = X_train_42[features_37].copy()
    X_test_37 = X_test_42[features_37].copy()

    print(f"Full feature count: {len(features_42)}")
    print(f"Reduced feature count: {len(features_37)}")
    print(f"Training set: {len(X_train_42)} rows, Testing set: {len(X_test_42)} rows.")

    results = []

    # 1. Ridge Regression (42 features)
    print("\nTraining Ridge Regression (42 features)...")
    num_42, cat_42 = get_feature_types(X_train_42)
    ridge_42 = build_pipeline(num_42, cat_42, model_type="ridge", alpha=1.0)
    metrics_ridge_42 = evaluate_model(ridge_42, X_train_42, y_train, X_test_42, y_test)
    results.append({"Model": "Ridge Regression (42 features)", **metrics_ridge_42})

    # 2. Random Forest (42 features, optimized leaf/trees for benchmark)
    print("Training Random Forest (42 features)...")
    rf = build_pipeline(num_42, cat_42, model_type="random_forest", n_estimators=100)
    metrics_rf = evaluate_model(rf, X_train_42, y_train, X_test_42, y_test)
    results.append({"Model": "Random Forest (42 features)", **metrics_rf})

    # 3. Ridge Without Audited/Suspicious Features (37 features)
    print("Training Ridge Without Audited Features (37 features)...")
    num_37, cat_37 = get_feature_types(X_train_37)
    ridge_37 = build_pipeline(num_37, cat_37, model_type="ridge", alpha=1.0)
    metrics_ridge_37 = evaluate_model(ridge_37, X_train_37, y_train, X_test_37, y_test)
    results.append({"Model": "Ridge Without Audited Features (37 features)", **metrics_ridge_37})

    comparison_df = pd.DataFrame(results)
    print("\n===== MODEL COMPARISON RESULTS =====")
    print(comparison_df.to_string(index=False))

    # Persist the 37-feature Ridge model
    save_model(ridge_37, MODEL_RIDGE_37_PATH)
    print(f"\n37-feature Ridge model saved to: {MODEL_RIDGE_37_PATH}")

    return comparison_df


if __name__ == "__main__":
    run_comparison()

