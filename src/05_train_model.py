"""HERAI Model Training Pipeline.

Trains the 42-feature Ridge Regression pipeline using user-grouped train/test splitting.
Saves the serialized pipeline to models/herai_ridge_model.pkl.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from herai.config import (
    ENGINEERED_DATA_PATH,
    MODEL_RIDGE_PATH,
    TARGET_COLUMN,
    EXCLUDED_FROM_TRAINING,
)
from herai.models import (
    split_by_user,
    get_feature_types,
    build_pipeline,
    evaluate_model,
    save_model,
)


def run_training() -> Pipeline:
    """Execute model training on the 42-feature dataset with user-grouped splitting.
    
    Returns:
        Pipeline: Trained scikit-learn pipeline.
    """
    print("\n========== HERAI MODEL TRAINING ==========")
    print(f"Loading engineered data from: {ENGINEERED_DATA_PATH}")

    if not ENGINEERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {ENGINEERED_DATA_PATH}. Run src/03_create_features.py first."
        )

    df = pd.read_csv(ENGINEERED_DATA_PATH)
    clean_df = df.dropna(subset=[TARGET_COLUMN]).copy()

    feature_cols = [c for c in clean_df.columns if c not in EXCLUDED_FROM_TRAINING]
    print(f"Number of input features: {len(feature_cols)}")
    print(f"Number of valid samples: {len(clean_df)}")
    print(f"Unique users: {clean_df['user_id'].nunique()}")

    # User-grouped train/test split (80/20)
    X_train, X_test, y_train, y_test, g_train, g_test = split_by_user(
        clean_df, feature_cols=feature_cols, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples across {g_train.nunique()} users.")
    print(f"Testing set: {len(X_test)} samples across {g_test.nunique()} users.")
    overlap = set(g_train).intersection(set(g_test))
    print(f"User overlap between splits: {len(overlap)} (Must be 0)")

    # Partition features
    numeric_features, categorical_features = get_feature_types(X_train)
    print(f"Numerical features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")

    # Build and evaluate Ridge pipeline
    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type="ridge",
        alpha=1.0,
        random_state=42,
    )

    metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)

    print("\n========== MODEL EVALUATION (TEST SPLIT) ==========")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Baseline comparison (predicting mean)
    baseline_pred = np.full(shape=len(y_test), fill_value=float(y_train.mean()))
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    b_mae = float(mean_absolute_error(y_test, baseline_pred))
    b_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    b_r2 = float(r2_score(y_test, baseline_pred))

    print("\n========== DUMMY BASELINE COMPARISON ==========")
    print(f"  Baseline MAE:  {b_mae:.4f}")
    print(f"  Baseline RMSE: {b_rmse:.4f}")
    print(f"  Baseline R2:   {b_r2:.4f}")

    # Save trained model
    save_model(pipeline, MODEL_RIDGE_PATH)
    print(f"\nModel saved successfully to: {MODEL_RIDGE_PATH}")

    return pipeline


if __name__ == "__main__":
    run_training()
