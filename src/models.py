"""Model pipelines, training configurations, and artifact serialization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FraudFeatureEngineer


def get_model_pipeline(
    model_type: str,
    random_state: int = 42,
    engineer_features: bool = True,
) -> Pipeline:
    """Build an end-to-end model pipeline with preprocessing.

    Args:
        model_type: One of 'logistic_regression', 'random_forest',
                    'hist_gradient_boosting', or 'dummy'.
        random_state: Seed for reproducibility.
        engineer_features: Whether to include domain feature engineering.

    Returns:
        Configured scikit-learn Pipeline.
    """
    steps: List[tuple] = []

    if engineer_features:
        steps.append(("fe", FraudFeatureEngineer()))

    steps.append(("imputer", SimpleImputer(strategy="median")))

    if model_type == "dummy":
        steps.append(("model", DummyClassifier(strategy="most_frequent")))

    elif model_type == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            )
        )

    elif model_type == "random_forest":
        steps.append(
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            )
        )

    elif model_type == "hist_gradient_boosting":
        steps.append(
            (
                "model",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=150,
                    min_samples_leaf=20,
                    random_state=random_state,
                ),
            )
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            "Choose from ['dummy', 'logistic_regression', 'random_forest', 'hist_gradient_boosting']."
        )

    return Pipeline(steps)


def get_fraud_scores(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Safely extract 1D continuous fraud risk scores from an estimator.

    Args:
        estimator: Trained model or pipeline.
        X: Feature DataFrame.

    Returns:
        1D numpy array of fraud risk scores (probabilities or decision scores).
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(X)
        # Normalize decision function between 0 and 1 via logistic sigmoid
        return 1.0 / (1.0 + np.exp(-raw))
    return estimator.predict(X).astype(float)


def save_model_artifact(
    model: Any,
    threshold: float,
    model_name: str,
    feature_columns: List[str],
    output_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Package and persist model pipeline and decision metadata.

    Args:
        model: Trained pipeline.
        threshold: Chosen decision threshold.
        model_name: Descriptive name of model.
        feature_columns: List of input feature names expected by the model.
        output_path: Target .joblib filepath.
        metadata: Additional metadata (metrics, training date, etc.).

    Returns:
        Path: Resolved file path.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "threshold": float(threshold),
        "model_name": model_name,
        "feature_columns": list(feature_columns),
        "target_name": "Class",
        "positive_class": 1,
        "metadata": metadata or {},
    }
    joblib.dump(artifact, dest)
    return dest.resolve()


def load_model_artifact(artifact_path: str | Path) -> Dict[str, Any]:
    """Load model artifact dictionary.

    Args:
        artifact_path: Path to .joblib file.

    Returns:
        Dictionary containing model, threshold, feature_columns, etc.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path.resolve()}")
    return joblib.load(path)
