"""Data loading, validation, and group-aware train/val/test splitting.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load credit card transactions CSV.

    Args:
        data_path: Path to the creditcard.csv file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path.resolve()}. "
            "Please ensure creditcard.csv is placed in the data/ folder."
        )
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame, target_col: str = "Class") -> Dict[str, any]:
    """Perform data quality and schema checks on transaction data.

    Args:
        df: Input DataFrame.
        target_col: Name of binary target label column.

    Returns:
        Dict containing summary checks: shape, missing_values, duplicate_count, fraud_rate.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    unique_targets = set(df[target_col].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Expected binary {target_col} in {{0, 1}}, got {unique_targets}")

    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    fraud_count = int((df[target_col] == 1).sum())
    legit_count = int((df[target_col] == 0).sum())
    fraud_rate = float(fraud_count / len(df))

    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": missing_count,
        "duplicate_rows": duplicate_count,
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_rate_pct": round(fraud_rate * 100, 4),
    }


def split_data_group_stratified(
    df: pd.DataFrame,
    target_col: str = "Class",
    random_state: int = 42,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Perform a 60/20/20 train/validation/test split using StratifiedGroupKFold.

    Identical feature vectors receive the same group ID (hash) to prevent
    leakage across train, validation, and test partitions.

    Args:
        df: Input DataFrame with features and target.
        target_col: Target column name.
        random_state: Random state for reproducibility.
        n_splits: Number of folds (default 5 => 60% train, 20% val, 20% test).

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()

    # Hash feature vectors so identical rows share group IDs
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = list(splitter.split(X, y, groups=groups))

    # Fold 0 = Test, Fold 1 = Validation, Folds 2..4 = Training
    test_idx = folds[0][1]
    val_idx = folds[1][1]
    train_idx = np.concatenate([folds[i][1] for i in range(2, n_splits)])

    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    X_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test
