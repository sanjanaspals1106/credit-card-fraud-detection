"""Feature engineering and preprocessing utilities.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature transformer for credit card transactions.

    Transforms:
    - Amount: Adds log-transformed Amount to tame extreme positive skew.
    - Time: Extracts 24-hour cycle and computes sine/cosine representations.
    """

    def __init__(self, add_time_features: bool = True, add_amount_log: bool = True):
        self.add_time_features = add_time_features
        self.add_amount_log = add_amount_log
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from raw transaction attributes.

        Args:
            X: Input DataFrame containing features (Time, V1..V28, Amount).

        Returns:
            pd.DataFrame with engineered columns.
        """
        X_out = X.copy()

        if self.add_amount_log and "Amount" in X_out.columns:
            # log1p handles 0-value transactions gracefully
            X_out["Amount_log"] = np.log1p(np.maximum(X_out["Amount"], 0.0))

        if self.add_time_features and "Time" in X_out.columns:
            # Dataset spans ~48 hours. Map elapsed seconds to 24-hour cycle.
            hour_of_day = (X_out["Time"] / 3600.0) % 24.0
            X_out["Hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0)
            X_out["Hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0)

        self.feature_names_out_ = list(X_out.columns)
        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        if input_features is not None:
            return list(input_features)
        return []
