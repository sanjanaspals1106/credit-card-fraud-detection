"""Tests for src.features using Python unittest.
"""

import unittest
import numpy as np
import pandas as pd

from src.features import FraudFeatureEngineer


class TestFeatures(unittest.TestCase):
    def test_fraud_feature_engineer_transformation(self):
        df = pd.DataFrame(
            {
                "Time": [0.0, 3600.0 * 6, 3600.0 * 12, 3600.0 * 18],
                "Amount": [0.0, 10.0, 99.5, 500.0],
                "V1": [-1.0, 0.5, 1.2, -0.3],
            }
        )

        transformer = FraudFeatureEngineer()
        out = transformer.fit_transform(df)

        self.assertIn("Amount_log", out.columns)
        self.assertIn("Hour_sin", out.columns)
        self.assertIn("Hour_cos", out.columns)

        np.testing.assert_allclose(out["Amount_log"].values, np.log1p(df["Amount"].values))
        self.assertTrue((out["Hour_sin"] >= -1.0).all() and (out["Hour_sin"] <= 1.0).all())
        self.assertTrue((out["Hour_cos"] >= -1.0).all() and (out["Hour_cos"] <= 1.0).all())
        self.assertEqual(out.isna().sum().sum(), 0)

    def test_fraud_feature_engineer_missing_time_or_amount(self):
        df = pd.DataFrame({"V1": [1.0, 2.0], "V2": [3.0, 4.0]})
        transformer = FraudFeatureEngineer()
        out = transformer.fit_transform(df)

        self.assertEqual(list(out.columns), ["V1", "V2"])


if __name__ == "__main__":
    unittest.main()
