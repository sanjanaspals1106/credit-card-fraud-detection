"""Tests for src.data_loader using Python unittest.
"""

import unittest
import numpy as np
import pandas as pd

from src.data_loader import split_data_group_stratified, validate_dataset


class TestDataLoader(unittest.TestCase):
    def test_validate_dataset_success(self):
        df = pd.DataFrame(
            {
                "Time": [0.0, 1.0, 2.0, 3.0],
                "Amount": [10.0, 20.0, 50.0, 100.0],
                "V1": [0.1, -0.2, 0.5, -0.9],
                "Class": [0, 0, 1, 0],
            }
        )
        result = validate_dataset(df)
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["fraud_count"], 1)
        self.assertEqual(result["legit_count"], 3)
        self.assertEqual(result["missing_values"], 0)

    def test_validate_dataset_missing_target(self):
        df = pd.DataFrame({"V1": [1.0, 2.0], "Amount": [10.0, 20.0]})
        with self.assertRaises(ValueError):
            validate_dataset(df)

    def test_validate_dataset_invalid_labels(self):
        df = pd.DataFrame({"V1": [1.0, 2.0], "Class": [0, 2]})
        with self.assertRaises(ValueError):
            validate_dataset(df)

    def test_split_data_group_stratified_no_leakage(self):
        np.random.seed(42)
        base_rows = pd.DataFrame(
            {
                "Time": np.random.uniform(0, 3600, 50),
                "V1": np.random.randn(50),
                "Amount": np.random.exponential(50, 50),
                "Class": np.random.choice([0, 1], size=50, p=[0.9, 0.1]),
            }
        )
        duplicated_rows = base_rows.iloc[:10].copy()
        full_df = pd.concat([base_rows, duplicated_rows], ignore_index=True)

        X_train, y_train, X_val, y_val, X_test, y_test = split_data_group_stratified(
            full_df, n_splits=5, random_state=42
        )

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(full_df))

        # Check zero hash overlap between partitions
        train_hashes = set(pd.util.hash_pandas_object(X_train, index=False))
        val_hashes = set(pd.util.hash_pandas_object(X_val, index=False))
        test_hashes = set(pd.util.hash_pandas_object(X_test, index=False))

        self.assertEqual(len(train_hashes.intersection(test_hashes)), 0)
        self.assertEqual(len(train_hashes.intersection(val_hashes)), 0)
        self.assertEqual(len(val_hashes.intersection(test_hashes)), 0)


if __name__ == "__main__":
    unittest.main()
