"""Tests for src.models using Python unittest.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.models import (
    get_fraud_scores,
    get_model_pipeline,
    load_model_artifact,
    save_model_artifact,
)


class TestModels(unittest.TestCase):
    def test_model_pipeline_training_and_scoring(self):
        X = pd.DataFrame(
            {
                "Time": np.linspace(0, 10000, 40),
                "Amount": np.random.uniform(5, 200, 40),
                "V1": np.random.randn(40),
                "V2": np.random.randn(40),
            }
        )
        y = pd.Series(np.array([0] * 35 + [1] * 5))

        pipeline = get_model_pipeline("logistic_regression", random_state=42)
        pipeline.fit(X, y)

        scores = get_fraud_scores(pipeline, X)
        self.assertEqual(len(scores), len(X))
        self.assertTrue((scores >= 0.0).all() and (scores <= 1.0).all())

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "test_model.joblib"
            save_model_artifact(
                model=pipeline,
                threshold=0.35,
                model_name="TestLogisticRegression",
                feature_columns=list(X.columns),
                output_path=artifact_path,
            )

            loaded = load_model_artifact(artifact_path)
            self.assertEqual(loaded["model_name"], "TestLogisticRegression")
            self.assertEqual(loaded["threshold"], 0.35)
            self.assertEqual(loaded["feature_columns"], list(X.columns))


if __name__ == "__main__":
    unittest.main()
