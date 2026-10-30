"""Tests for src.evaluate using Python unittest.
"""

import unittest
import numpy as np

from src.evaluate import (
    compute_metrics,
    find_best_f1_threshold,
    find_cost_optimal_threshold,
)


class TestEvaluate(unittest.TestCase):
    def test_compute_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.6, 0.4, 0.9])

        metrics = compute_metrics(
            y_true, y_pred, scores, label="Test", cost_fp=10.0, cost_fn=250.0
        )

        self.assertEqual(metrics["Fraud Caught (TP)"], 1)
        self.assertEqual(metrics["False Alarms (FP)"], 1)
        self.assertEqual(metrics["Fraud Missed (FN)"], 1)
        self.assertEqual(metrics["Legit Cleared (TN)"], 1)
        self.assertEqual(metrics["Est. Financial Loss ($)"], 260.0)
        self.assertEqual(metrics["Accuracy"], 0.5)

    def test_find_best_f1_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        best_th, max_f1, table = find_best_f1_threshold(y_true, scores)
        self.assertTrue(0.3 <= best_th <= 0.8)
        self.assertGreater(max_f1, 0.8)
        self.assertFalse(table.empty)

    def test_find_cost_optimal_threshold(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([0.05, 0.10, 0.15, 0.20, 0.70, 0.85])

        opt_th, min_loss, curve = find_cost_optimal_threshold(
            y_true, scores, cost_fp=10.0, cost_fn=500.0
        )
        self.assertTrue(0.2 < opt_th < 0.7)
        self.assertEqual(min_loss, 0.0)
        self.assertGreater(len(curve), 0)


if __name__ == "__main__":
    unittest.main()
