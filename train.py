#!/usr/bin/env python3
"""Command-line interface to train and evaluate Credit Card Fraud Detection models.

Usage:
    python train.py --help
    python train.py --model hist_gradient_boosting
    python train.py --model all
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

from src.data_loader import load_dataset, split_data_group_stratified, validate_dataset
from src.evaluate import (
    compute_metrics,
    find_best_f1_threshold,
    find_cost_optimal_threshold,
)
from src.models import (
    get_fraud_scores,
    get_model_pipeline,
    save_model_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and tune Credit Card Fraud Detection models."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard.csv",
        help="Path to creditcard.csv dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save artifacts and evaluation tables",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "hist_gradient_boosting", "random_forest", "logistic_regression"],
        default="all",
        help="Model to train (or 'all' to compare all candidates)",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=10.0,
        help="Estimated dollar cost of a false alarm (FP)",
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=250.0,
        help="Estimated dollar cost of missed fraud (FN)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row limit for rapid testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("🛡️  CARDGUARD: CREDIT CARD FRAUD DETECTION TRAINING PIPELINE")
    print("==================================================================")

    # 1. Load data
    print(f"\n[1/5] Loading dataset from: {args.data_path}")
    df = load_dataset(args.data_path)
    if args.sample_rows and args.sample_rows < len(df):
        print(f"Sampling {args.sample_rows:,} rows for rapid verification...")
        df = df.sample(n=args.sample_rows, random_state=args.random_state).reset_index(drop=True)

    summary = validate_dataset(df)
    print(
        f"✓ Dataset verified: {summary['rows']:,} rows, {summary['fraud_count']:,} frauds "
        f"({summary['fraud_rate_pct']}%), {summary['duplicate_rows']:,} duplicates"
    )

    # 2. Group Stratified Split
    print("\n[2/5] Creating group-aware stratified train (60%) / val (20%) / test (20%) splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_group_stratified(
        df, random_state=args.random_state
    )
    print(f"  - Train:      {len(X_train):,} rows | {int(y_train.sum()):,} frauds")
    print(f"  - Validation: {len(X_val):,} rows | {int(y_val.sum()):,} frauds")
    print(f"  - Test:       {len(X_test):,} rows | {int(y_test.sum()):,} frauds")

    # 3. Model Training & Validation Comparison
    if args.model == "all":
        candidate_models = ["hist_gradient_boosting", "random_forest", "logistic_regression", "dummy"]
    else:
        candidate_models = [args.model, "dummy"]

    print(f"\n[3/5] Training candidate models: {candidate_models}...")
    fitted_models = {}
    val_metrics = []

    for model_name in candidate_models:
        print(f"  --> Training: {model_name}...")
        pipeline = get_model_pipeline(model_name, random_state=args.random_state)
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        # Validation scores
        scores = get_fraud_scores(pipeline, X_val)
        preds = (scores >= 0.5).astype(int)

        m = compute_metrics(
            y_val, preds, scores, label=model_name, cost_fp=args.cost_fp, cost_fn=args.cost_fn
        )
        val_metrics.append(m)

    val_df = pd.DataFrame(val_metrics).sort_values("PR-AUC (AP)", ascending=False).reset_index(drop=True)
    val_csv = output_dir / "validation_model_comparison.csv"
    val_df.to_csv(val_csv, index=False)

    print("\n=== Validation Performance Summary ===")
    print(val_df[["Setting", "PR-AUC (AP)", "ROC-AUC", "Recall", "Precision", "F1", "Est. Financial Loss ($)"]].to_string(index=False))

    # 4. Threshold Tuning on Validation
    best_model_name = val_df.iloc[0]["Setting"]
    best_pipeline = fitted_models[best_model_name]
    best_val_scores = get_fraud_scores(best_pipeline, X_val)

    print(f"\n[4/5] Tuning decision threshold for best model: '{best_model_name}'...")
    best_f1_th, max_f1, _ = find_best_f1_threshold(y_val, best_val_scores)
    best_cost_th, min_cost, _ = find_cost_optimal_threshold(
        y_val, best_val_scores, cost_fp=args.cost_fp, cost_fn=args.cost_fn
    )

    print(f"  - Default threshold 0.5000: F1={val_df.iloc[0]['F1']:.4f}")
    print(f"  - Max-F1 threshold:         {best_f1_th:.4f} (Validation F1: {max_f1:.4f})")
    print(f"  - Cost-optimal threshold:   {best_cost_th:.4f} (Validation Loss: ${min_cost:,.2f})")

    # 5. Held-out Test Evaluation
    print(f"\n[5/5] Evaluating once on held-out test set...")
    test_scores = get_fraud_scores(best_pipeline, X_test)

    test_pred_default = (test_scores >= 0.5).astype(int)
    test_pred_f1 = (test_scores >= best_f1_th).astype(int)
    test_pred_cost = (test_scores >= best_cost_th).astype(int)

    test_rows = [
        compute_metrics(y_test, test_pred_default, test_scores, "Default threshold (0.50)", args.cost_fp, args.cost_fn),
        compute_metrics(y_test, test_pred_f1, test_scores, f"Max-F1 threshold ({best_f1_th:.4f})", args.cost_fp, args.cost_fn),
        compute_metrics(y_test, test_pred_cost, test_scores, f"Cost-optimal threshold ({best_cost_th:.4f})", args.cost_fp, args.cost_fn),
    ]

    test_df = pd.DataFrame(test_rows)
    test_csv = output_dir / "test_evaluation.csv"
    test_df.to_csv(test_csv, index=False)

    print("\n=== Final Held-Out Test Evaluation ===")
    print(test_df[["Setting", "Fraud Caught (TP)", "False Alarms (FP)", "Fraud Missed (FN)", "Precision", "Recall", "F1", "Est. Financial Loss ($)"]].to_string(index=False))

    # Persist artifact
    artifact_path = output_dir / "fraud_detection_model.joblib"
    save_model_artifact(
        model=best_pipeline,
        threshold=best_f1_th,
        model_name=best_model_name,
        feature_columns=list(X_train.columns),
        output_path=artifact_path,
        metadata={
            "f1_threshold": best_f1_th,
            "cost_threshold": best_cost_th,
            "cost_fp": args.cost_fp,
            "cost_fn": args.cost_fn,
        },
    )
    print(f"\n✓ Successfully saved model artifact to: {artifact_path.resolve()}")
    print(f"✓ Saved validation comparison: {val_csv.resolve()}")
    print(f"✓ Saved test evaluation: {test_csv.resolve()}")
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
