"""
leave_model.py
Trains a 5-class multiclass RandomForest leave decision model.

Classes:
    0 = Reject (policy violation)
    1 = Auto Approve
    2 = Manager Review Required
    3 = Documentation Required
    4 = HR Escalation
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "employee_leave_tracking_data.csv"
)
MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "leave_xgb_model.joblib")


def synthesize_label(row: pd.Series) -> int:
    """Generate a realistic 5-class label from the raw CSV row."""
    days  = float(row.get("Days Taken", 0))
    rem   = float(row.get("Remaining Leaves", 0))
    taken = float(row.get("Leave Taken So Far", 0))
    ltype = str(row.get("Leave Type", "")).lower()

    if days > rem + 5:                              return 0  # Reject
    if "medical" in ltype:                          return 1  # Auto Approve
    if days > 10:                                   return 4  # HR Escalation
    if "earned" in ltype and days > 5:              return 2  # Manager Review
    if "casual" in ltype and (days > 3 or taken > 10): return 3  # Doc Required
    return 1                                                  # Auto Approve


def train_leave_model() -> None:
    """Load data, engineer features, train RandomForest, evaluate, and save."""

    # ── Load ─────────────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} rows from dataset.")

    # ── Target ───────────────────────────────────────────────────────────────
    if "Leave_Status" not in df.columns:
        print("Synthesizing 5-class ground truth labels...")
        df["Leave_Status"] = df.apply(synthesize_label, axis=1)

    # ── Features ─────────────────────────────────────────────────────────────
    # Ensure 'month' column exists (should be in CSV already)
    df["month"] = df["month"].fillna("Jan")

    FEATURE_COLS = [
        "Department", "Position", "Leave Type",
        "Days Taken", "Remaining Leaves", "Leave Taken So Far", "month"
    ]
    CAT = ["Department", "Position", "Leave Type", "month"]
    NUM = ["Days Taken", "Remaining Leaves", "Leave Taken So Far"]

    X = df[FEATURE_COLS]
    y = df["Leave_Status"]

    # ── Preprocessing ─────────────────────────────────────────────────────────
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", NUM),
    ])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # ── Train / Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    print("Training RandomForest model...")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred,
          labels=[0, 1, 2, 3, 4],
          target_names=["Reject","AutoApprove","ManagerReview","DocRequired","HREscalation"],
          zero_division=0))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(pipeline, MODEL_OUTPUT)
    print(f"\nModel saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    train_leave_model()
