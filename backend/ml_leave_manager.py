import os
import joblib
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List
from dataclasses import asdict
from models import LeaveRequest, LeavePolicy

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "data")
LEAVE_CSV = os.path.join(DATA_DIR, "employee_leave_tracking.csv")

class MLLeaveManager:
    """Leave governance engine backed by the trained RandomForest model."""

    def __init__(self):
        self.model_path = os.path.join(
            os.path.dirname(__file__), "..", "leave_xgb_model.joblib"
        )
        self.model = None
        self.classes = ["Reject", "AutoApprove", "ManagerReview", "DocRequired", "HREscalation"]
        self.decision_logs = []
        self.employee_data: Dict = {}
        self._load_model()
        self._load_data()
        self._seed_dummy_logs()

    def _load_model(self):
        """Load the saved RandomForest pipeline."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Leave model loaded from %s", self.model_path)
            else:
                logger.warning("Model not found at %s – rule-only mode.", self.model_path)
        except Exception as e:
            logger.error("Model load error: %s", e)

    def _load_data(self):
        """Load employee leave data from CSV."""
        try:
            if os.path.exists(LEAVE_CSV):
                df = pd.read_csv(LEAVE_CSV)
                for _, row in df.iterrows():
                    emp_id = str(row.get("Employee ID", row.get("Employee Name", "")))
                    self.employee_data[emp_id] = row.to_dict()
                logger.info("Loaded %d employee records", len(self.employee_data))
        except Exception as e:
            logger.error("Data load error: %s", e)

    def _seed_dummy_logs(self):
        """Add some example logs for UI demonstration."""
        base = datetime.now()
        examples = [
            ("EMP001", "Vacation", 3, True, "AutoApprove"),
            ("EMP002", "Medical", 5, True, "AutoApprove"),
            ("EMP003", "Casual", 8, False, "Reject"),
            ("EMP004", "Earned", 6, False, "ManagerReview"),
        ]
        for emp_id, ltype, days, approved, label in examples:
            self.decision_logs.append({
                "timestamp": base.isoformat(),
                "employee_id": emp_id,
                "leave_type": ltype,
                "days": days,
                "ground_truth": approved,
                "result": {
                    "approved": approved,
                    "predicted_class": 1 if approved else 0,
                    "class_label": label,
                    "comment": "Seeded example decision.",
                    "violations": [] if approved else ["Policy exceeded"]
                }
            })

    def _extract_features(self, request: LeaveRequest):
        """Build the feature list used by the model (matches leave_model.py)."""
        start = request.start_date if isinstance(request.start_date, datetime) \
                else datetime.fromisoformat(str(request.start_date))
        end   = request.end_date   if isinstance(request.end_date,   datetime) \
                else datetime.fromisoformat(str(request.end_date))
        days = (end - start).days + 1
        emp_info = self.employee_data.get(request.employee_id, {})
        return pd.DataFrame([{
            "Department":         emp_info.get("Department", "Engineering"),
            "Position":           emp_info.get("Position", "Employee"),
            "Leave Type":         request.leave_type,
            "Days Taken":         days,
            "Remaining Leaves":   emp_info.get("Remaining Leaves", 15),
            "Leave Taken So Far": emp_info.get("Leave Taken So Far", 0),
            "month":              start.strftime("%b"),
        }])

    def _get_local_reasoning(self, pred_class: int, ctx: Dict) -> str:
        """Provide a data-driven explanation citing specific policy bounds."""
        days  = ctx.get("days", 0)
        rem   = ctx.get("rem", 0)
        taken = ctx.get("taken", 0)
        ltype = str(ctx.get("ltype", "")).lower()

        if pred_class == 1:
            return f"Approved: request of {days} days is within standard healthy bounds for {ltype.capitalize()}."
        if pred_class == 0:
            if days > rem:
                return f"Rejected: request of {days} days exceeds your remaining balance of {rem}."
            return f"Rejected: pattern matches high-risk policy violation ({days} days requested)."
        if pred_class == 2:
            return f"Manager Review Required: request of {days} days for {ltype.capitalize()} triggers mandatory executive oversight."
        if pred_class == 3:
            reason = "High frequency" if taken > 10 else f"Duration ({days} days)"
            return f"Documentation Required: {reason} for {ltype.capitalize()} requires supporting evidence."
        if pred_class == 4:
            return f"Internal HR Audit: request length of {days} days deviates significantly from department benchmarks."
        return "Standard policy evaluation complete: manual oversight recommended."

    def get_metrics(self) -> Dict:
        """Calculate TP, TN, FP, FN for the dashboard."""
        TP = TN = FP = FN = 0
        for log in self.decision_logs:
            gt  = log.get("ground_truth", False)
            pred = log.get("result", {}).get("approved", False)
            if gt and pred:   TP += 1
            elif not gt and not pred: TN += 1
            elif not gt and pred: FP += 1
            elif gt and not pred: FN += 1
        total   = max(TP + TN + FP + FN, 1)
        accuracy = (TP + TN) / total
        return {
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "total": total,
            "summary": {
                "accuracy":  round(accuracy * 100, 1),
                "precision": round(TP / max(TP + FP, 1), 4),
                "recall":    round(TP / max(TP + FN, 1), 4),
            }
        }

    def process_request(self, request: LeaveRequest, policy: LeavePolicy) -> Dict:
        """Main entry point: policy checks + ML inference."""
        violations: List[str] = []

        # Parse dates
        start = request.start_date if isinstance(request.start_date, datetime) \
                else datetime.fromisoformat(str(request.start_date))
        end   = request.end_date   if isinstance(request.end_date,   datetime) \
                else datetime.fromisoformat(str(request.end_date))
        days_requested = (end - start).days + 1

        emp_info = self.employee_data.get(request.employee_id, {})
        current_balance = emp_info.get("Remaining Leaves", 15)

        # Hard policy checks
        if int(days_requested) > int(current_balance):
            violations.append("Insufficient leave balance")
        if int(days_requested) > int(policy.max_consecutive_days):
            violations.append(f"Exceeds maximum consecutive days ({policy.max_consecutive_days})")

        try:
            notice_days = (start.date() - datetime.now().date()).days
            if notice_days < policy.min_notice_days:
                violations.append(f"Insufficient notice period ({notice_days} days vs required {policy.min_notice_days})")
        except Exception as e:
            logger.error("Notice calculation error: %s", e)

        if policy.requires_document and not request.reason:
            violations.append("Supporting documentation/reason required for this leave type")

        if violations:
            primary = violations[0]
            if len(violations) > 1:
                primary += f" (and {len(violations)-1} other policy conflicts)"
            return {
                "approved": False,
                "predicted_class": 0,
                "class_label": "Rejected",
                "probabilities": {"Rejected": 1.0},
                "violations": violations,
                "comment": f"Policy Conflict: {primary}.",
            }

        # ML inference
        pred_class_idx = 3
        proba_dict = {label: 0.0 for label in self.classes}

        if self.model:
            try:
                features = self._extract_features(request)
                proba = self.model.predict_proba(features)[0]
                pred_class_idx = int(self.model.predict(features)[0])
                proba_dict = {self.classes[i]: float(proba[i]) for i in range(len(self.classes))}
            except Exception as e:
                logger.error("Inference failed: %s", e)
        else:
            if days_requested <= 2:   pred_class_idx = 1
            elif days_requested <= 5: pred_class_idx = 2
            else:                     pred_class_idx = 3

        final_approved = (pred_class_idx == 1)
        status_label   = "Approved" if final_approved else "Rejected"

        ctx = {
            "days":  days_requested,
            "rem":   current_balance,
            "taken": emp_info.get("Leave Taken So Far", 0),
            "ltype": request.leave_type,
        }
        explanation = self._get_local_reasoning(pred_class_idx, ctx)

        result = {
            "approved":        final_approved,
            "predicted_class": pred_class_idx,
            "class_label":     status_label,
            "probabilities":   proba_dict,
            "violations":      [],
            "comment":         explanation,
        }

        self.decision_logs.append({
            "timestamp":   datetime.now().isoformat(),
            "employee_id": request.employee_id,
            "leave_type":  request.leave_type,
            "days":        days_requested,
            "ground_truth": days_requested <= 4,
            "result":      result,
        })

        return result
