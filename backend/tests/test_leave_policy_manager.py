"""
tests/test_leave_policy_manager.py
Pytest unit tests covering all 5 outcome classes.
Run with: python3 -m pytest backend/tests/test_leave_policy_manager.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from ml_leave_policy_manager import MLLeavePolicyManager
from models import LeaveRequest, LeavePolicy


# ─── Shared fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def manager():
    """Manager with a mocked model so tests run without a real .joblib file."""
    mgr = MLLeavePolicyManager(model_path="nonexistent.joblib")
    mgr.model = MagicMock()
    return mgr


@pytest.fixture
def std_policy():
    return LeavePolicy(
        leave_type="Vacation",
        annual_quota=20,
        max_consecutive_days=7,
        min_notice_days=2,
        requires_document=False,
    )


def _make_request(days: int = 3, notice: int = 5, ltype: str = "Vacation") -> LeaveRequest:
    start = datetime.now() + timedelta(days=notice)
    end   = start + timedelta(days=days - 1)
    return LeaveRequest("LR001", "EMP001", ltype, start, end, "test reason")


def _mock_ml(manager: MLLeavePolicyManager, cls: int, confidence: float = 0.9):
    """Configure the mocked model to return a specific class."""
    probas = [0.025] * 5
    probas[cls] = confidence
    manager.model.predict.return_value = [cls]
    manager.model.predict_proba.return_value = [probas]


# ─── Tests ────────────────────────────────────────────────────────────────────
class TestPolicyViolationReject:
    """Class 0 – hard policy rejection (balance overdraft)."""

    def test_insufficient_balance_triggers_reject(self, manager, std_policy):
        req    = _make_request(days=10)      # 10 days requested
        result = manager.process_leave_request(req, std_policy, current_balance=5)

        assert result["approved"] is False
        assert result["predicted_class"] == 0
        assert result["predicted_label"] == "Reject"
        assert "Insufficient leave balance" in result["violations"]
        assert result["policy_checks"]["days_requested"] == 10

    def test_notice_period_violation(self, manager, std_policy):
        start  = datetime.now() + timedelta(days=0)   # same-day request
        end    = start + timedelta(days=2)
        req    = LeaveRequest("LR002", "EMP002", "Vacation", start, end, "urgent")
        result = manager.process_leave_request(req, std_policy, current_balance=15)

        assert result["approved"] is False
        assert "Insufficient notice period" in result["violations"]


class TestAutoApprove:
    """Class 1 – ML says approve."""

    def test_auto_approve(self, manager, std_policy):
        _mock_ml(manager, cls=1)
        req    = _make_request(days=2)
        result = manager.process_leave_request(req, std_policy, current_balance=15)

        assert result["approved"] is True
        assert result["predicted_class"] == 1
        assert result["predicted_label"] == "AutoApprove"
        assert result["violations"] == []


class TestManagerReview:
    """Class 2 – ML requires manager review."""

    def test_manager_review(self, manager, std_policy):
        _mock_ml(manager, cls=2)
        req    = _make_request(days=5)
        result = manager.process_leave_request(req, std_policy, current_balance=15)

        assert result["approved"] is False
        assert result["predicted_class"] == 2
        assert result["predicted_label"] == "ManagerReview"
        assert "ManagerReview" in result["reasons"][-1]


class TestDocRequired:
    """Class 3 – ML requires documentation."""

    def test_doc_required(self, manager, std_policy):
        _mock_ml(manager, cls=3)
        req    = _make_request(days=4, ltype="Casual")
        result = manager.process_leave_request(req, std_policy, current_balance=15)

        assert result["approved"] is False
        assert result["predicted_class"] == 3
        assert result["predicted_label"] == "DocRequired"


class TestHREscalation:
    """Class 4 – ML requires HR escalation."""

    def test_hr_escalation(self, manager, std_policy):
        _mock_ml(manager, cls=4)
        req    = _make_request(days=7, ltype="Sabbatical")
        result = manager.process_leave_request(req, std_policy, current_balance=15)

        assert result["approved"] is False
        assert result["predicted_class"] == 4
        assert result["predicted_label"] == "HREscalation"

    def test_response_schema_complete(self, manager, std_policy):
        """Verify all 9 required keys are always present."""
        _mock_ml(manager, cls=4)
        req    = _make_request(days=7)
        result = manager.process_leave_request(req, std_policy, current_balance=15)

        required_keys = {
            "approved", "predicted_class", "predicted_label", "probabilities",
            "violations", "reasons", "policy_checks", "model_version", "decision_timestamp"
        }
        assert required_keys.issubset(result.keys())
