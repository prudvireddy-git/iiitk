from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum

class PipelineStatus(Enum):
    APPLIED = "applied"
    SHORTLISTED = "shortlisted"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    INTERVIEWED = "interviewed"
    SELECTED = "selected"
    REJECTED = "rejected"

    @staticmethod
    def valid_transitions():
        return {
            "applied": ["shortlisted", "rejected"],
            "shortlisted": ["interview_scheduled", "rejected"],
            "interview_scheduled": ["interviewed", "rejected"],
            "interviewed": ["selected", "rejected"],
            "selected": [],
            "rejected": [],
        }

@dataclass
class Candidate:
    candidate_id: str
    name: str
    email: str
    resume_text: str
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    match_score: float = 0.0
    status: str = "applied"

@dataclass
class JobDescription:
    job_id: str
    title: str
    description: str
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_experience: float = 0.0

@dataclass
class InterviewSlot:
    slot_id: str
    interviewer_id: str
    start_time: datetime
    end_time: datetime
    is_available: bool = True

@dataclass
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str
    start_date: datetime
    end_date: datetime
    reason: str
    status: str = "pending"
    policy_violations: List[str] = field(default_factory=list)

@dataclass
class LeavePolicy:
    leave_type: str
    annual_quota: int
    max_consecutive_days: int
    min_notice_days: int
    requires_document: bool = False
