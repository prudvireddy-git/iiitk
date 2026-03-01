"""
main.py – Full HR Agent FastAPI Backend
Handles: Candidate Screening, Interview Scheduling, Question Generation,
         Leave Management, and Model Evaluation.
"""

import os
import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional, Dict

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import (
    Candidate, JobDescription, InterviewSlot,
    LeaveRequest, LeavePolicy, PipelineStatus
)
from ml_leave_manager import MLLeaveManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "llm_model":           os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "data_dir":            os.getenv("DATA_DIR", "data"),
    "resume_csv":          "resume_jd_dataset.csv",
    "cand_slots_csv":      "candidate_free_slots.csv",
    "int_slots_csv":       "interviewers_free_slots.csv",
    "top_n":               10,
    "min_experience":      0,
    "groq_api_key":        os.getenv("GROQ_API_KEY", ""),
}

app = FastAPI(title="AI HR Agent API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─── Helper utilities ─────────────────────────────────────────────────────────
def _path(filename: str) -> str:
    return os.path.join(CONFIG["data_dir"], filename)


# ─── Candidate Ranker ─────────────────────────────────────────────────────────
class CandidateRanker:
    """Ranks candidates against a JD using TF-IDF cosine similarity."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def rank(self, candidates: List[Candidate], jd: JobDescription,
             min_exp: float = 0) -> List[Candidate]:
        eligible = [c for c in candidates if c.experience_years >= min_exp]
        if not eligible:
            return candidates[:CONFIG["top_n"]]

        docs = [jd.description] + [c.resume_text for c in eligible]
        try:
            matrix = self.vectorizer.fit_transform(docs)
            scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        except Exception as e:
            logger.error("Ranking error: %s", e)
            scores = np.zeros(len(eligible))

        for i, c in enumerate(eligible):
            c.match_score = round(float(scores[i]) * 100, 2)

        return sorted(eligible, key=lambda c: c.match_score, reverse=True)[:CONFIG["top_n"]]


# ─── Scheduler ────────────────────────────────────────────────────────────────
class InterviewScheduler:
    """Bipartite-matching scheduler for interviews."""

    @staticmethod
    def _overlap(s1, e1, s2, e2) -> bool:
        return s1 < e2 and s2 < e1

    def schedule(self, candidates: List[Candidate],
                 cand_slots: List[dict], int_slots: pd.DataFrame) -> List[dict]:
        G = nx.Graph()
        top_names = [c.name for c in candidates]

        for name in top_names:
            G.add_node(name, bipartite="candidates")

        avail = int_slots[int_slots["is_available"] == True]
        for _, row in avail.iterrows():
            G.add_node(row["slot_id"], bipartite="slots")

        for cs in cand_slots:
            if cs["Name"] not in top_names:
                continue
            for _, ir in avail.iterrows():
                if self._overlap(cs["slot_start"], cs["slot_end"],
                                 ir["start_time"], ir["end_time"]):
                    G.add_edge(cs["Name"], ir["slot_id"])

        matching = nx.algorithms.bipartite.matching.maximum_matching(
            G, top_nodes=set(top_names)
        )

        results = []
        for c in candidates:
            slot_id = matching.get(c.name)
            if slot_id:
                slot_row = avail[avail["slot_id"] == slot_id].iloc[0]
                results.append({
                    "candidate":    c.name,
                    "candidate_id": c.candidate_id,
                    "slot":         slot_row.to_dict(),
                    "status":       "scheduled",
                })
            else:
                results.append({
                    "candidate":    c.name,
                    "candidate_id": c.candidate_id,
                    "slot":         None,
                    "status":       "no_slot_available",
                    "reason":       "no_slot_available",
                })
        return results


# ─── LLM Question Generator ──────────────────────────────────────────────────
class LLMQuestionnaireGenerator:
    def __init__(self):
        self.client = Groq(api_key=CONFIG["groq_api_key"])

    def generate_questions(self, jd: JobDescription,
                           candidate: Optional[Candidate] = None) -> List[dict]:
        prompt = f"""Generate 10 structured interview questions for:
Role: {jd.title}
Description: {jd.description}
Candidate: {candidate.name if candidate else 'Generic'}
Skills: {candidate.skills if candidate else 'N/A'}

Return ONLY a JSON array: [{{"question":"...","type":"technical|behavioral|situational|career","category":"..."}}]"""

        try:
            resp = self.client.chat.completions.create(
                model=CONFIG["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "")
            start, end = text.find("["), text.rfind("]")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
        except Exception as e:
            logger.error("Question generation error: %s", e)
        return []


# ─── HR Agent orchestrator ────────────────────────────────────────────────────
class HRAgent:
    def __init__(self):
        self.pipeline: Dict[str, Candidate] = {}
        self.candidates: List[Candidate] = []
        self.jd: Optional[JobDescription] = None
        self.ranker    = CandidateRanker()
        self.scheduler = InterviewScheduler()
        self.question_gen = LLMQuestionnaireGenerator()
        self.leave_mgr = MLLeaveManager()
        self._load_data()

    def _load_data(self):
        try:
            df = pd.read_csv(_path(CONFIG["resume_csv"]))
            self.candidates: List[Candidate] = []
            for idx, row in df.iterrows():
                skills = [s.strip() for s in str(row.get("Skills", "")).split(",")
                          if str(row.get("Skills", "")).strip()]
                txt = " ".join(filter(None, [
                    str(row.get("Current_Job_Title", "")),
                    str(row.get("Previous_Job_Titles", "")),
                    " ".join(skills),
                    str(row.get("Certifications", "")),
                ]))
                c = Candidate(
                    candidate_id=str(idx),
                    name=str(row.get("Name", f"Candidate_{idx}")),
                    email=str(row.get("Email", "")),
                    resume_text=txt,
                    skills=skills,
                    experience_years=float(row.get("Experience_Years", 0)),
                )
                self.candidates.append(c)
                self.pipeline[c.candidate_id] = c
            logger.info("Loaded %d candidates", len(self.candidates))
        except Exception as e:
            logger.error("Data load error: %s", e)
            self.candidates = []

        try:
            cf = pd.read_csv(_path(CONFIG["cand_slots_csv"]))
            cf["slot_start"] = pd.to_datetime(cf["slot_start"])
            cf["slot_end"]   = pd.to_datetime(cf["slot_end"])
            self.cand_slots = cf.to_dict("records")
        except Exception as e:
            logger.warning("Candidate slots load error: %s", e)
            self.cand_slots = []

        try:
            self.int_slots = pd.read_csv(_path(CONFIG["int_slots_csv"]))
            self.int_slots["start_time"] = pd.to_datetime(self.int_slots["start_time"])
            self.int_slots["end_time"]   = pd.to_datetime(self.int_slots["end_time"])
        except Exception as e:
            logger.warning("Interviewer slots load error: %s", e)
            self.int_slots = pd.DataFrame()


agent = HRAgent()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class JDInput(BaseModel):
    title: str
    description: str
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    min_experience: float = 0.0

class ScreenRequest(BaseModel):
    job_description: JDInput
    min_experience:  float = 0.0
    top_n:           int   = 10

class LeaveRequestInput(BaseModel):
    employee_id: str
    leave_type:  str
    start_date:  str
    end_date:    str
    reason:      Optional[str] = ""

class StatusUpdate(BaseModel):
    candidate_id: str
    status:       str


class ResumeScreeningRequest(BaseModel):
    title: str
    description: str
    min_experience: float = 0.0
    shortlist_limit: int = 10

# ─── Screening endpoints ──────────────────────────────────────────────────────
@app.post("/resume_screening")
async def resume_screening(req: ResumeScreeningRequest):
    """Direct implementation for the ScreeningPage.jsx frontend."""
    if not agent.candidates:
        raise HTTPException(status_code=503, detail="Candidate data not loaded")

    jd = JobDescription(
        job_id="JID_" + datetime.now().strftime("%H%M%S"),
        title=req.title,
        description=req.description,
        required_skills=[],
        preferred_skills=[],
        min_experience=req.min_experience,
    )
    agent.jd = jd
    ranked = agent.ranker.rank(agent.candidates, jd, min_exp=req.min_experience)
    
    # Respect the shortlist limit from frontend
    final_candidates = ranked[:req.shortlist_limit]

    return [
        {
            "candidate_id":    c.candidate_id,
            "name":            c.name,
            "match_score":     c.match_score,
            "experience_years": c.experience_years,
            "skills":          c.skills[:5],
            "status":          c.status,
        }
        for c in final_candidates
    ]


@app.post("/screen")
async def screen_candidates(req: ScreenRequest):
    if not agent.candidates:
        raise HTTPException(status_code=503, detail="Candidate data not loaded")

    jd = JobDescription(
        job_id="JID_" + datetime.now().strftime("%H%M%S"),
        title=req.job_description.title,
        description=req.job_description.description,
        required_skills=req.job_description.required_skills,
        preferred_skills=req.job_description.preferred_skills,
        min_experience=req.min_experience,
    )
    agent.jd = jd
    ranked = agent.ranker.rank(agent.candidates, jd, min_exp=req.min_experience)

    return {
        "job_title":    jd.title,
        "total_pool":   len(agent.candidates),
        "shortlisted":  len(ranked),
        "candidates": [
            {
                "candidate_id":    c.candidate_id,
                "name":            c.name,
                "match_score":     c.match_score,
                "experience_years": c.experience_years,
                "skills":          c.skills[:5],
                "status":          c.status,
            }
            for c in ranked
        ],
    }


@app.get("/candidates")
async def list_candidates():
    return [
        {
            "candidate_id":    c.candidate_id,
            "name":            c.name,
            "match_score":     c.match_score,
            "experience_years": c.experience_years,
            "skills":          c.skills[:5],
            "status":          c.status,
        }
        for c in agent.candidates
        if c.match_score > 0
    ]


@app.post("/candidates/status")
async def update_status(req: StatusUpdate):
    c = agent.pipeline.get(req.candidate_id)
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found")
    c.status = req.status
    return {"candidate_id": req.candidate_id, "status": c.status}


# ─── Scheduling endpoints ──────────────────────────────────────────────────────
@app.post("/schedule")
async def schedule_interviews():
    if not agent.jd:
        raise HTTPException(status_code=400, detail="Run /screen first")
    shortlisted = [c for c in agent.candidates
                   if c.status == "shortlisted" and c.match_score > 0]
    if not shortlisted:
        shortlisted = sorted(agent.candidates, key=lambda c: c.match_score, reverse=True)[:10]

    if agent.int_slots.empty:
        raise HTTPException(status_code=503, detail="Interviewer slots not loaded")

    raw_results = agent.scheduler.schedule(shortlisted, agent.cand_slots, agent.int_slots)
    
    scheduled = []
    conflicts = []
    
    for r in raw_results:
        # Add match score to the result for UI visibility
        cand_obj = agent.pipeline.get(r["candidate_id"])
        if cand_obj:
            r["match_score"] = cand_obj.match_score / 100.0 # Normalize to 0-1 for UI
            
        if r["status"] == "scheduled":
            scheduled.append(r)
        else:
            conflicts.append(r)
            
    return {
        "scheduled": scheduled,
        "conflicts": conflicts
    }


# ─── Question generation ──────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    candidate_name: str
    job_title: Optional[str] = None
    job_description: Optional[str] = None

@app.post("/questions")
async def generate_questions(req: QuestionRequest):
    if not agent.jd:
        raise HTTPException(status_code=400, detail="Run /screen first")

    # Find candidate by name
    target = next((c for c in agent.candidates if c.name == req.candidate_name), None)
    if not target:
        raise HTTPException(status_code=404, detail="Candidate not found")

    qs = agent.question_gen.generate_questions(agent.jd, target)
    return {"questions": qs}


# ─── Leave management ─────────────────────────────────────────────────────────
@app.post("/leave/apply")
async def apply_leave(request: LeaveRequestInput):
    try:
        start = datetime.fromisoformat(request.start_date)
        end   = datetime.fromisoformat(request.end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    req_obj = LeaveRequest(
        request_id="LR_" + datetime.now().strftime("%H%M%S"),
        employee_id=request.employee_id,
        leave_type=request.leave_type,
        start_date=start,
        end_date=end,
        reason=request.reason or "",
    )
    policy = LeavePolicy(
        leave_type=request.leave_type,
        annual_quota=20,
        max_consecutive_days=5,
        min_notice_days=0,
        requires_document=(request.leave_type.lower() == "medical"),
    )
    return agent.leave_mgr.process_request(req_obj, policy)


@app.get("/leave/logs")
async def get_leave_logs():
    return agent.leave_mgr.decision_logs


@app.get("/leave/metrics")
async def get_leave_metrics():
    return agent.leave_mgr.get_metrics()


@app.get("/leave/evaluation")
async def get_leave_evaluation():
    metrics = agent.leave_mgr.get_metrics()
    return {
        "summary": {
            "accuracy":  metrics["summary"]["accuracy"],
            "precision": metrics["summary"]["precision"],
            "recall":    metrics["summary"]["recall"],
        },
        "confusion_matrix": {
            "TP": metrics["TP"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
        },
        "model_version": "v3.0-RandomForest-Multiclass",
        "total_decisions": metrics["total"],
    }


@app.get("/final_results")
async def get_final_results():
    """Comprehensive intelligence report for the ResultsPage.jsx."""
    return {
        "team_id": "ANTIGRAVITY-CORE-V3",
        "track": "Advanced_Agentic_Coding",
        "results": {
            "pipeline": {
                c.candidate_id: {
                    "name": c.name,
                    "status": c.status,
                    "score": c.match_score / 100.0 # Normalize for UI
                }
                for c in agent.candidates if c.match_score > 0
            }
        }
    }


@app.get("/leave/employees")
async def get_leave_employees():
    """Return all real employees from the leave CSV for the UI dropdown."""
    employees = []
    for name, info in agent.leave_mgr.employee_data.items():
        employees.append({
            "name":           name,
            "department":     info.get("Department", ""),
            "position":       info.get("Position", ""),
            "remaining_days": int(info.get("Remaining Leaves", 0)),
            "leave_type":     info.get("Leave Type", ""),
        })
    return employees


# ─── Pipeline overview ────────────────────────────────────────────────────────
@app.get("/pipeline")
async def get_pipeline():
    return {
        "total_candidates": len(agent.candidates),
        "jd_loaded":        agent.jd is not None,
        "results": {
            "pipeline": {
                cid: {"name": c.name, "status": c.status, "score": c.match_score}
                for cid, c in agent.pipeline.items()
            }
        },
    }


# ─── Adaptive Interview endpoints ─────────────────────────────────────────────
from adaptive_interview_llm import AdaptiveInterviewEngine, AdaptiveQuestionGenerator

# Initialize the adaptive interview engine
adaptive_engine = AdaptiveInterviewEngine(threshold=75)
question_generator = AdaptiveQuestionGenerator()


class InterviewRequest(BaseModel):
    candidate_name: str
    candidate_email: str
    skills: List[str]
    job_role: str


class SubmitAnswersRequest(BaseModel):
    candidate_name: str
    candidate_email: str
    skills: List[str]
    job_role: str
    answers: List[str]


@app.post("/interview/generate-questions")
async def generate_interview_questions(request: InterviewRequest):
    """Generate adaptive interview questions based on candidate skills and job role."""
    try:
        questions = question_generator.generate(
            candidate_name=request.candidate_name,
            skills=request.skills,
            job_role=request.job_role
        )
        return {
            "success": True,
            "questions": questions,
            "candidate_name": request.candidate_name,
            "job_role": request.job_role
        }
    except Exception as e:
        logger.error(f"Question generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interview/submit-answers")
async def submit_interview_answers(request: SubmitAnswersRequest):
    """Submit answers and get evaluation. If score >= threshold, send email."""
    try:
        result = adaptive_engine.run(
            candidate_name=request.candidate_name,
            candidate_email=request.candidate_email,
            skills=request.skills,
            job_role=request.job_role,
            answers=request.answers
        )
        return result
    except Exception as e:
        logger.error(f"Interview evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/interview/config")
async def get_interview_config():
    """Get interview configuration including threshold."""
    return {
        "threshold": adaptive_engine.threshold,
        "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "email_enabled": bool(os.getenv("SMTP_HOST"))
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
