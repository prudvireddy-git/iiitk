import os
from dotenv import load_dotenv
load_dotenv()
# Fix Windows SSL cert bug
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
import json
from typing import List, Dict
from groq import Groq
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



# ---------------------------
# GROQ CLIENT
# ---------------------------

def groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=key)


def get_model():
    return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ---------------------------
# QUESTION GENERATOR (LLM)
# ---------------------------

class AdaptiveQuestionGenerator:

    def generate(self, candidate_name: str, skills: List[str], job_role: str) -> List[str]:

        client = groq_client()

        prompt = f"""
You are a senior technical interviewer.

Generate minimum 8 interview questions for:

Candidate: {candidate_name}
Role: {job_role}
Skills: {skills}

Rules:
- Question 1 = Basic level
- Gradually increase difficulty
- Last 2 questions must be advanced
- Cover multiple skills
- Return ONLY JSON list format:

[
  "Question 1",
  "Question 2",
  ...
]
"""

        response = client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "")

        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1:
            return json.loads(text[start:end+1])

        raise ValueError("Invalid LLM question output")


# ---------------------------
# ANSWER EVALUATOR (LLM)
# ---------------------------

class LLMAnswerEvaluator:

    def evaluate(self,
                 candidate_name: str,
                 job_role: str,
                 skills: List[str],
                 questions: List[str],
                 answers: List[str]) -> Dict:

        client = groq_client()

        qa_block = ""
        for q, a in zip(questions, answers):
            qa_block += f"\nQ: {q}\nA: {a}\n"

        prompt = f"""
You are evaluating a technical interview.

Candidate: {candidate_name}
Role: {job_role}
Skills Required: {skills}

Evaluate answers strictly.

Return ONLY valid JSON:

{{
  "score": integer 0-100,
  "verdict": "PASS" or "FAIL",
  "feedback": "short explanation",
  "per_question": [
    {{
      "question": "...",
      "quality": "poor/average/good/excellent",
      "points": integer
    }}
  ]
}}

Interview Data:
{qa_block}
"""

        response = client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "")

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            return json.loads(text[start:end+1])

        raise ValueError("Invalid LLM grading output")


# ---------------------------
# EMAIL SENDER
# ---------------------------

class EmailSender:

    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.from_email = os.getenv("FROM_EMAIL")

    def send(self, to_email: str, subject: str, body: str):

        if not self.smtp_user or not self.smtp_pass:
            print("⚠ SMTP not configured. Email skipped.")
            return

        msg = MIMEMultipart()
        msg["From"] = self.from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_pass)
            server.send_message(msg)


# ---------------------------
# MAIN INTERVIEW ENGINE
# ---------------------------

class AdaptiveInterviewEngine:

    def __init__(self, threshold: float = 75):
        self.threshold = threshold
        self.qgen = AdaptiveQuestionGenerator()
        self.evaluator = LLMAnswerEvaluator()
        self.email_sender = EmailSender()

    def run(self,
            candidate_name: str,
            candidate_email: str,
            skills: List[str],
            job_role: str,
            answers: List[str]) -> Dict:

        questions = self.qgen.generate(candidate_name, skills, job_role)

        result = self.evaluator.evaluate(
            candidate_name,
            job_role,
            skills,
            questions[:len(answers)],
            answers
        )

        score = result["score"]
        hired = score >= self.threshold

        if hired:
            subject = "Congratulations! You Are Selected 🎉"
            body = f"""
Dear {candidate_name},

Congratulations! You have been selected for {job_role}.
Your Score: {score}

HR Team
"""
        else:
            subject = "Interview Result"
            body = f"""
Dear {candidate_name},

Thank you for your time.
Your Score: {score}

Unfortunately, you were not selected.

HR Team
"""

        self.email_sender.send(candidate_email, subject, body)

        return {
            "candidate": candidate_name,
            "questions": questions,
            "evaluation": result,
            "hired": hired
        }