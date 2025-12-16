import os
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="AI Teaching Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LessonStepRequest(BaseModel):
    subject: str = Field(..., examples=["Java", "Logical Reasoning"])
    topic: str = Field(..., examples=["Classes and Objects"])
    level: str = Field("beginner", examples=["beginner", "intermediate"])
    session_id: Optional[str] = None
    last_answer: Optional[str] = None
    confusion: bool = False
    misconceptions: Optional[List[str]] = None


class LessonStepResponse(BaseModel):
    session_id: str
    step: str
    checkpoint_question: str
    recap: str


class PracticeRequest(BaseModel):
    subject: str
    topic: str
    level: str = "beginner"
    session_id: Optional[str] = None


class PracticeItem(BaseModel):
    question: str
    kind: str = Field("concept", examples=["concept", "applied", "code"])
    answer: Optional[str] = None


class PracticeResponse(BaseModel):
    session_id: str
    practice: List[PracticeItem]


SYSTEM_PROMPT = """You are an AI Teaching Assistant.
Goal:
- Teach concepts from scratch
- Explain using simple language
- Use real-life examples
- Build logic step-by-step

Teaching Style:
- Start from basics
- Use examples and analogies
- Ask small checkpoint questions
- Avoid unnecessary jargon

Subjects:
- Java
- Logical Reasoning
- Aptitude
- Data Structures
- Full Stack Development

Rules:
- Teach like a mentor, not a textbook
- If student is confused, re-explain differently
- Provide practice questions at the end
- Keep responses concise; focus on clarity.
"""

# In-memory session store. Replace with a DB later.
SESSIONS: Dict[str, Dict] = {}

client = None
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)


def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in SESSIONS:
        return session_id
    new_id = session_id or str(uuid.uuid4())
    SESSIONS[new_id] = {"history": []}
    return new_id


def build_messages(req: LessonStepRequest) -> List[Dict]:
    history = SESSIONS.get(req.session_id, {}).get("history", [])
    user_note = f"Subject: {req.subject}. Topic: {req.topic}. Level: {req.level}."
    if req.misconceptions:
        user_note += f" Known misconceptions: {', '.join(req.misconceptions)}."
    if req.last_answer:
        user_note += f" Learner previous answer: {req.last_answer}."
    if req.confusion:
        user_note += " Learner is confused; re-explain with a different analogy."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_note})
    return messages


def call_model(messages: List[Dict], output_kind: str = "lesson") -> str:
    if client is None:
        # Offline fallback to keep API shape working during local dev.
        if output_kind == "practice":
            return "Q1) Concept check.\nQ2) Applied scenario.\nQ3) Code tweak."
        return "Here is a concise teaching step with a checkpoint question."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6,
        max_tokens=500,
    )
    return response.choices[0].message.content


def split_lesson_output(text: str) -> (str, str, str):
    # Expecting the model to return step + checkpoint + recap; use simple parsing.
    checkpoint = "Checkpoint: What is one key idea here?"
    recap = "Quick recap: key idea in one line."
    parts = text.split("Checkpoint:", 1)
    step_part = parts[0].strip()
    if len(parts) > 1:
        rest = parts[1].split("Recap:", 1)
        checkpoint = "Checkpoint: " + rest[0].strip()
        if len(rest) > 1:
            recap = "Recap: " + rest[1].strip()
    return step_part, checkpoint, recap


@app.post("/lesson-step", response_model=LessonStepResponse)
def lesson_step(req: LessonStepRequest):
    session_id = get_or_create_session(req.session_id)
    req.session_id = session_id

    messages = build_messages(req)
    messages.append(
        {
            "role": "user",
            "content": "Generate a single teaching step. Include one checkpoint question prefixed with 'Checkpoint:'. End with a brief 'Recap:' line.",
        }
    )
    try:
        raw = call_model(messages, output_kind="lesson")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    step, checkpoint, recap = split_lesson_output(raw)

    SESSIONS[session_id]["history"].append({"role": "assistant", "content": raw})

    return LessonStepResponse(
        session_id=session_id, step=step, checkpoint_question=checkpoint, recap=recap
    )


@app.post("/practice", response_model=PracticeResponse)
def practice(req: PracticeRequest):
    session_id = get_or_create_session(req.session_id)
    prompt = (
        f"Create 5 practice questions for subject {req.subject}, topic {req.topic}, "
        f"level {req.level}. Mix conceptual, applied, and one small code or worked example. "
        "Return numbered items."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = call_model(messages, output_kind="practice")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    practice_items: List[PracticeItem] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        # Strip leading numbering like "1) " or "1. "
        cleaned = cleaned.lstrip("0123456789). ")
        kind = "concept"
        if "code" in cleaned.lower():
            kind = "code"
        elif "apply" in cleaned.lower() or "scenario" in cleaned.lower():
            kind = "applied"
        practice_items.append(PracticeItem(question=cleaned, kind=kind))

    if not practice_items:
        practice_items.append(
            PracticeItem(
                question="Describe one key idea from the lesson in your own words.",
                kind="concept",
            )
        )

    return PracticeResponse(session_id=session_id, practice=practice_items)


@app.get("/health")
def health():
    return {"status": "ok"}

