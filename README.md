# AI Teaching Assistant

Angular front end + FastAPI back end for a mentor-style teaching agent (Java, Logical Reasoning, Aptitude, Data Structures, Full Stack).

## Run backend (FastAPI)
1) Create venv (optional) and install deps:
```
cd backend
python -m venv .venv
.venv\\Scripts\\activate  # PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```
2) Set `OPENAI_API_KEY` in your environment (or run without it for mocked text).
3) Start API:
```
uvicorn app.main:app --reload --port 8000
```

## Run frontend (Angular)
```
cd frontend
npm install
npm start
```
Frontend expects the API at `http://localhost:8000`.