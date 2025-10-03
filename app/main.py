from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-latest")
else:
    model = None
    print("⚠️ GEMINI_API_KEY missing")

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    timestamp: str
    context_used: bool = False

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not model:
        reply = "⚠️ Gemini AI is not configured."
    else:
        try:
            response = model.generate_content(request.message)
            reply = response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            reply = f"❌ Gemini error: {e}"

    return ChatResponse(
        reply=reply,
        timestamp=datetime.now().isoformat(),
        context_used=bool(request.context)
    )
