import os
from typing import Optional, Dict, Any
import google.generativeai as genai

def setup_gemini():
    """Configure Gemini AI model"""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-flash-latest")
    else:
        print("⚠️ GEMINI_API_KEY missing")
        return None

async def chat_with_gemini(message: str, context: Optional[Dict[str, Any]], model):
    """Send user message to Gemini and get response"""
    if not model:
        return "⚠️ Gemini AI is not configured."
    try:
        response = model.generate_content(message)
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"❌ Gemini error: {e}"
