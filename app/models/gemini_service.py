#!/usr/bin/env python3
"""
Gemini AI Service for Cosmic Hunter
Provides educational responses about exoplanet science
"""

import os
import google.generativeai as genai
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

for m in genai.list_models():
    print(m.name, " -> ", m.supported_generation_methods)


class GeminiAIService:
    """Gemini AI service for exoplanet education"""
    
    def __init__(self):
        """Initialize the Gemini AI service"""
        # Read the API key from the environment variable defined in env_example.txt
        # (do NOT hard-code an API key here)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = None
        self.chat_session = None
        if self.api_key:
            try:                
                genai.configure(api_key=self.api_key)
                self.model_name = 'models/gemini-2.5-flash'
                self.model = True
                print("✅ Gemini AI service configured (API key present)")
            except Exception as e:
                print(f"❌ Failed to configure Gemini AI: {e}")
                self.model = None
        else:
            print("⚠️ GEMINI_API_KEY not found. Using fallback responses.")
            self.model = None
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the AI assistant"""
        return """You are Cosmic AI, an expert assistant specializing in exoplanet science and detection. You help users understand:

1. Exoplanet detection methods (transit, radial velocity, direct imaging, etc.)
2. Physical characteristics of exoplanets and their host stars
3. The significance of different features in exoplanet analysis
4. How machine learning models work for exoplanet classification
5. The science behind the Kepler mission and exoplanet discoveries

Key topics you should be knowledgeable about:
- Transit photometry and light curves
- Planetary radius, temperature, and habitability
- Stellar properties and their effects on planets
- The importance of feature importance in ML models
- Recent exoplanet discoveries and their significance

Always provide accurate, educational responses that help users understand exoplanet science. If asked about specific analysis results, explain what the features mean and their implications for exoplanet detection.

Keep responses concise but informative, and use analogies when helpful to explain complex concepts."""
    
    def generate_response(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using Gemini AI"""
        if not self.model:
            return self._get_fallback_response(user_message, context)

        prompt = self._build_prompt(user_message, context)

        try:
            text = self._call_genai(prompt, user_message, context)
            if text:
                return text.strip()
            else:
                return self._get_fallback_response(user_message, context)
        except Exception as e:
            print(f"❌ Gemini AI error while generating response: {e}")
            return self._get_fallback_response(user_message, context)

    def _call_genai(self, prompt: str, user_message: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Call Gemini API using official client."""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            if response and hasattr(response, "text"):
                return response.text
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
        return None


    def _extract_text_from_response(self, resp: Any) -> Optional[str]:
        """Robustly extract text from different response shapes returned by genai."""
        try:
            # Common attributes
            if resp is None:
                return None
            # Try a few known attribute names and structures
            if hasattr(resp, 'text') and resp.text:
                return resp.text
            if hasattr(resp, 'content') and resp.content:
                # content may be a list of dicts or a string
                if isinstance(resp.content, list) and len(resp.content) > 0:
                    first = resp.content[0]
                    if isinstance(first, dict) and 'text' in first:
                        return first['text']
                    if isinstance(first, dict) and 'content' in first:
                        return first['content']
                elif isinstance(resp.content, str):
                    return resp.content
            if hasattr(resp, 'output') and resp.output:
                out = resp.output
                if isinstance(out, list) and len(out) > 0:
                    first = out[0]
                    if isinstance(first, dict) and 'content' in first:
                        # content sometimes is a list of dicts
                        c = first['content']
                        if isinstance(c, list) and len(c) > 0 and 'text' in c[0]:
                            return c[0]['text']
                        if isinstance(c, str):
                            return c
            # Some clients put candidates or outputs under different keys
            if hasattr(resp, 'candidates') and resp.candidates:
                cand = resp.candidates[0]
                if isinstance(cand, dict) and 'content' in cand:
                    return cand['content']
                if hasattr(cand, 'text'):
                    return cand.text

            # Fall back to string conversion
            s = str(resp)
            if s:
                return s
        except Exception as e:
            print(f"⚠️ Failed to extract text from genai response: {e}")
        return None
    
    def _build_prompt(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the complete prompt for Gemini AI"""
        prompt_parts = [self.get_system_prompt()]
        
        # Add context if available
        if context:
            context_info = self._format_context(context)
            prompt_parts.append(f"\nCurrent Analysis Context:\n{context_info}")
        
        # Add user message
        prompt_parts.append(f"\nUser Question: {user_message}")
        
        return "\n".join(prompt_parts)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format analysis context for the AI"""
        if isinstance(context, list) and len(context) > 0:
            # Batch results
            result = context[0]  # Use first result as example
            return f"""
Recent analysis results:
- Classification: {result.get('classification', 'Unknown')}
- Confidence: {result.get('confidence', 0):.1%}
- Features analyzed: 8 physical characteristics
- Model used: XGBoost with 92.7% AUC accuracy
"""
        elif isinstance(context, dict):
            # Single result
            return f"""
Recent analysis result:
- Classification: {context.get('classification', 'Unknown')}
- Confidence: {context.get('confidence', 0):.1%}
- Features analyzed: 8 physical characteristics
- Model used: XGBoost with 92.7% AUC accuracy
"""
        return "No specific analysis context available."
    
    def _get_fallback_response(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Get fallback responses when Gemini AI is not available"""
        message_lower = user_message.lower()
        
        # Context-aware responses
        if context:
            if isinstance(context, list) and len(context) > 0:
                result = context[0]
                classification = result.get('classification', 'Unknown')
                confidence = result.get('confidence', 0)
                
                if classification == 'CONFIRMED':
                    return f"Based on your analysis showing a {confidence:.1%} confidence for a CONFIRMED exoplanet, this suggests strong evidence for a real planetary transit. The high confidence indicates that the observed features are consistent with a genuine exoplanet rather than a false positive signal."
                else:
                    return f"Your analysis shows a {confidence:.1%} confidence for a NOT_CONFIRMED result. This could indicate the signal might be a false positive, possibly caused by stellar variability, instrumental noise, or other astrophysical phenomena that mimic planetary transits."
        
        # General topic-based responses
        if any(word in message_lower for word in ['exoplanet', 'planet', 'kepler']):
            return "Exoplanets are planets that orbit stars other than our Sun. The Kepler mission has discovered thousands of exoplanets using the transit method, which detects the dimming of a star when a planet passes in front of it. Our AI model analyzes 8 key features to determine if a transit signal represents a confirmed exoplanet."
        
        elif any(word in message_lower for word in ['model', 'prediction', 'ai', 'machine learning']):
            return "Our XGBoost machine learning model achieves 92.7% AUC accuracy by analyzing 8 physical features: planetary radius (most important at 23.4%), insolation flux (16.0%), radius ratio (14.8%), equilibrium temperature, transit depth, stellar temperature, transit duration, and impact parameter. The model was trained on 9,201 exoplanet candidates from the Kepler mission."
        
        elif any(word in message_lower for word in ['feature', 'importance', 'radius', 'temperature']):
            return "The most important features for exoplanet detection are: 1) Planetary radius (23.4% importance) - larger planets create deeper transits, 2) Insolation flux (16.0%) - the amount of stellar energy the planet receives, 3) Radius ratio (14.8%) - the relative size of planet to star. These features help distinguish real exoplanets from false positives."
        
        elif any(word in message_lower for word in ['transit', 'depth', 'duration']):
            return "Transit depth measures how much the star dims during a planetary transit (in parts per million). Transit duration is how long the transit lasts (in hours). These measurements help determine the planet's size and orbital characteristics. Deeper, longer transits often indicate larger planets in wider orbits."
        
        elif any(word in message_lower for word in ['habitable', 'life', 'earth']):
            return "The habitable zone is the region around a star where liquid water could exist on a planet's surface. Our model doesn't directly assess habitability, but the features it analyzes (like insolation flux and equilibrium temperature) are important factors in determining if a planet could potentially support life."
        
        elif any(word in message_lower for word in ['kepler', 'mission', 'nasa']):
            return "The Kepler mission was a NASA space telescope that discovered thousands of exoplanets by monitoring the brightness of stars. It used the transit method to detect planets and provided the data that trained our AI model. Kepler's discoveries revolutionized our understanding of planetary systems in our galaxy."
        
        else:
            return "I'm here to help you understand exoplanet science! You can ask me about exoplanet detection methods, the features our AI model analyzes, how machine learning works for exoplanet classification, or any other questions about planetary science. What would you like to know?"

# Global instance
gemini_service = GeminiAIService()

def get_gemini_response(message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Get a response from Gemini AI service"""
    return gemini_service.generate_response(message, context)

