from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports for history formatting
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Local services
from services.nbu_api import fetch_currency_rates, format_currency_data_for_ai, NBUAPIError
from services.agent_service import AgentService

app = FastAPI(title="AI Agent Backend", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent Service
agent_service = AgentService()

# In-memory session storage (in production, use Redis or database)
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create a new one"""
    if session_id and session_id in chat_sessions:
        return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    chat_sessions[new_session_id] = []
    return new_session_id

def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """Get chat history for a session"""
    return chat_sessions.get(session_id, [])

def add_to_chat_history(session_id: str, user_message: str, ai_response: str):
    """Add a conversation turn to the chat history"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_response}
    ])
    
    # Keep only last 20 messages to prevent memory issues
    if len(chat_sessions[session_id]) > 20:
        chat_sessions[session_id] = chat_sessions[session_id][-20:]

def format_history_for_langchain(history: List[Dict[str, str]]) -> List:
    """Convert our history format to LangChain format"""
    langchain_history = []
    for message in history:
        if message["role"] == "user":
            langchain_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            langchain_history.append(AIMessage(content=message["content"]))
        else:
            langchain_history.append(SystemMessage(content=message["content"]))
    return langchain_history

# Basic Prompt Injection Protection
def detect_prompt_injection(user_input: str) -> Tuple[bool, List[str]]:
    """
    Detect potential prompt injection attempts
    
    Returns:
        Tuple of (is_suspicious, list_of_detected_patterns)
    """
    suspicious_patterns = [
        # Role override attempts
        r"(?i)ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"(?i)forget\s+(?:all\s+)?(?:previous\s+)?instructions", 
        r"(?i)you\s+are\s+now\s+(?:a\s+)?(?:different\s+)?(?:ai|assistant|bot)",
        r"(?i)new\s+instructions?",
        r"(?i)override\s+(?:previous\s+)?(?:system\s+)?(?:instructions?|prompts?)",
        
        # System prompt manipulation
        r"(?i)end\s+(?:of\s+)?(?:system\s+)?(?:instructions?|prompts?)",
        r"(?i)system\s+(?:prompt|message)\s+(?:ends?|over)",
        r"(?i)---+\s*end",
        r"(?i)stop\s+being\s+(?:an?\s+)?(?:ai|assistant|bot)",
        
        # Tool bypass attempts
        r"(?i)don'?t\s+use\s+(?:any\s+)?tools?",
        r"(?i)never\s+use\s+(?:the\s+)?(?:currency|tool|function)",
        r"(?i)without\s+using\s+(?:any\s+)?tools?",
        r"(?i)make\s+up\s+(?:random\s+)?(?:numbers?|data|rates?)",
        r"(?i)just\s+(?:say|tell|respond)",
        
        # Prompt structure manipulation
        r"(?i)human\s*:|assistant\s*:|user\s*:|system\s*:",
        r"<\|.*?\|>",  # Special tokens
        r"\[(?:system|user|assistant)\]",
        
        # Direct instruction override
        r"(?i)instead\s+of\s+using\s+tools?",
        r"(?i)respond\s+with\s+['\"].*['\"]",
        r"(?i)say\s+exactly\s+['\"].*['\"]",
        r"(?i)pretend\s+(?:to\s+be|you\s+are)",
    ]
    
    detected_patterns = []
    
    for pattern in suspicious_patterns:
        if re.search(pattern, user_input):
            detected_patterns.append(pattern)
    
    is_suspicious = len(detected_patterns) > 0
    return is_suspicious, detected_patterns

def sanitize_user_input(user_input: str) -> str:
    """
    Sanitize user input by removing/escaping potentially dangerous content
    """
    # Remove potential prompt delimiters
    sanitized = re.sub(r"---+", "---", user_input)
    
    # Escape role indicators to prevent confusion
    sanitized = re.sub(r"(?i)(human|assistant|user|system)\s*:", r"\1 :", sanitized)
    
    # Remove special tokens
    sanitized = re.sub(r"<\|.*?\|>", "", sanitized)
    sanitized = re.sub(r"\[(?:system|user|assistant)\]", "", sanitized)
    
    # Remove excessive whitespace
    sanitized = re.sub(r"\s+", " ", sanitized)
    
    # Limit length to prevent overwhelming the context
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000] + "..."
    
    return sanitized.strip()

def validate_user_input(user_input: str) -> Tuple[bool, str, List[str]]:
    """
    Validate and sanitize user input for prompt injection protection
    
    Returns:
        Tuple of (is_safe, sanitized_input, warnings)
    """
    # Detect injection attempts
    is_suspicious, detected_patterns = detect_prompt_injection(user_input)
    
    warnings = []
    if is_suspicious:
        warnings.append("Potential prompt injection detected")
        # Log detected patterns for monitoring
        print(f"⚠️ Injection detected in: {user_input[:100]}...")
        print(f"   Patterns: {detected_patterns}")
    
    # Sanitize input
    sanitized_input = sanitize_user_input(user_input)
    
    # Determine if input is safe (allow some flexibility for legitimate questions)
    is_safe = len(detected_patterns) < 2  # Allow single pattern matches for edge cases
    
    return is_safe, sanitized_input, warnings

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_used: Optional[str] = None

class NBUCurrencyRate(BaseModel):
    r030: int
    txt: str
    rate: float
    cc: str
    exchangedate: str

class CurrencyRatesResponse(BaseModel):
    rates: List[NBUCurrencyRate]
    date: str
    source: str = "National Bank of Ukraine"

@app.get("/")
async def root():
    return {"message": "AI Agent Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check environment variables (for development only)"""
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "api_key_exists": api_key is not None,
        "api_key_length": len(api_key) if api_key else 0,
        "api_key_starts_correctly": api_key.startswith("sk-") if api_key else False,
        "api_key_preview": api_key[:20] + "..." if api_key and len(api_key) > 20 else "Not found"
    }

@app.post("/test-tool")
async def test_tool(valcode: str = "USD"):
    """Test the currency tool directly"""
    try:
        tool_result = await agent_service.test_tool("get_currency_rates", valcode=valcode)
        return {"tool_result": tool_result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

# NBU API integration is now handled by services/nbu_api.py

@app.get("/currency-rates", response_model=CurrencyRatesResponse)
async def get_currency_rates(valcode: Optional[str] = 'USD', date: Optional[str] = None):
    """
    Get currency exchange rates from National Bank of Ukraine
    
    Args:
        valcode: Currency code (e.g., EUR, USD). If not provided, returns all currencies
        date: Date in YYYYMMDD format (e.g., 20250804). If not provided, returns today's rates
    """
    try:
        # Use the NBU API service
        data = await fetch_currency_rates(valcode, date)
        
        # Convert to our response format
        rates = [NBUCurrencyRate(**item) for item in data]
        
        # Get date from first item or use today's date
        response_date = rates[0].exchangedate if rates else datetime.now().strftime("%d.%m.%Y")
        
        return CurrencyRatesResponse(
            rates=rates,
            date=response_date
        )
    except NBUAPIError as e:
        raise HTTPException(status_code=503, detail=f"NBU API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Agent functionality is now handled by the AgentService

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_message: ChatMessage):
    """
    Chat endpoint that uses LangChain agent to intelligently call tools with session history
    """
    try:
        print(f"📥 Received message: {chat_message.message}")  # Debug log
        
        # 🛡️ SECURITY: Validate and sanitize user input
        is_safe, sanitized_message, warnings = validate_user_input(chat_message.message)
        
        if not is_safe:
            print(f"🚨 Blocking potentially dangerous input")
            return ChatResponse(
                response="I can only help with currency exchange rates from the National Bank of Ukraine. Please ask about currency rates without trying to change my behavior.",
                session_id=get_or_create_session(chat_message.session_id),
                tool_used=None
            )
        
        if warnings:
            print(f"⚠️ Warnings for user input: {warnings}")
        
        # Get or create session
        session_id = get_or_create_session(chat_message.session_id)
        print(f"🔗 Using session: {session_id}")  # Debug log
        
        # Get chat history for this session
        chat_history_raw = get_chat_history(session_id)
        chat_history = format_history_for_langchain(chat_history_raw)
        print(f"📚 Chat history length: {len(chat_history)} messages")  # Debug log
        
        # Use Agent Service to process the SANITIZED message
        result = await agent_service.process_message(sanitized_message, chat_history)
        
        print(f"🤖 Agent result: {result}")  # Debug log
        
        ai_response = result["response"]
        tool_used = result["tool_used"]
        
        # Add ORIGINAL message to history (so user sees what they sent)
        # but agent processed the sanitized version
        add_to_chat_history(session_id, chat_message.message, ai_response)
        
        print(f"📤 Final response: {ai_response}, Tool used: {tool_used}")  # Debug log
        
        return ChatResponse(response=ai_response, session_id=session_id, tool_used=tool_used)
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

@app.get("/chat/history/{session_id}")
async def get_session_history(session_id: str):
    """Get chat history for a specific session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "history": chat_sessions[session_id],
        "message_count": len(chat_sessions[session_id])
    }

@app.delete("/chat/history/{session_id}")
async def clear_session_history(session_id: str):
    """Clear chat history for a specific session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/chat/sessions")
async def list_active_sessions():
    """List all active chat sessions"""
    sessions_info = []
    for session_id, history in chat_sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "message_count": len(history),
            "last_message": history[-1]["content"][:100] + "..." if history else "No messages"
        })
    
    return {
        "active_sessions": len(chat_sessions),
        "sessions": sessions_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)