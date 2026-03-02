from fastapi import APIRouter
from pydantic import BaseModel
import time
from typing import Optional, List, Dict, Any

from src.services.api.pipeline import process_user_message

openai_router = APIRouter()
DEFAULT_MODEL = "llama3.2"

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]

@openai_router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    print("OPENAI_ROUTE_HIT ✅")

    # ✅ content her durumda tanımlı olsun
    content = ""

    # son user mesajını al
    user_text = ""
    for m in reversed(req.messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            user_text = m.get("content") or ""
            break

    # pipeline çağır
    try:
        result = process_user_message(user_text)
        content = result.get("answer", "")
        if not isinstance(content, str):
            content = str(content)
        if content.strip() == "":
            content = "(empty answer from middleware)"
    except Exception as e:
        content = f"Middleware error: {e}"

    # ✅ damgayı en sonda ekle
    content = "MW_OK ✅ " + content

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }