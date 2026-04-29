from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.responses import Response, StreamingResponse
import asyncio
import hashlib
import json
import time

# Kendi pipeline fonksiyonun
from src.services.api.pipeline import process_user_message

openai_router = APIRouter()


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    # Optional session/conversation tracking.
    # Custom clients can pass a stable ID; LibreChat can be configured to
    # include it.  When omitted, a deterministic ID is derived from the
    # first user message so the same conversation always maps to the same
    # parent session.
    session_id: Optional[str] = None


def _derive_session_id(req: "ChatCompletionRequest") -> str:
    """Return a stable session ID for this request.

    Priority:
      1. Explicit session_id in the request body.
      2. Deterministic hash of first user message + model name.
    """
    if req.session_id:
        return req.session_id.strip()
    first_user = ""
    for m in req.messages or []:
        if m.get("role") == "user":
            first_user = str(m.get("content", ""))
            break
    seed = f"{req.model or 'default'}::{first_user[:200]}"
    return "conv-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


@openai_router.get("/v1/models")
async def list_models():
    now = int(time.time())
    model_ids = [
        "llama3.2",
        "cognitwin-student-llm",
        "cognitwin-developer",
        "cognitwin-scrum",
        "cognitwin-product-owner",
        "cognitwin-composer",
        "cognitwin-buyer",
    ]
    payload = {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": now,
                "owned_by": "cognitwin",
            }
            for mid in model_ids
        ],
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )


@openai_router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # 1) En son USER mesajını güvenli seç
    user_text = ""
    for m in reversed(req.messages or []):
        if m.get("role") == "user":
            user_text = m.get("content", "") or ""
            break

    # 1b) Oturum kimliğini belirle — explicit veya türetilmiş
    session_id = _derive_session_id(req)

    # 2) Pipeline çalıştır
    try:
        result = process_user_message(
            user_text,
            model=req.model or "llama3.2",
            messages=req.messages,
            session_id=session_id,
        )

        if isinstance(result, dict):
            final_answer = result.get("answer", "")
            if not isinstance(final_answer, str):
                final_answer = str(final_answer)
            if not final_answer.strip():
                final_answer = "Hata: Boş yanıt döndü."
            # Carry workflow session metadata from pipeline result.
            workflow_meta = result.get("workflow_meta", {})
        else:
            final_answer  = str(result)
            workflow_meta = {}

    except Exception as e:
        final_answer  = f"Pipeline Hatası: {str(e)}"
        workflow_meta = {}

    now     = int(time.time())
    model   = req.model or "llama3.2"
    chat_id = f"chatcmpl-{now}"

    # Child session IDs to surface in the response so clients can build
    # inspection URLs like GET /sessions/<child_id>.
    child_sessions = workflow_meta.get("child_sessions", [])
    session_info: dict = {
        "session_id":     session_id,
        "child_sessions": child_sessions,
    }

    # 3) Stream varsa SSE formatında dön
    #    OpenAI uyumlu: role chunk → içerik chunk'ları → stop chunk → [DONE]
    if req.stream:
        async def gen():
            # Chunk 1: role only (no content)
            role_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

            # Content chunks: split by words to simulate incremental streaming
            words = final_answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                content_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            # Stop chunk
            stop_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=headers,
        )

    # 4) Stream yoksa normal OpenAI chat.completion dön
    response_payload = {
        "id": chat_id,
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_answer,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        # Non-standard but harmless extra field — clients can read child
        # session IDs from here and call GET /sessions/<id> to inspect each
        # agent's work.  OpenAI-only clients silently ignore unknown fields.
        "cognitwin_session": session_info,
    }

    # Türkçe karakterler bozulmasın:
    return Response(
        content=json.dumps(response_payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )
