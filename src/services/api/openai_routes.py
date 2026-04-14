from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.responses import Response, StreamingResponse
import asyncio
import json
import time

# Kendi pipeline fonksiyonun
from src.services.api.pipeline import process_user_message

openai_router = APIRouter()


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False


@openai_router.get("/v1/models")
async def list_models():
    now = int(time.time())
    model_ids = ["llama3.2", "cognitwin-student-llm", "cognitwin-developer", "cognitwin-scrum"]
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
    # 1) En son USER mesajını güvenli seç (son eleman her zaman user olmayabilir)
    user_text = ""
    for m in reversed(req.messages or []):
        if m.get("role") == "user":
            user_text = m.get("content", "") or ""
            break

    # 2) Pipeline çalıştır
    try:
        result = process_user_message(
            user_text,
            model=req.model or "llama3.2",
            messages=req.messages,
        )

        if isinstance(result, dict):
            final_answer = result.get("answer", "")
            if not isinstance(final_answer, str):
                final_answer = str(final_answer)
            if not final_answer.strip():
                final_answer = "Hata: Boş yanıt döndü."
        else:
            final_answer = str(result)

    except Exception as e:
        final_answer = f"Pipeline Hatası: {str(e)}"

    now = int(time.time())
    model = req.model or "llama3.2"
    chat_id = f"chatcmpl-{now}"

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
    }

    # Türkçe karakterler bozulmasın:
    return Response(
        content=json.dumps(response_payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )