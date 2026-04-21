from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.responses import Response, StreamingResponse
import asyncio
import hashlib
import json
import logging
import time

from src.services.api.pipeline import process_user_message

logger = logging.getLogger(__name__)

openai_router = APIRouter()

# Maximum wall-clock time for any single pipeline run (seconds).
#
# Worst-case call count (with MAX_REDO=2 per stage):
#   PO decompose_goal  (1 + 2 REDO) = 3
#   PO generate_stories(1 + 2 REDO) = 3
#   SM _sm_chat        (1, no REDO) = 1
#   Dev generate()     (1 + 2 REDO) = 3
#   PO final review    (1 + 2 REDO) = 3
#   ─────────────────────────────────
#   Total worst-case                = 13 LLM calls
#
# At ~114 s/call (local Ollama on CPU): 13 × 114 ≈ 1 482 s
# Best-case (no REDO):                   5 × 114 ≈   570 s
#
# 1 800 s (30 min) covers worst-case REDO with ~5 min spare.
_PIPELINE_TIMEOUT = 1800.0

# How often to send an SSE keepalive comment while waiting for the pipeline.
# Must be shorter than LibreChat's inactivity timeout (~120 s).
# 20 s gives a 6× safety margin.
_KEEPALIVE_INTERVAL = 20.0


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    session_id: Optional[str] = None


def _derive_session_id(req: "ChatCompletionRequest") -> str:
    """Return a stable session ID.

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


def _make_chunk(chat_id: str, now: int, model: str, delta: dict, finish_reason=None) -> str:
    """Build a single SSE data line from a delta dict."""
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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
    ]
    payload = {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "cognitwin"}
            for mid in model_ids
        ],
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )


@openai_router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # 1) Extract the most recent user message
    user_text = ""
    for m in reversed(req.messages or []):
        if m.get("role") == "user":
            user_text = m.get("content", "") or ""
            break

    session_id = _derive_session_id(req)
    model      = req.model or "llama3.2"

    def _run_pipeline() -> dict:
        """Synchronous pipeline wrapper — runs in a thread executor."""
        result = process_user_message(
            user_text,
            model=model,
            messages=req.messages,
            session_id=session_id,
        )
        return result if isinstance(result, dict) else {"answer": str(result)}

    def _extract_answer(result: dict) -> tuple[str, dict]:
        answer = result.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        if not answer.strip():
            answer = "Hata: Boş yanıt döndü."
        return answer, result.get("workflow_meta") or {}

    # ── STREAMING path ────────────────────────────────────────────────────────
    #
    # Why this structure matters:
    #   The agile workflow makes 5+ sequential LLM calls and takes 3-10 minutes.
    #   If we call process_user_message() BEFORE returning StreamingResponse,
    #   the ASGI event loop is blocked and LibreChat gets zero SSE bytes for
    #   the entire duration, triggering its inactivity timeout (~120 s).
    #
    # Fix:
    #   1. Return StreamingResponse immediately so HTTP headers reach LibreChat.
    #   2. Inside gen(), yield the role chunk first so LibreChat shows activity.
    #   3. Submit the pipeline to run_in_executor (releases the event loop).
    #   4. Poll with asyncio.shield + short timeout, yielding SSE comment
    #      keepalives every _KEEPALIVE_INTERVAL seconds.
    #      SSE comments (": …\n\n") are invisible to the user but keep the
    #      TCP connection alive, preventing LibreChat's "terminated" abort.
    #   5. When the pipeline finishes, stream content word-by-word.
    #
    if req.stream:
        async def gen():
            now_     = int(time.time())
            chat_id_ = f"chatcmpl-{now_}"

            # Role chunk — arrives in < 1 s, prevents immediate timeout
            yield _make_chunk(chat_id_, now_, model, {"role": "assistant"})
            await asyncio.sleep(0)  # flush to LibreChat before blocking work starts

            # Submit synchronous pipeline to thread executor.
            # asyncio.get_running_loop() is the correct modern API (3.10+).
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, _run_pipeline)

            final_answer = ""
            elapsed      = 0.0

            while elapsed < _PIPELINE_TIMEOUT:
                try:
                    # asyncio.shield keeps the executor future alive even when
                    # wait_for raises TimeoutError after _KEEPALIVE_INTERVAL.
                    result = await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=_KEEPALIVE_INTERVAL,
                    )
                    # Pipeline completed — break out of the polling loop
                    final_answer, _ = _extract_answer(result)
                    logger.info(
                        "openai-routes: pipeline done in %.0f s  session=%s",
                        elapsed, session_id,
                    )
                    break

                except asyncio.TimeoutError:
                    # Not done yet — send an SSE comment to keep the connection
                    # alive.  Comments are invisible in LibreChat but prevent
                    # the browser/fetch client from aborting the SSE stream.
                    elapsed += _KEEPALIVE_INTERVAL
                    yield f": keepalive {int(elapsed)}s\n\n"
                    logger.debug(
                        "openai-routes: keepalive sent  elapsed=%.0f s  session=%s",
                        elapsed, session_id,
                    )
                    continue

                except Exception as exc:
                    final_answer = f"Pipeline Hatası: {exc}"
                    logger.error(
                        "openai-routes: pipeline error  session=%s: %s",
                        session_id, exc, exc_info=True,
                    )
                    break
            else:
                # Total timeout exceeded
                final_answer = (
                    "⚠ Workflow zaman aşımına uğradı (10 dakika). "
                    "Lütfen isteği daha kısa tutun veya daha sonra tekrar deneyin."
                )
                logger.warning(
                    "openai-routes: pipeline timeout (%.0f s)  session=%s",
                    _PIPELINE_TIMEOUT, session_id,
                )

            # Stream content word-by-word so LibreChat renders progressively
            words = final_answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield _make_chunk(chat_id_, now_, model, {"content": token})
                await asyncio.sleep(0.01)

            # Close the SSE stream cleanly
            yield _make_chunk(chat_id_, now_, model, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "Connection":        "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── NON-STREAMING path ────────────────────────────────────────────────────
    # Also run in executor so the event loop is not blocked for 3-10 minutes.
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_pipeline),
            timeout=_PIPELINE_TIMEOUT,
        )
        final_answer, workflow_meta = _extract_answer(result)
    except asyncio.TimeoutError:
        final_answer  = (
            "⚠ Workflow zaman aşımına uğradı (10 dakika). "
            "Lütfen isteği daha kısa tutun veya daha sonra tekrar deneyin."
        )
        workflow_meta = {}
    except Exception as exc:
        final_answer  = f"Pipeline Hatası: {exc}"
        workflow_meta = {}

    now_    = int(time.time())
    chat_id = f"chatcmpl-{now_}"

    child_sessions = workflow_meta.get("child_sessions", [])
    session_info: dict = {
        "session_id":     session_id,
        "child_sessions": child_sessions,
    }

    response_payload = {
        "id": chat_id,
        "object": "chat.completion",
        "created": now_,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_answer},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cognitwin_session": session_info,
    }

    return Response(
        content=json.dumps(response_payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )
