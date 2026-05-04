import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.services.api.routes import router
from src.services.api.openai_routes import openai_router
from src.services.api.cockpit_routes import cockpit_router
from src.services.api.project_routes import project_router
from src.services.api.product_backlog_routes import product_backlog_router
from src.services.api.student_routes import student_router
from src.services.api.student_agent_routes import student_agent_router
from src.core.session_store import SESSION_STORE

logger = logging.getLogger("cognitwin.bootstrap")

SEED_FILE = "src/data/masked/footprints_masked.txt"


def _seed_academic_memory() -> None:
    """Populate the academic ChromaDB collection if it is empty."""
    from src.database.chroma_manager import db_manager

    count = db_manager.collection.count()
    if count > 0:
        logger.info("[BOOTSTRAP] academic_memory already has %d records — skipping ingest.", count)
        return

    logger.info("[BOOTSTRAP] academic_memory is empty — starting ingest from %s", SEED_FILE)
    try:
        from src.database.bulk_ingest import bulk_ingest_masked_data

        bulk_ingest_masked_data(SEED_FILE)
        new_count = db_manager.collection.count()
        logger.info("[BOOTSTRAP] Ingest completed — %d records now in academic_memory.", new_count)
    except Exception:
        logger.exception("[BOOTSTRAP] Ingest failed — student memory will be empty until next restart.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_academic_memory()
    yield


app = FastAPI(title="CogniTwin Middleware API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
#  Session inspection endpoints
#  Allow users (and tools) to retrieve per-agent work after a workflow run.
#
#  GET /sessions/{session_id}
#      Returns the full record for one agent session.
#      session_id examples:
#        conv-abc123def456      ← Composer / parent session
#        conv-abc123def456/po   ← Product Owner session
#        conv-abc123def456/sm   ← Scrum Master session
#        conv-abc123def456/dev  ← Developer session
#
#  GET /sessions/{session_id}/children
#      Returns all child sessions (PO, SM, Developer) for a parent session.
#
#  GET /sessions
#      Returns the 20 most recently created sessions.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id:path}")
async def get_session(session_id: str):
    """Retrieve a single agent session by ID.

    Use the child session IDs returned in the `cognitwin_session` field of
    any /v1/chat/completions response to inspect individual agent outputs.
    """
    record = SESSION_STORE.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return record


@app.get("/sessions/{session_id:path}/children")
async def get_session_children(session_id: str):
    """Return all child agent sessions for a given parent session.

    Equivalent to calling GET /sessions/<child_id> for each child listed
    in the parent's `children` array, but in one request.
    """
    children = SESSION_STORE.list_children(session_id)
    return {"session_id": session_id, "children": children}


@app.get("/sessions")
async def list_sessions(n: int = 20):
    """Return the n most recently created sessions (default 20)."""
    return {"sessions": SESSION_STORE.recent(n)}


app.include_router(router)
app.include_router(openai_router)
app.include_router(cockpit_router)
app.include_router(project_router)
app.include_router(product_backlog_router)
app.include_router(student_router)
app.include_router(student_agent_router)
