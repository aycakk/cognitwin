import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.services.api.routes import router
from src.services.api.openai_routes import openai_router

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


app.include_router(router)
app.include_router(openai_router)
