from fastapi import FastAPI
from src.services.api.routes import router
from src.services.api.openai_routes import openai_router

app = FastAPI(title="CogniTwin Middleware API")


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(router)
app.include_router(openai_router)