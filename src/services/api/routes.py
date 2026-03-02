from fastapi import APIRouter
from pydantic import BaseModel
from src.services.api.pipeline import process_user_message

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    return process_user_message(req.message)