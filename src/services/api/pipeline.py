import re
import uuid
from pathlib import Path

from ollama import chat
from src.utils.masker import PIIMasker
from src.database.chroma_manager import ChromaManager
from src.services.api.ttl_store import TTLStore

# --- TTL SETUP ---
ttl = TTLStore()

BASE_DIR = Path(__file__).resolve().parents[3]  # CogniTwin kök klasörü
TTL_DIR = BASE_DIR / "ontologies"

try:
    ttl.load(
        str(TTL_DIR / "student_ontology.ttl"),
        str(TTL_DIR / "cognitwin-upper.ttl"),
    )
    print(f"[TTL] Loaded: {TTL_DIR}")
except Exception as e:
    print(f"[TTL] Load failed: {e}")

SYSTEM_PROMPT = (
    "Sen bir Student ajanısın.\n"
    "SADECE MEMORY bölümünde verilen bilgilere dayanarak cevap verebilirsin.\n"
    "MEMORY'de ilgili bilgi yoksa aynen şu cümleyi yaz: 'Bunu hafızamda bulamadım.'\n"
    "Tahmin yapma. Soru sorma. Kod yazma.\n"
)

masker = PIIMasker()
memory = ChromaManager()


def extract_course_code(text: str) -> str | None:
    # COM8090, CS101 gibi
    m = re.search(r"\b([A-Z]{2,4}\d{3,4})\b", (text or "").upper())
    return m.group(1) if m else None


def process_user_message(user_text: str) -> dict:
    if "title for the conversation" in user_text.lower():
     return {"answer": "Conversation Title"}
    try:
        # 1) Mask
        masked_text = masker.mask_data(user_text)

        # 2) Save to Chroma
        doc_id = f"chat_{uuid.uuid4().hex}"
        memory.add_academic_info(
          masked_text,
          metadata={"type": "chat_footprint"},
          doc_id=doc_id
)

        # 3) Retrieve
        retrieved = memory.query_memory(masked_text, n_results=5)
        if retrieved is None:
            retrieved = []

        context = "\n".join(retrieved[:5]) if retrieved else "Bunu hafızamda bulamadım."

        # 4) LLM
        resp = chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"MEMORY:\n{context}\n\nSORU:\n{masked_text}"},
            ],
        )

        answer = resp.get("message", {}).get("content", "Bunu hafızamda bulamadım.")

        return {
            "masked_input": masked_text,
            "saved_doc_id": doc_id,
            "retrieved_top5": retrieved[:5] if retrieved else [],
            "answer": answer,
            "context": context,
        }

    except Exception as e:
        return {
            "answer": f"İşlem sırasında bir hata oluştu: {str(e)}",
            "masked_input": user_text,
        }