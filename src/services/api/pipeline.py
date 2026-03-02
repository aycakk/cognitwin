from src.utils.masker import PIIMasker
from src.database.chroma_manager import ChromaManager
from ollama import chat
import uuid

SYSTEM_PROMPT = (
    "Sen bir Student ajanısın. MEMORY içindeki [MASKED] etiketli veriler, kullanıcının gerçek ama gizlenmiş bilgileridir.\n"
    "Bu etiketleri (örn: [STUDENT_ID_MASKED]) birer 'anahtar' gibi düşün. Eğer hafızada bu anahtara karşılık gelen bir bilgi (sınav saati, sınıf vb.) varsa, kullanıcıya o bilgiyi sun.\n"
    "Eşleşme varsa cevap ver, yoksa 'Bunu hafızamda bulamadım' yaz."
)

masker = PIIMasker()
memory = ChromaManager()

def process_user_message(user_text: str) -> dict:
    masked_text = masker.mask_data(user_text)

    doc_id = f"chat_{uuid.uuid4().hex}"
    memory.add_academic_info(masked_text, info_type="chat_footprint", doc_id=doc_id)

    retrieved = memory.query_memory(masked_text) or []
    context = "\n".join(retrieved[:5])

    resp = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"MEMORY:\n{context}\n\nSORU:\n{masked_text}"}
        ]
    )
    answer = resp["message"]["content"]

    return {
        "masked_input": masked_text,
        "saved_doc_id": doc_id,
        "retrieved_top5": retrieved[:5],
        "answer": answer,
        "context": context
    }