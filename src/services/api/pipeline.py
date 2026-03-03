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
    try:
        # 1. Maskeleme işlemi
        masked_text = masker.mask_data(user_text)

        # 2. Hafızaya ekleme
        doc_id = f"chat_{uuid.uuid4().hex}"
        memory.add_academic_info(masked_text, info_type="chat_footprint", doc_id=doc_id)

        # 3. Hafızadan geri çağırma (Gelen veri None ise boş liste yap)
        retrieved = memory.query_memory(masked_text)
        if retrieved is None:
            retrieved = []
            
        context = "\n".join(retrieved[:5]) if retrieved else "Hafızada ilgili bilgi bulunamadı."

        # 4. Ollama ile iletişim
        try:
            resp = chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"MEMORY:\n{context}\n\nSORU:\n{masked_text}"}
                ]
            )
            # Yanıtın varlığını kontrol et
            if 'message' in resp and 'content' in resp['message']:
                answer = resp["message"]["content"]
            else:
                answer = "Ollama'dan geçersiz yanıt formatı alındı."
        except Exception as ollama_err:
            print(f"Ollama Hatası: {ollama_err}")
            answer = f"Dil modeli şu an yanıt veremiyor. (Hata: {ollama_err})"

        return {
            "masked_input": masked_text,
            "saved_doc_id": doc_id,
            "retrieved_top5": retrieved[:5] if retrieved else [],
            "answer": answer,
            "context": context
        }

    except Exception as general_err:
        print(f"Genel Pipeline Hatası: {general_err}")
        return {
            "answer": f"İşlem sırasında bir hata oluştu: {str(general_err)}",
            "masked_input": user_text
        }