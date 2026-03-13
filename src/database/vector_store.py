import os
import uuid
import chromadb
from chromadb.config import Settings


# Proje kök dizinini (CogniTwin) bulup veritabanı yolunu static/chromadb'ye kilitler.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(BASE_DIR, "static", "chromadb")
COLLECTION_NAME = "academic_memory"

def _get_collection():
    """
    ChromaDB bağlantısını sağlar ve ilgili koleksiyonu döndürür.
    Klasör yoksa otomatik olarak oluşturur.
    """
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR, exist_ok=True)

    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)

# ─────────────────────────────────────────────────────────────────────────────
#  BÖLÜM 1 ▸ HAFIZA YÖNETİMİ VE ARAMA 
# ─────────────────────────────────────────────────────────────────────────────

def add_memory(text: str, metadata: dict | None = None, memory_id: str | None = None) -> str:
    """
    Maskelenmiş bir veriyi (footprint) vektör veritabanına mühürler.
    'Upsert' kullanarak mükerrer kayıtları engeller.
    """
    if not text or not text.strip():
        return ""

    col = _get_collection()
    metadata = metadata or {}
    memory_id = memory_id or str(uuid.uuid4())

    # Veri ekleme işlemi
    col.upsert(ids=[memory_id], documents=[text], metadatas=[metadata])
    return memory_id

def search_memory(query: str, k: int = 25) -> list[dict]:
    """
    Semantik arama yaparak kullanıcı sorgusuyla en ilgili 'k' adet kaydı döndürür.
    k=25 değeri, derin muhakeme (reasoning) için optimize edilmiştir.
    """
    if not query or not query.strip():
        return []

    col = _get_collection()
    
    # Vektör uzayında arama yapma
    res = col.query(query_texts=[query], n_results=k)

    # Sonuçları pipeline için standart bir listeye dönüştürme
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    return [
        {"id": ids[i], "text": docs[i], "meta": metas[i]}
        for i in range(len(docs))
    ]

if __name__ == "__main__":
    # Terminal bilgilendirmesi
    print(f"📂 [BİLGİ] Veritabanı Yolu: {DB_DIR}")
    print(f"📊 [BİLGİ] Toplam Kayıt Sayısı: {_get_collection().count()}")