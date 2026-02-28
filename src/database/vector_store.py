import os
import uuid
import chromadb

# DB dosyaları repo içinde dursun
DB_DIR = os.path.join(os.getcwd(), "static", "chromadb")
COLLECTION_NAME = "academic_memory"

def _collection():
   from chromadb.config import Settings

client = chromadb.PersistentClient(
    path=DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def add_memory(text: str, metadata: dict | None = None, memory_id: str | None = None) -> str:
    """
    Maskelenmiş bir footprint'i (tek satır/metin) hafızaya yazar.
    """
    if not text or not text.strip():
        raise ValueError("text is empty")

    col = _collection()
    if metadata is None:
        metadata = {}

    if memory_id is None:
        memory_id = str(uuid.uuid4())

    col.upsert(ids=[memory_id], documents=[text], metadatas=[metadata])
    return memory_id

def search_memory(query: str, k: int = 5) -> list[dict]:
    """
    Soruya göre en ilgili k kayıtları döndürür.
    """
    col = _collection()
    res = col.query(query_texts=[query], n_results=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    out = []
    for i, d in enumerate(docs):
        out.append({
            "id": ids[i] if i < len(ids) else None,
            "text": d,
            "meta": metas[i] if i < len(metas) else {}
        })
    return out
