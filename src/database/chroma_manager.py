import logging
import os
import json
import uuid
import chromadb
from datetime import datetime

logger = logging.getLogger(__name__)

# Chroma telemetry kapat (Terminal kirliliğini engeller)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# ---------------------------------------------------------------------------
# Namespace → collection name map.
# "academic" keeps the historical name for backward compatibility so that
# existing ChromaDB data on disk is found without migration.
# ---------------------------------------------------------------------------
NAMESPACE_MAP: dict[str, str] = {
    "academic":  "academic_memory",   # student path — original collection name
    "developer": "developer_memory",  # developer path — isolated from student data
    "agile":     "agile_memory",      # future scrum/agile documents
}


class ChromaManager:
    """
    CogniTwin - Vektör Veritabanı ve Hafıza Yönetim Merkezi.
    Hem kalıcı veri saklama (Persistence) hem de tutarlılık denetimi (D-07) yapar.

    Namespace support
    -----------------
    Each agent role writes to and reads from its own isolated collection:
      "academic"  → academic_memory   (student path; historical name preserved)
      "developer" → developer_memory  (developer path)
      "agile"     → agile_memory      (scrum / agile path)

    All legacy call-sites that use add_academic_info() and query_memory()
    continue to work unchanged — they implicitly use the "academic" namespace.
    """

    def __init__(self):
        """
        Veritabanı yolunu projenin ana dizinindeki 'static/chromadb' olarak mühürler.
        Bu sayede proje hangi klasörden çalıştırılırsa çalıştırılsın aynı veriye erişir.
        """
        # Proje kök dizinine erişim (src -> database -> CogniTwin)
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

        # Veritabanı ve Log yollarını sabitle
        self.db_path = os.path.join(project_root, "static", "chromadb")
        self.log_dir = os.path.join(project_root, "logs")

        # Klasörleri oluştur
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Kalıcı istemciyi başlat
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Legacy single-collection reference (academic/student path).
        # Kept so that existing callers using self.collection directly
        # continue to work without modification.
        self.collection = self.client.get_or_create_collection(name="academic_memory")

        # Multi-namespace collection cache.  Populated lazily on first access.
        self._collections: dict[str, chromadb.Collection] = {
            "academic": self.collection,  # reuse already-created handle
        }

    # -------------------------------------------------------------------------
    # Namespace-aware API (new)
    # -------------------------------------------------------------------------

    def get_collection(self, namespace: str) -> chromadb.Collection:
        """
        Return (and lazily create) the ChromaDB collection for *namespace*.

        Raises KeyError if namespace is not in NAMESPACE_MAP so callers
        catch misconfiguration at call-time rather than at query-time.
        """
        if namespace not in NAMESPACE_MAP:
            raise KeyError(
                f"Unknown ChromaDB namespace {namespace!r}. "
                f"Valid namespaces: {sorted(NAMESPACE_MAP)}"
            )
        if namespace not in self._collections:
            col_name = NAMESPACE_MAP[namespace]
            self._collections[namespace] = self.client.get_or_create_collection(
                name=col_name
            )
        return self._collections[namespace]

    def add_with_namespace(
        self,
        text: str,
        namespace: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> None:
        """Add *text* to the collection for *namespace*."""
        if metadata is None:
            metadata = {}
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        safe_metadata: dict = {}
        for k, v in metadata.items():
            if v is not None and isinstance(v, (str, int, float, bool)):
                safe_metadata[str(k)] = v
            else:
                safe_metadata[str(k)] = str(v)

        self.get_collection(namespace).add(
            documents=[text],
            metadatas=[safe_metadata],
            ids=[doc_id],
        )

    def query_by_namespace(
        self,
        question: str,
        namespace: str,
        n_results: int = 10,
    ) -> list[str]:
        """Query the collection for *namespace* and return matching documents."""
        try:
            results = self.get_collection(namespace).query(
                query_texts=[question],
                n_results=n_results,
            )
            return results.get("documents", [[]])[0] or []
        except Exception as exc:
            logger.warning("[CHROMA] query_by_namespace error (ns=%r): %s", namespace, exc)
            return []

    # -------------------------------------------------------------------------
    # Legacy API (backward compatible — student / academic path)
    # -------------------------------------------------------------------------

    def add_academic_info(self, text, metadata=None, doc_id=None):
        """Akademik bilgiyi temizleyerek hafızaya kaydeder (academic namespace)."""
        self.add_with_namespace(text, namespace="academic", metadata=metadata, doc_id=doc_id)

    def query_memory(self, question, n_results=10):
        """Soruyla ilgili en yakın kayıtları döndürür (academic namespace)."""
        return self.query_by_namespace(question, namespace="academic", n_results=n_results)

    def check_consistency(self, question, n_results=5):
        """
        D-07 Consistency Check: Hafızada çelişkili bilgi olup olmadığını denetler.
        Örn: Aynı ders için iki farklı vize saati varsa True döner.
        """
        results = self.collection.query(query_texts=[question], n_results=n_results)
        docs = results.get("documents", [[]])[0] or []

        if len(docs) <= 1:
            return False, {"reason": "Yetersiz veri", "matches": len(docs)}

        # Küçük harf normalizasyonu ile benzersiz kayıtları bul
        unique_statements = list(dict.fromkeys([d.strip().lower() for d in docs if isinstance(d, str)]))
        has_conflict = len(unique_statements) >= 2

        details = {
            "question": question,
            "matches": len(docs),
            "conflict_detected": has_conflict,
            "unique_count": len(unique_statements),
            "top_docs": docs[:5]
        }

        self._log_to_json(details)
        return has_conflict, details

    def _log_to_json(self, details):
        """Dashboard entegrasyonu için JSON log üretir."""
        log_file = os.path.join(self.log_dir, "consistency_logs.json")
        log_entry = {"timestamp": datetime.now().isoformat(), "details": details}

        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []

        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)


# Singleton Instance
db_manager = ChromaManager()

if __name__ == "__main__":
    count = db_manager.collection.count()
    print(f"[CHROMA] Record count (academic): {count}")
    for ns in NAMESPACE_MAP:
        col = db_manager.get_collection(ns)
        print(f"[CHROMA] namespace={ns!r} collection={col.name!r} count={col.count()}")
