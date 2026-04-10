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

class ChromaManager:
    """
    CogniTwin - Vektör Veritabanı ve Hafıza Yönetim Merkezi.
    Hem kalıcı veri saklama (Persistence) hem de tutarlılık denetimi (D-07) yapar.
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
        self.collection = self.client.get_or_create_collection(name="academic_memory")

    def add_academic_info(self, text, metadata=None, doc_id=None):
        """Akademik bilgiyi temizleyerek hafızaya kaydeder."""
        if metadata is None: metadata = {}
        if doc_id is None: doc_id = str(uuid.uuid4())

        # Metadata temizliği (Chroma sadece basit tipleri kabul eder)
        safe_metadata = {}
        for k, v in metadata.items():
            if v is not None and isinstance(v, (str, int, float, bool)):
                safe_metadata[str(k)] = v
            else:
                safe_metadata[str(k)] = str(v)

        self.collection.add(
            documents=[text],
            metadatas=[safe_metadata],
            ids=[doc_id]
        )

    def query_memory(self, question, n_results=10):
        """Soruyla ilgili en yakın kayıtları döndürür."""
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results
            )
            return results.get("documents", [[]])[0] or []
        except Exception as e:
            logger.warning("[CHROMA] Query error: %s", e)
            return []

    def check_consistency(self, question, n_results=5):
        """
        D-07 Consistency Check: Hafızada çelişkili bilgi olup olmadığını denetler.
        Örn: Aynı ders için iki farklı vize saati varsa True döner.
        """
        results = self.collection.query(query_texts=[question], n_results=n_results)
        docs = results.get("documents", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []

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
            except: logs = []
        
        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

# Singleton Instance
db_manager = ChromaManager()

if __name__ == "__main__":
    count = db_manager.collection.count()
    print(f"[CHROMA] Record count: {count}")