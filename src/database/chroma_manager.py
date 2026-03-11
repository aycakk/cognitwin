import os
import json
import uuid
from datetime import datetime

# Chroma telemetry kapat
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb


class ChromaManager:
    def __init__(self):
        # Proje kök dizinini sabit olarak belirler
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # Veritabanı dosyalarını proje içindeki data/database klasöründe saklar
        db_path = os.path.join(project_root, "data", "database")
        os.makedirs(db_path, exist_ok=True)

        print(f"[CHROMA] Using database path: {db_path}")

        self.client = chromadb.PersistentClient(path=db_path)

        # Tek bir sabit collection adı kullanılır
        self.collection = self.client.get_or_create_collection(name="academic_memory")

    def add_academic_info(self, text, metadata=None, doc_id=None):
        # Maskelenmiş veriyi hafızaya kaydeder
        if metadata is None:
            metadata = {}

        safe_metadata = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                safe_metadata[str(k)] = v
            else:
                safe_metadata[str(k)] = str(v)

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        self.collection.add(
            documents=[text],
            metadatas=[safe_metadata],
            ids=[doc_id]
        )
        print(f"[CHROMA] Saved: {doc_id}")

    def query_memory(self, question, n_results=5):
        # Soruyla ilgili en yakın bilgiyi hafızadan bulur
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        documents = results.get("documents", [])
        if documents and len(documents) > 0:
            return documents[0]

        return []

    def log_consistency_check(self, details):
        # Dashboard için sonuçları logs klasörüne JSON olarak kaydeder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "consistency_logs.json")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "details": details
        }

        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except Exception:
                    logs = []

        logs.append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

    def check_consistency(self, question, n_results=5):
        # Hafızada çelişki denetimi yapar ve sonucu loglar
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        docs = results.get("documents", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []

        if len(docs) <= 1:
            return False, {"reason": "Yeterli sonuç yok", "matches": len(docs)}

        normalized = [d.strip().lower() for d in docs if isinstance(d, str)]
        unique = list(dict.fromkeys(normalized))

        has_conflict = len(unique) >= 2

        details = {
            "question": question,
            "matches": len(docs),
            "unique_statements": len(unique),
            "top_docs": docs[:min(5, len(docs))],
            "top_metadatas": metas[:min(5, len(metas))]
        }

        self.log_consistency_check(details)

        return has_conflict, details


if __name__ == "__main__":
    db = ChromaManager()
    print("ChromaManager ready.")