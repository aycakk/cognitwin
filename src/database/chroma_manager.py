import os
import json
import uuid
from datetime import datetime

# Chroma telemetry kapat (Hata mesajlarını temizler)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb

class ChromaManager:
    def __init__(self):
        # Veritabanı dosyalarını projenin 'data' klasöründe saklıyoruz
        db_path = os.path.join(os.getcwd(), "data", "database")
        self.client = chromadb.PersistentClient(path=db_path)

        # 'academic_memory' adında bir hafıza alanı oluşturuyoruz
        self.collection = self.client.get_or_create_collection(name="academic_memory")

    def add_academic_info(self, text, metadata=None, doc_id=None):
        """Maskelenmiş veriyi hafızaya kaydeder."""
        if metadata is None:
            metadata = {}

        safe_metadata = {}
        for k, v in metadata.items():
            if v is None: continue
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
        print(f"Hafızaya kaydedildi: {doc_id}")

    def query_memory(self, question, n_results=2):
        """Soruyla ilgili en yakın bilgiyi hafızadan bulur."""
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return results["documents"][0] if results.get("documents") else ["Bilgi bulunamadı."]

    def log_consistency_check(self, details):
        """Dashboard (D-07) için sonuçları logs klasörüne JSON olarak kaydeder."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        # Logs klasörü yoksa oluştur
        log_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, "consistency_logs.json")
        
        # Mevcut logları oku veya yeni liste oluştur
        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                try: logs = json.load(f)
                except: logs = []
        
        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

    def check_consistency(self, question, n_results=5):
        """Hafızada çelişki denetimi yapar ve sonucu loglar."""
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

        # DASHBOARD LOGLAMA BURADA ÇALIŞIYOR
        self.log_consistency_check(details)
        
        return has_conflict, details

if __name__ == "__main__":
    db = ChromaManager()
    print("Melih, sistem tüm özellikleriyle hazır!")