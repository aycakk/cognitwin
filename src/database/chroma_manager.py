import os

# Chroma telemetry kapat (posthog uyumsuzluğu hatasını keser)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"  # bazı sürümlerde bu da okunuyor

import chromadb
import os

class ChromaManager:
    def __init__(self):
        # Veritabanı dosyalarını projenin 'data' klasöründe saklıyoruz
        db_path = os.path.join(os.getcwd(), "data", "database")
        self.client = chromadb.PersistentClient(path=db_path)

        # 'academic_memory' adında bir hafıza alanı oluşturuyoruz
        self.collection = self.client.get_or_create_collection(name="academic_memory")

    def add_academic_info(self, text, metadata=None, doc_id=None):
        """
        Maskelenmiş veriyi hafızaya kaydeder.
        text: string
        metadata: dict -> örn {"role":"student","source":"footprints_masked"}
        doc_id: string (opsiyonel)
        """
        if metadata is None:
            metadata = {}

        # Chroma metadata değerleri: str/int/float/bool olmalı.
        # list/dict gibi şeyler gelirse string'e çeviriyoruz.
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
        print(f"Hafızaya kaydedildi: {doc_id}")

    def query_memory(self, question, n_results=2):
        """Soruyla ilgili en yakın bilgiyi hafızadan bulur (RAG Akışı)"""
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return results["documents"][0] if results.get("documents") else ["Bilgi bulunamadı."]

    def check_consistency(self, question, n_results=5):
        """
        Veritabanından en alakalı kayıtları çeker ve basit bir tutarlılık kontrolü yapar.
        Dönüş:
          - (has_conflict: bool, details: dict)
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        docs = results.get("documents", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []

        # Hiç sonuç yoksa çelişki yok say
        if len(docs) <= 1:
            return False, {"reason": "Yeterli sonuç yok", "matches": len(docs)}

        # Basit kontrol mantığı:
        # Aynı soru için dönen cevaplar birbirinden ÇOK farklıysa conflict diyelim.
        # (Şimdilik “bariz fark” = metinler birbirine benzemiyorsa)
        # Bu, ileride tarih/yer çıkarımıyla güçlendirilecek.

        normalized = [d.strip().lower() for d in docs if isinstance(d, str)]
        unique = list(dict.fromkeys(normalized))  # tekrarları sil

        # Eğer çok sayıda farklı cevap döndüyse uyarı ver
        # (eşik: 2+ farklı bilgi)
        has_conflict = len(unique) >= 2

        details = {
            "question": question,
            "matches": len(docs),
            "unique_statements": len(unique),
            "top_docs": docs[:min(5, len(docs))],
            "top_metadatas": metas[:min(5, len(metas))]
        }
        return has_conflict, details


# --- TEST BÖLÜMÜ ---
if __name__ == "__main__":
    db = ChromaManager()

    # Test verisi ekleyelim
    db.add_academic_info(
        "[DERS] vize sınavı 20 Mart'ta.",
        {"category": "vize", "role": "student", "source": "test"},
        "test_1"
    )

    # Hafızadan sorgulama yapalım
    soru = "Vize ne zaman?"
    cevap = db.query_memory(soru)

    print(f"\nSoru: {soru}")
    print(f"Hafızadan Gelen Bilgi: {cevap}")