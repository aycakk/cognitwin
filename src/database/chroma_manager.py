import chromadb
import os

class ChromaManager:
    def __init__(self):
        # Veritabanı dosyalarını projenin 'data' klasöründe saklıyoruz
        db_path = os.path.join(os.getcwd(), "data", "database")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 'academic_memory' adında bir hafıza alanı oluşturuyoruz [cite: 39]
        self.collection = self.client.get_or_create_collection(name="academic_memory")

    def add_academic_info(self, text, info_type, doc_id):
        """Üye 2'den gelen maskelenmiş veriyi hafızaya kaydeder [cite: 39]"""
        self.collection.add(
            documents=[text],
            metadatas=[{"category": info_type}],
            ids=[doc_id]
        )
        print(f"Hafızaya kaydedildi: {text}")

    def query_memory(self, question):
        """Soruyla ilgili en yakın bilgiyi hafızadan bulur (RAG Akışı) [cite: 40]"""
        results = self.collection.query(
            query_texts=[question],
            n_results=2
        )
        return results['documents'][0] if results['documents'] else ["Bilgi bulunamadı."]

# --- TEST BÖLÜMÜ ---
if __name__ == "__main__":
    db = ChromaManager()
    
    # Test verisi ekleyelim (Bu veriler normalde footprints.txt'den gelecek) [cite: 27]
    db.add_academic_info("[DERS] vize sınavı 20 Mart'ta.", "vize", "test_1")
    
    # Hafızadan sorgulama yapalım
    soru = "Vize ne zaman?"
    cevap = db.query_memory(soru)
    print(f"\nSoru: {soru}")
    print(f"Hafızadan Gelen Bilgi: {cevap}")