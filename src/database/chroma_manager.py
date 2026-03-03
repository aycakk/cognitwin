import chromadb
import os

class ChromaManager:
    def __init__(self):
        
        # Veritabanı dosyalarını projenin 'data' klasöründe saklıyoruz
        db_path = os.path.join(os.getcwd(), "data", "database")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 'academic_memory' adında bir hafıza alanı oluşturuyoruz [cite: 39]
        self.collection = self.client.get_or_create_collection(name="academic_memory")
    def _get_collection(self, info_type: str):
     name = info_type if info_type else "default"
     return self.client.get_or_create_collection(name=name)
        

    def add_academic_info(self, text: str, info_type: str, doc_id: str):
     collection = self._get_collection(info_type)
     collection.add(documents=[text], ids=[doc_id])
     print(f"Hafızaya kaydedildi: {text}")

    def query_memory(self, question: str, info_type: str = "footprints", n_results: int = 5):
     collection = self._get_collection(info_type)
     res = collection.query(query_texts=[question], n_results=n_results)
     return res.get("documents", [[]])[0]
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