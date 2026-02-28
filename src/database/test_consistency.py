from src.database.chroma_manager import ChromaManager


def run_consistency_test():
    db = ChromaManager()
    
    print("--- 1. Senaryo: Çelişkili Veri Girişi ---")
    # Aynı ID ile iki farklı saat giriyoruz
    db.add_academic_info("[DERS] sinavi saat 10:00'da.", "exam", "id_101")
    db.add_academic_info("[DERS] sinavi saat 14:00'da.", "exam", "id_101") 
    
    print("\n--- 2. Senaryo: Sorgulama ---")
    soru = "Sınav saat kaçta?"
    cevap = db.query_memory(soru)
    
    print(f"Soru: {soru}")
    print(f"Hafızadan Gelen Bilgiler: {cevap}")
    
    # Burada bir mantık denetimi yapıyoruz
    if len(cevap) > 1:
        print("\n[UYARI]: Hafızada bu konuyla ilgili birden fazla (çelişkili) bilgi bulundu!")
    else:
        print("\n[BİLGİ]: Hafıza tutarlı görünüyor.")

if __name__ == "__main__":
    run_consistency_test()