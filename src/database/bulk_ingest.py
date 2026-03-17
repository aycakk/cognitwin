import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT SETUP
# Import hatalarını önlemek için proje kök dizinini Python yoluna ekliyoruz.
# ─────────────────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/database
project_root = os.path.dirname(os.path.dirname(current_dir)) # CogniTwin

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Merkezi ChromaManager'ı içe aktarıyoruz
from src.database.chroma_manager import db_manager

def bulk_ingest_masked_data(file_path: str):
    """
    Belirtilen metin dosyasındaki her satırı akademik hafızaya (ChromaDB) mühürler.
    """
    
    # Dosya yolunu mutlak absolute path çeviriyoruz
    if not os.path.isabs(file_path):
        file_path = os.path.normpath(os.path.join(project_root, file_path))

    if not os.path.exists(file_path):
        print(f"❌ [HATA] Kaynak dosya bulunamadı: {file_path}")
        return

    print(f"🚀 [INGEST] Veriler mühürleniyor: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            count = 0
            for i, line in enumerate(lines):
                clean_line = line.strip()
                if clean_line:
                    # Metadata: Verinin kaynağını ve rolünü belirtir (Filtreleme için kritik)
                    metadata = {
                        "role": "student", 
                        "source": os.path.basename(file_path),
                        "type": "academic_footprint"
                    }
                    # Benzersiz ID oluşturma (Dosya ismi + satır numarası)
                    doc_id = f"fp_{os.path.basename(file_path)}_{i}"
                    
                    # db_manager üzerinden veritabanına ekle
                    db_manager.add_academic_info(clean_line, metadata, doc_id)
                    count += 1
            
            print(f"✅ [BAŞARILI] {count} adet akademik kayıt sisteme işlendi.")
            
    except Exception as e:
        print(f"⚠️ [KRİTİK HATA] İşlem sırasında hata: {e}")

if __name__ == "__main__":
    # Veri kaynağı yolu 
    target_file = "src/data/masked/footprints_masked.txt" 
    
    bulk_ingest_masked_data(target_file)

    # Final Hafıza Raporu
    print("\n" + "═" * 40)
    final_count = db_manager.collection.count()
    print(f"📊 SİSTEM HAFIZASI GÜNCEL DURUM: {final_count} Kayıt")
    print("═" * 40)