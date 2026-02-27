import os
from chroma_manager import ChromaManager

def bulk_ingest_masked_data(file_path):
    db = ChromaManager()
    
    if not os.path.exists(file_path):
        print(f"Hata: {file_path} dosyası bulunamadı!")
        return

    print(f"--- {file_path} verileri yükleniyor ---")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if clean_line:
                # Metadata etiketleme görevini burada yapıyoruz
                metadata = {"role": "student", "source": "footprints_masked"}
                # Her satır için benzersiz bir ID oluşturuyoruz
                doc_id = f"student_footprint_{i}"
                
                db.add_academic_info(clean_line, metadata, doc_id)
                print(f"Yüklendi: {doc_id}")

    print("\n✅ Tüm veriler başarıyla yüklendi ve etiketlendi!")

if __name__ == "__main__":
    # Dosya yolunu senin klasör yapına göre ayarladım
    target_file = "src/data/masked/footprints_masked.txt"
    bulk_ingest_masked_data(target_file)