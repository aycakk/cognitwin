import sys
import os
import ollama

from src.pipeline.shared import DEFAULT_MODEL

# 1. ADIM: Klasör yollarını çok daha sağlam bir yöntemle tanıtıyoruz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

print(f"Proje ana dizini: {BASE_DIR}") # Nerede olduğumuzu görelim

# 2. ADIM: İlayda'nın modüllerini çağırıyoruz
try:
    from utils.masker import PIIMasker
    from agents.student_agent import StudentAgent
    print("✅ İlayda'nın güvenlik ve ajan modülleri yüklendi.")
except ImportError as e:
    print(f"❌ Modüller yüklenemedi! Hata: {e}")
    print("Lütfen src/utils/masker.py ve src/agents/student_agent.py dosyalarının varlığını kontrol et.")
    sys.exit()

# 3. ADIM: Senin verini okuyoruz (Üye 1 Görevi)
data_path = os.path.join(BASE_DIR, "src", "data", "footprints.txt")

try:
    with open(data_path, "r", encoding="utf-8") as f:
        ham_veri = f.read()
    print("✅ 'footprints.txt' başarıyla okundu.")
except FileNotFoundError:
    print(f"❌ HATA: {data_path} bulunamadı!")
    sys.exit()

# 4. ADIM: Maskeleme ve LLM Testi
masker = PIIMasker()
agent = StudentAgent()

temiz_veri = masker.mask_data(ham_veri)
print("\n--- MASKELEŞMİŞ VERİ ---")
print(temiz_veri)

print("\n--- YAPAY ZEKA CEVAP VERİYOR... ---")
response = ollama.chat(model=DEFAULT_MODEL, messages=[
    {"role": "system", "content": agent.get_system_prompt()},
    {"role": "user", "content": f"Verileri incele ve grup üyelerini say: {temiz_veri}"}
])

print("\n🤖 SONUÇ:")
print(response['message']['content'])