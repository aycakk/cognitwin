import ollama
import os

# 1. Veriyi Oku (Senin görevin: Veri Kaynağı)
path = os.path.join("src", "data", "footprints.txt")

with open(path, "r", encoding="utf-8") as f:
    veriler = f.read()

# 2. Modeli Test Et
response = ollama.chat(model='llama3.2', messages=[
  {
    'role': 'system',
    'content': 'Sen COGNITWIN Student ajanısın. SADECE sana verilen notlardaki bilgileri kullan. Eğer bilgi notlarda yoksa "Bilmiyorum" de.',
  },
  {
    'role': 'user',
    'content': f"Şu notları incele:\n{veriler}\n\nSoru: Notlarımda adı geçen grup arkadaşlarım kimlerdir?",
  },
])

print("\n--- AJAN CEVABI ---")
print(response['message']['content'])