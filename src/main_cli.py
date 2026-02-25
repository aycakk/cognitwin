import os
from ollama import chat
from utils.masker import PIIMasker

def load_footprints():
    base = os.path.dirname(__file__)  # .../src
    raw_path = os.path.join(base, "data", "footprints.txt")
    masked_path = os.path.join(base, "data", "masked", "footprints_masked.txt")

    # Önce masked varsa onu kullan, yoksa raw'ı anlık maskele
    if os.path.exists(masked_path):
        with open(masked_path, "r", encoding="utf-8") as f:
            return f.read()

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = f.read()

    masker = PIIMasker()
    return masker.mask_data(raw)

def ask():
    memory = load_footprints()

    system_prompt = (
        "Sen bir 'Student' ajanısın. Akademik tonda kısa ve net cevap ver. "
        "Sadece verilen hafıza (MEMORY) içinden kesin bilgi varsa kullan. "
        "MEMORY'de yoksa 'Bunu hafızamda bulamadım' de ve tahmin yürütme."
    )

    print("Cognitwin CLI hazır. Çıkmak için: /exit\n")

    while True:
        q = input("Soru > ").strip()
        if q.lower() in ["/exit", "exit", "quit"]:
            break

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"MEMORY:\n{memory}\n\nSORU: {q}"}
        ]

        resp = chat(model="llama3.2", messages=messages)
        print("\nYanıt:", resp["message"]["content"], "\n")

if __name__ == "__main__":
    ask()