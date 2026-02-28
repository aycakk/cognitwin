import os
from src.database.chroma_manager import ChromaManager

MASKED_FP = os.path.join("src", "data", "masked", "footprints_masked.txt")

def main():
    assert os.path.exists(MASKED_FP), f"Missing: {MASKED_FP}"

    with open(MASKED_FP, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    cm = ChromaManager()

    # INGEST
    for i, ln in enumerate(lines):
        cm.add_academic_info(ln, info_type="footprints", doc_id=f"ayca_fp_{i:04d}")

    print(f"INGEST OK: {len(lines)} satır eklendi")

    # QUERY
    questions = [
        "COM8090 haftalık raporlar ne zaman teslim ediliyor?",
        "COM8090 vize tarihi ve saati nedir?",
        "COM8090 vize hangi sınıfta?",
        "DM510 K-Means raporunun son teslim zamanı nedir?",
        "OR701 Quiz 1 hangi sınıfta?"
    ]

    for q in questions:
         res = cm.query_memory(q)
         print("\nQ:", q)
         print("Top-3:")
         for r in res[:3]:
            print("-", r)

if __name__ == "__main__":
    main()