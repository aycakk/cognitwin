import os
from src.database.chroma_manager import ChromaManager

MASKED_FP = os.path.join("src", "data", "masked", "footprints_masked.txt")

def main():
    assert os.path.exists(MASKED_FP), f"Missing: {MASKED_FP}"
    cm = ChromaManager()

    with open(MASKED_FP, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    for i, ln in enumerate(lines):
        cm.add_academic_info(ln, info_type="footprints", doc_id=f"fp_{i:04d}")

    print(f"INGEST OK: {len(lines)} satır eklendi")

if __name__ == "__main__":
    main()