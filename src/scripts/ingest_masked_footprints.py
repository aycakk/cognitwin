import os
from src.database.chroma_manager import ChromaManager

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    masked_fp = os.path.join(project_root, "src", "data", "masked", "footprints_masked.txt")

    if not os.path.exists(masked_fp):
        raise FileNotFoundError(f"Missing: {masked_fp}")

    cm = ChromaManager()

    with open(masked_fp, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    for i, ln in enumerate(lines):
        cm.add_academic_info(
            ln,
            metadata={"info_type": "footprints", "line": i},
            doc_id=f"fp_{i:04d}",
        )

    print(f"✅ INGEST OK: {len(lines)} satır eklendi")

if __name__ == "__main__":
    main()