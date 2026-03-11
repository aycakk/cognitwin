import os
import sys

# Proje kök dizinini Python path'ine ekler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.masker import PIIMasker

def main():
    # Proje kökü: .../CogniTwin
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    raw_path = os.path.join(project_root, "src", "data", "footprints.txt")

    out_dir = os.path.join(project_root, "src", "data", "masked")
    out_path = os.path.join(out_dir, "footprints_masked.txt")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"footprints.txt yok: {raw_path}")

    masker = PIIMasker()

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    masked_lines = [masker.mask_data(ln) for ln in lines]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(masked_lines))

    print("✅ MASK OK ->", out_path, " | satır:", len(masked_lines))

if __name__ == "__main__":
    main()