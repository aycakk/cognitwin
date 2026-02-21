import os
from src.utils.masker import PIIMasker

def main():
    # Proje kökü (CogniTwin/CogniTwin)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # .../CogniTwin
    raw_path = os.path.join(project_root, "src", "data", "footprints.txt")

    out_dir = os.path.join(project_root, "src", "data", "masked")
    out_path = os.path.join(out_dir, "footprints_masked.txt")
    os.makedirs(out_dir, exist_ok=True)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = f.read()

    masker = PIIMasker()
    masked = masker.mask_data(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(masked)

    print("OK ->", out_path)

if __name__ == "__main__":
    main()