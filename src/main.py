from utils.masker import PIIMasker
from agents.student_agent import StudentAgent

def test_infrastructure():
    masker = PIIMasker()
    # Buradaki ismi 'Alex Smith' yaparak masker içindeki mantığı doğruluyoruz
    raw_data = "Merhaba, ben Alex Smith. Öğrenci numaram 2024105060 ve mailim alex@university.edu"
    
    masked_data = masker.mask_data(raw_data)
    
    print("--- Test Sonucu ---")
    print(f"Girdi: {raw_data}")
    print(f"Çıktı: {masked_data}")

if __name__ == "__main__":
    test_infrastructure()