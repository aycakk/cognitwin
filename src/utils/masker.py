import re

class PIIMasker:
    def __init__(self):
        # Genel akademik standartlara uygun Regex kalıpları
        self.patterns = {
            # İsim ve Soyisim (Büyük harfle başlayan iki kelime)
            "USER_NAME": r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b',
            
            # Akademik E-posta (edu, edu.tr, ac.uk, vb. tüm akademik uzantılar)
            "ACADEMIC_EMAIL": r'[\w\.-]+@[\w\.-]+\.(edu|edu\.tr|ac\.\w+|org\.tr)',
            
            # Öğrenci Numarası (Dünya genelinde yaygın 8-12 hane arası sayı dizileri)
            "STUDENT_ID": r'\b\d{8,12}\b',
            
            # Uluslararası Telefon Formatı (+ veya 0 ile başlayan)
            "PHONE": r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            
            # TC No veya benzeri 11 haneli hassas kimlik numaraları
            "ID_NUMBER": r'\b\d{11}\b',

            # Student attendance 
            "ATTENDANCE": r'\b(Present|Absent|Katıldı|Katılmadı|%\d{1,3})\b',

            # Maaş veya bütçe gibi rakamsal veriler 
            "FINANCIAL": r'\b\d{1,3}(\.\d{3})*(\,\d{2})?\s?(TL|USD|EUR|TRY)\b'
        }

    def mask_data(self, text):
        """
        Metindeki hassas verileri bulur ve içeriği gizleyerek etiketler.
        Örn: 'test@mit.edu' -> '[ACADEMIC_EMAIL_MASKED]'
        """
        if not text:
            return text
            
        masked_text = text
        for label, pattern in self.patterns.items():
            masked_text = re.sub(pattern, f"[{label}_MASKED]", masked_text)
            
        return masked_text

# Test etmek istersen burayı çalıştır
if __name__ == "__main__":
    masker = PIIMasker()
    sample = "Merhaba, ben Alex Smith. Öğrenci numaram 2024105060 ve mailim alex@university.edu"
    print(f"Maskelenmiş Metin: {masker.mask_data(sample)}")