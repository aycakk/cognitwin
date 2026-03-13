import re

class PIIMasker:
    def __init__(self):
        # Genel akademik standartlara uygun Regex kalıpları
        self.patterns = {
            
            # email 11 rol için 
            "EMAIL": r'[\w\.-]+@[\w\.-]+\.\w+',
            
            # Uluslararası Telefon Formatı (+ veya 0 ile başlayan)
            "PHONE": r'\b(\+90\s?)?(0?5\d{2})[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b',
            
            # KİMLİK NUMARALARI (Öğrenci No, Personel No veya TC No - 8-11 hane arası)
            "ID_NUMBER": r'\b\d{8,11}\b',

            # Student attendance 
            "ATTENDANCE": r'\b(Present|Absent|Katıldı|Katılmadı|%\d{1,3})\b',

            # Maaş veya bütçe gibi rakamsal veriler 11 rol için
            "FINANCIAL": r'\b\d{1,3}(\.\d{3})*(\,\d{2})?\s?(TL|TL\.|₺|USD|EUR|TRY)\b'
        }

    def mask_data(self, text):
        """
        Metindeki hassas verileri bulur ve içeriği gizleyerek etiketler.
        Örn: 'test@mit.edu' -> '[EMAIL_MASKED]'
        """
        if not text:
            return text
            
        masked_text = text
        for label, pattern in self.patterns.items():
            masked_text = re.sub(pattern, f"[{label}_MASKED]", masked_text)
            
        return masked_text