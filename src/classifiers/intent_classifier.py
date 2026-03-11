class IntentClassifier:
    # Kullanıcı sorusunun DATA mı COMMENT mi olduğunu belirler

    DATA_KEYWORDS = [
        "ne zaman",
        "nerede",
        "kaç",
        "kaçta",
        "kim",
        "hangi",
        "not",
        "notum",
        "puan",
        "puanım",
        "tarih",
        "deadline",
        "exam date",
        "grade",
        "status",
        "durum",
        "sonuç",
        "teslim",
        "saat"
    ]

    def classify(self, question: str) -> str:
        # Soruyu küçük harfe çevirerek anahtar kelime kontrolü yapar
        normalized_question = question.lower()

        for keyword in self.DATA_KEYWORDS:
            if keyword in normalized_question:
                return "DATA"

        return "COMMENT"
if __name__ == "__main__":
    classifier = IntentClassifier()

    print(classifier.classify("Vize ne zaman"))
    print(classifier.classify("Bu derse nasıl çalışmalıyım"))