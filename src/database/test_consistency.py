from src.database.chroma_manager import ChromaManager


def run_test():
    db = ChromaManager()

    # 1) Normal bir soru (veritabanında tek tip bilgi varsa conflict=false)
    q1 = "Ders programım nedir?"
    conflict1, details1 = db.check_consistency(q1)
    print("\n--- TEST 1 ---")
    print("Soru:", q1)
    print("Çelişki var mı?:", conflict1)
    print("Detay:", details1["matches"], "eşleşme,", details1["unique_statements"], "benzersiz ifade")

    # 2) Yapay çelişki ekleyelim (aynı konudan 2 farklı bilgi)
    # Bu sayede fonksiyonun gerçekten uyardığını görürüz.
    db.add_academic_info(
        "Vize sınavı 20 Mart'ta.",
        {"role": "student", "source": "consistency_test", "category": "exam"},
        "conflict_test_1"
    )
    db.add_academic_info(
        "Vize sınavı 25 Mart'ta.",
        {"role": "student", "source": "consistency_test", "category": "exam"},
        "conflict_test_2"
    )

    q2 = "Vize sınavı ne zaman?"
    conflict2, details2 = db.check_consistency(q2)

    print("\n--- TEST 2 ---")
    print("Soru:", q2)
    print("Çelişki var mı?:", conflict2)
    print("Top docs:")
    for d in details2["top_docs"]:
        print("-", d)

if __name__ == "__main__":
    run_test()