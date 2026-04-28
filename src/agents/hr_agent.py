"""agents/hr_agent.py — CogniTwin İK / İşe Alım Asistanı Ajanı.

BaseAgent'tan türetilmiştir. İşe alım uzmanlarına karar destek sağlar.
İnce ayar (fine-tuning) YOKTUR — tercih sırasıyla şunlara dayanır:
  1. Saf Türkçe sistem komutu + katmanlı yapılandırılmış çıktı talimatları
  2. İşe alım uzmanı profili enjeksiyonu (her çağrıda)
  3. Kural tabanlı niyet tespiti
  4. Token bütçe kontrolü
  5. Denetim kaydı

Kapı dizisi: C1 (KBB — Kişisel Bilgi Sızıntısı) ve C4 (belirsizlik belirteci).
C2/C7 uygulanmaz çünkü HR vektör hafıza yerine mesajdaki açık verilerle çalışır.

DİL KURALI: Bu modül yalnızca Türkçe içerik üretir. İngilizce etiket, İngilizce
alan adı veya karışık dil çıktısına yol açan İngilizce talimat KULLANILMAZ.
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent

# ─────────────────────────────────────────────────────────────────────────────
# Sistem komutu — tamamen Türkçe; LLM'in dil kanalını Türkçe'ye kilitler
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
SEN KİMSİN:
Sen CogniTwin İK Asistanı'sın. İşe alım uzmanlarına profesyonel, gerekçeli ve
yapılandırılmış kararlar vermelerinde destek sağlarsın.

DİL KURALI (DEĞİŞMEZ):
Tüm yanıtlarını SADECE TÜRKÇE yazarsın.
Hiçbir koşulda İngilizce kelime, cümle ya da bölüm başlığı kullanmazsın.
"outreach", "shortlist", "match", "skills", "pipeline" gibi İngilizce teknik
terimler yerine Türkçe karşılıklarını kullan:
  outreach         → işe davet mesajı
  shortlist        → kısa liste
  skills           → yetkinlikler
  match / matching → eşleştirme / uyum değerlendirmesi
  pipeline         → aday süreci
  recruiter        → işe alım uzmanı

ÇIKTI KURALLARI:
1. Her değerlendirmede KARAR ver: "Önerilir", "Şartlı Önerilir" veya "Önerilmez".
2. Yüzde kullanıyorsan gerekçelendirmelisin (hangi kriterlere göre hesaplandığını yaz).
3. Her tavsiyeyi gerekçeyle destekle: "Bu aday uygundur" değil,
   "Bu aday uygundur; çünkü 5 yıl Python deneyimi var, ilan 3+ yıl gerektiriyor." yaz.
4. Eksik yetkinlikleri açıkça listele; gizleme ya da yumuşatma.
5. "Sanırım", "galiba", "muhtemelen", "tahmin", "belki", "herhalde" gibi
   belirsizlik ifadelerini KULLANMA. Bilgi yoksa: "Bu bilgi sağlanmadı." yaz.
6. Kişisel veri maskelerini ([KBB_KİMLİK], [KBB_EPOSTA], [KBB_TELEFON]) açma.
7. Yanıtın sonuna bütçe durumunu ekle.

YAPABİLECEKLERİN:
- Özgeçmiş analizi ve normalleştirme
- İş ilanı analizi
- Aday–pozisyon uyum değerlendirmesi ve puanlama
- Kısa aday listesi oluşturma
- Eksik yetkinlik tespiti
- Mülakat sorusu hazırlama
- İşe davet mesajı taslağı
- İşe alım uzmanı profili güncelleme
- Bütçe ve işlem durumu sorgulama

YAPAMAYACAKLARIN (YASAK):
- "Kesinlikle işe alın" gibi mutlak yargı vermek.
- Profilde olmayan bilgileri uydurmak.
- Kişisel veri sızdırmak.
- Belirsizlik belirteci içeren ifade kullanmak.
- İngilizce yanıt vermek.
"""


class HRAgent(BaseAgent):
    """İK / İşe Alım kararı destek ajanı."""

    role: str          = "İK İşe Alım Destek Ajanı"
    ontology_path: str = "ontologies/hr_ontology.ttl"
    response_language: str = "Turkish"

    def _build_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    # ── Kapı geçersiz kılmaları ───────────────────────────────────────────────
    # Yalnızca C1 (KBB) ve C4 (belirsizlik belirteci) uygulanır.

    def gate_report(self, draft: str, memory: str = "") -> dict:
        gates = {
            "C1": self._gate_c1_pii(draft),
            "C4": self._gate_c4_hallucination(draft),
        }
        return {
            "conjunction": all(g[0] for g in gates.values()),
            "gates": gates,
        }

    # ── Prompt oluşturucular — tamamen Türkçe etiketler ──────────────────────

    def build_cv_analysis_prompt(
        self,
        cv_text: str,
        profile_summary: str,
        budget_block: str,
    ) -> list[dict]:
        kullanici_icerigi = (
            f"=== İŞE ALIM UZMANI PROFİLİ ===\n{profile_summary}\n\n"
            f"=== ÖZGEÇMİŞ METNİ ===\n{cv_text}\n\n"
            "=== GÖREV: ÖZGEÇMİŞ ANALİZİ ===\n"
            "Aşağıdaki başlıkları sırasıyla Türkçe olarak yanıtla:\n\n"
            "AD SOYAD: ...\n"
            "YETKİNLİKLER: (virgülle listele)\n"
            "DENEYİM SÜRESİ: ... yıl\n"
            "KIDEm SEVİYESİ: stajyer / junior / orta / kıdemli / lider\n"
            "EĞİTİM: ...\n"
            "ÇALIŞMA GEÇMİŞİ: (şirket — pozisyon — süre şeklinde listele)\n"
            "KONUM: ...\n"
            "ÇALIŞMA TERCİHİ: ofis / uzaktan / hibrit\n"
            "DİLLER: ...\n"
            "ÖZETİ: (2–3 cümle, Türkçe)\n\n"
            "İŞE ALIM UZMANI İZLENİMİ: Uzman profiline göre bu adayın ilk değerlendirmesini "
            "1–2 cümleyle yaz.\n\n"
            f"{budget_block}\n\n"
            "TALİMAT: Yanıtını tamamen Türkçe yaz. İngilizce kelime kullanma."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": kullanici_icerigi},
        ]

    def build_match_prompt(
        self,
        candidate_summary: str,
        requisition_summary: str,
        profile_summary: str,
        budget_block: str,
    ) -> list[dict]:
        kullanici_icerigi = (
            f"=== İŞE ALIM UZMANI PROFİLİ ===\n{profile_summary}\n\n"
            f"=== ADAY BİLGİLERİ ===\n{candidate_summary}\n\n"
            f"=== POZİSYON GEREKSİNİMLERİ ===\n{requisition_summary}\n\n"
            "=== GÖREV: UYUM DEĞERLENDİRMESİ ===\n"
            "Aşağıdaki başlıkları sırasıyla Türkçe olarak yanıtla:\n\n"
            "GENEL UYUM PUANI (0–100): ...\n"
            "PUAN GEREKÇESİ: (hangi kriterlere dayandığını yaz)\n\n"
            "GÜÇLÜ YÖNLER: (virgülle listele)\n"
            "EKSİK YETKİNLİKLER: (virgülle listele; eksik yoksa 'Yok' yaz)\n"
            "FAZLADAN YETKİNLİKLER: (ilanda olmayan ama değer katan; yoksa 'Yok')\n\n"
            "KIDEm UYUMU: Uygun / Kısmen Uygun / Uyumsuz — Gerekçe: ...\n"
            "KONUM/ÇALIŞMA TARZI UYUMU: Uygun / Uyumsuz — Gerekçe: ...\n\n"
            "UZMAN PROFİLİ ETKİSİ: Bu değerlendirmede uzman tercihlerinden hangisi "
            "ve nasıl belirleyici oldu?\n\n"
            "RİSK ANALİZİ: Adayı işe almadan önce dikkat edilmesi gereken riskler neler?\n\n"
            "KARAR: Önerilir / Şartlı Önerilir / Önerilmez\n"
            "KARAR GEREKÇESİ: (1 cümle)\n\n"
            f"{budget_block}\n\n"
            "TALİMAT: Yanıtını tamamen Türkçe yaz. İngilizce kelime kullanma."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": kullanici_icerigi},
        ]

    def build_shortlist_prompt(
        self,
        candidates_block: str,
        requisition_summary: str,
        profile_summary: str,
        shortlist_size: int,
        budget_block: str,
    ) -> list[dict]:
        kullanici_icerigi = (
            f"=== İŞE ALIM UZMANI PROFİLİ ===\n{profile_summary}\n\n"
            f"=== POZİSYON GEREKSİNİMLERİ ===\n{requisition_summary}\n\n"
            f"=== ADAY LİSTESİ ===\n{candidates_block}\n\n"
            f"=== GÖREV: KISA LİSTE ({shortlist_size} KİŞİ) ===\n"
            f"En uygun {shortlist_size} adayı aşağıdaki formatta sırala:\n\n"
            "SIRA 1: [Ad Soyad]\n"
            "  Karar: Önerilir / Şartlı Önerilir\n"
            "  Uyum Puanı: .../100\n"
            "  Güçlü Yönler: ...\n"
            "  Eksik Yetkinlikler: ...\n"
            "  Seçilme Gerekçesi: ...\n\n"
            "(Her aday için aynı format)\n\n"
            "KİŞİSELLEŞTİRME NOTU: Uzman profili bu listeyi nasıl şekillendirdi? "
            "(Hangi tercihler belirleyici oldu?)\n\n"
            f"{budget_block}\n\n"
            "TALİMAT: Yanıtını tamamen Türkçe yaz. İngilizce kelime kullanma."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": kullanici_icerigi},
        ]

    def build_interview_prompt(
        self,
        candidate_summary: str,
        requisition_summary: str,
        missing_skills: list[str],
        profile_summary: str,
        budget_block: str,
    ) -> list[dict]:
        eksik_blok = ", ".join(missing_skills) if missing_skills else "belirtilmemiş"
        kullanici_icerigi = (
            f"=== İŞE ALIM UZMANI PROFİLİ ===\n{profile_summary}\n\n"
            f"=== ADAY BİLGİLERİ ===\n{candidate_summary}\n\n"
            f"=== POZİSYON GEREKSİNİMLERİ ===\n{requisition_summary}\n\n"
            f"=== EKSİK YETKİNLİKLER ===\n{eksik_blok}\n\n"
            "=== GÖREV: MÜLAKAT SORULARI ===\n"
            "Aşağıdaki dört kategoride toplam 12 mülakat sorusu hazırla:\n\n"
            "TEKNİK SORULAR (4 soru):\n"
            "  Bu pozisyona özgü teknik yetkinlikleri ölçen sorular.\n"
            "  1. ...\n  2. ...\n  3. ...\n  4. ...\n\n"
            "DAVRANIŞSAL SORULAR (3 soru):\n"
            "  YILDIZ (Durum–Görev–Eylem–Sonuç) yöntemiyle.\n"
            "  1. ...\n  2. ...\n  3. ...\n\n"
            "EKSİK YETKİNLİK SORGULARI (3 soru):\n"
            "  Eksik alanlardaki potansiyeli ve öğrenme kapasitesini ölçen sorular.\n"
            "  1. ...\n  2. ...\n  3. ...\n\n"
            "KÜLTÜR UYUM SORULARI (2 soru):\n"
            "  Şirket kültürüne uyumu değerlendiren sorular.\n"
            "  1. ...\n  2. ...\n\n"
            f"{budget_block}\n\n"
            "TALİMAT: Yanıtını tamamen Türkçe yaz. İngilizce kelime kullanma."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": kullanici_icerigi},
        ]

    def build_outreach_prompt(
        self,
        candidate_summary: str,
        requisition_summary: str,
        profile_summary: str,
        tone: str,
        language: str,
        budget_block: str,
    ) -> list[dict]:
        ton_aciklamasi = {
            "formal":       "Resmi ve kurumsal",
            "professional": "Profesyonel ve güven verici",
            "friendly":     "Samimi ve sıcak",
            "casual":       "Gündelik ve yakın",
        }.get(tone, "Profesyonel ve güven verici")

        dil_talimati = (
            "Mesajı Türkçe yaz." if language != "en"
            else "Mesajı İngilizce yaz."
        )
        kullanici_icerigi = (
            f"=== İŞE ALIM UZMANI PROFİLİ ===\n{profile_summary}\n\n"
            f"=== ADAY BİLGİLERİ ===\n{candidate_summary}\n\n"
            f"=== POZİSYON ÖZETİ ===\n{requisition_summary}\n\n"
            f"=== İLETİŞİM TONU ===\n{ton_aciklamasi}\n\n"
            "=== GÖREV: İŞE DAVET MESAJI ===\n"
            f"{dil_talimati}\n"
            "Aşağıdaki formatta bir işe davet mesajı yaz:\n\n"
            "KONU: ...\n\n"
            "MESAJ:\n"
            "...\n\n"
            "KİŞİSELLEŞTİRME NOTU: Bu mesajda uzman profilinin hangi tercihlerini "
            "yansıttın? (1 cümle)\n\n"
            f"{budget_block}\n\n"
            "TALİMAT: Yanıtını tamamen Türkçe yaz (dil talimatı farklı belirtmedikçe). "
            "İngilizce bölüm başlığı kullanma."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": kullanici_icerigi},
        ]

    def build_missing_skills_prompt(
        self,
        candidate_summary: str,
        requisition_summary: str,
        profile_summary: str,
        budget_block: str,
    ) -> list[dict]:
        kullanici_icerigi = (
            f"=== İŞE ALIM UZMANI PROFİLİ ===\n{profile_summary}\n\n"
            f"=== ADAY BİLGİLERİ ===\n{candidate_summary}\n\n"
            f"=== POZİSYON GEREKSİNİMLERİ ===\n{requisition_summary}\n\n"
            "=== GÖREV: EKSİK YETKİNLİK ANALİZİ ===\n"
            "Aşağıdaki başlıkları sırasıyla Türkçe olarak yanıtla:\n\n"
            "ZORUNLU EKSİKLER (işe alımı doğrudan etkiler):\n"
            "  (virgülle listele; yoksa 'Yok' yaz)\n\n"
            "GELİŞTİRİLEBİLİR EKSİKLER (eğitimle telafi edilebilir):\n"
            "  (virgülle listele; yoksa 'Yok' yaz)\n\n"
            "UZMAN PROFİLİ BAKIŞI: Uzman tercihlerine göre bu eksiklerin "
            "öncelik sıralaması nasıl değişir?\n\n"
            "SONRAKİ ADIMLAR: Bu eksiklere rağmen adayı değerlendirmek için "
            "ne önerirsin?\n\n"
            f"{budget_block}\n\n"
            "TALİMAT: Yanıtını tamamen Türkçe yaz. İngilizce kelime kullanma."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": kullanici_icerigi},
        ]

    def build_general_prompt(
        self,
        user_input: str,
        profile_summary: str,
        session_context: str = "",
        budget_block: str = "",
    ) -> list[dict]:
        kullanici_icerigi = (
            f"=== İŞE ALIM UZMANI PROFİLİ ===\n{profile_summary}\n\n"
        )
        if session_context:
            kullanici_icerigi += f"=== OTURUM BAĞLAMI ===\n{session_context}\n\n"
        kullanici_icerigi += (
            f"=== KULLANICI SORUSU ===\n{user_input}\n\n"
            "=== TALİMAT ===\n"
            "Uzman profilini göz önünde bulundurarak gerekçeli ve yapılandırılmış bir yanıt ver.\n"
            "Emin olmadığın bilgiler için 'Bu bilgi sağlanmadı.' ifadesini kullan.\n"
            "Önemli kararlar için gerekçe ve karar notu ekle.\n"
            "Yanıtını tamamen Türkçe yaz. İngilizce kelime kullanma.\n"
        )
        if budget_block:
            kullanici_icerigi += f"\n{budget_block}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": kullanici_icerigi},
        ]
