from __future__ import annotations
import sys
import os
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  BÖLÜM 0 ▸ ORTAM KURULUMU (ENVIRONMENT SETUP)
#  Proje kök dizinini ve veritabanı modüllerini Python yoluna (sys.path) ekler.
# ─────────────────────────────────────────────────────────────────────────────

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.agents.student_agent import StudentAgent
from src.utils.masker import PIIMasker
from src.database.vector_store import search_memory
from ollama import chat

# Global Nesneler: Bellekte bir kez oluşturulur, API çağrılarında hızı artırır.
_agent = StudentAgent()
_masker = PIIMasker()

@dataclass
class PipelineResult:
    """CogniTwin RAG hattı için standartlaştırılmış çıktı yapısı."""
    answer: str
    verification_status: str = "UNKNOWN"
    context_used: str = ""

# ─────────────────────────────────────────────────────────────────────────────
#  BÖLÜM 1 ▸ ANA PİPELİNE MANTIĞI
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(user_text: str) -> PipelineResult:
    """
    4 Aşamalı RAG Akışını yönetir: Veri Çekme -> Maskeleme -> LLM Sentezi -> Doğrulama.
    
    Args:
        user_text (str): Öğrenciden gelen ham sorgu metni.
        
    Returns:
        PipelineResult: Sentezlenmiş yanıtı ve meta verileri içeren nesne.
    """
    
    # --- AŞAMA 1: SEMANTİK VERİ ÇEKME (RETRIEVAL) ---
    # ChromaDB üzerinde vektör araması yapılır. İsimler bu aşamada açık tutulur 
    # çünkü özel isimler (Yusuf Hoca vb.) semantik benzerlik için kritiktir.
    memory_content = ""
    try:
        # k=25 derinliği, parçalı footprint verilerinden ipuçlarını toplamak için optimize edildi.
        res = search_memory(user_text, k=25)
        
        if res:
            extracted = [item['text'] if isinstance(item, dict) else str(item) for item in res]
            memory_content = "\n".join(extracted)
    except Exception as e:
        return PipelineResult(
            answer=f"Veritabanı Erişim Hatası: {e}", 
            verification_status="DB_ERROR"
        )

    # --- AŞAMA 2: GÜVENLİK VE MASKELEME ---
    # Kullanıcı sorgusundaki hassas PII verileri (E-posta, ID, Telefon) maskelenir.
    # Bu işlem, verinin dış LLM sağlayıcısına (Llama) gitmeden önceki güvenlik duvarıdır.
    masked_query = _masker.mask_data(user_text)

    # Eğer hafızada ilgili hiçbir kayıt bulunamazsa BlindSpot (Kör Nokta) tetiklenir.
    if not memory_content.strip():
        return PipelineResult(
            answer="Hafızamda bu konuyla ilgili yeterli kayıt bulamadım.", 
            verification_status="C7_TRIGGERED"
        )

    # --- AŞAMA 3: LLM SENTEZLEME (Llama 3.2) ---
    # Çekilen akademik kayıtlar ve maskelenmiş sorgu birleştirilir.
    # Ajan, kayıtlar arasındaki kopuk bilgileri muhakeme yoluyla bağlar.
    evidence_bundle = f"ACADEMIC_RECORDS:\n{memory_content}"
    messages = _agent.format_message_with_memory(masked_query, evidence_bundle)
    
    try:
        # Ollama üzerinden yerel Llama 3.2 modeline sentez çağrısı yapılır.
        resp = chat(model="llama3.2", messages=messages)
        answer = resp.get("message", {}).get("content", "").strip()
        
        return PipelineResult(
            answer=answer, 
            verification_status="SUCCESS", 
            context_used=memory_content
        )
    except Exception as e:
        return PipelineResult(
            answer=f"Sentez Motoru Hatası: {e}", 
            verification_status="LLM_ERROR"
        )

# ─────────────────────────────────────────────────────────────────────────────
#  BÖLÜM 2 ▸ DIŞ ARAYÜZ (API)
# ─────────────────────────────────────────────────────────────────────────────

def process_user_message(user_text: str) -> dict:
    """
    Dış servisler (LibreChat, Streamlit vb.) için ana giriş noktasıdır.
    Pipeline sonucunu seri hale getirilebilir (JSON uyumlu) bir sözlüğe dönüştürür.
    """
    res = _run_pipeline(user_text)
    return {
        "answer": res.answer,
        "verification_status": res.verification_status,
        "context_used": res.context_used
    }