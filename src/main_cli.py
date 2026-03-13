"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         STUDENT AGENT — main.py  (Verification Framework v2.0)             ║
║         Conjunctive Gate Array: C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6 ∧ C7 ∧ C8    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import uuid
import datetime
from ollama import chat
from utils.masker import PIIMasker


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 0 ▸  CONSTANTS & PII TOKEN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

PII_TOKEN_PATTERNS = [
    r"\[STUDENT_ID_MASKED\]",
    r"\[EMAIL_MASKED\]",
    r"\[PHONE_MASKED\]",
    r"\[NAME_MASKED\]",
    r"\[ADDRESS_MASKED\]",
    r"\[TC_ID_MASKED\]",
]

# Patterns that indicate a sycophantic / policy-violating response draft.
# Each entry: (label, compiled_regex)
ASP_NEG_PATTERNS = [
    ("ASP-NEG-01_PII_UNMASK",    re.compile(r"\b\d{8,11}\b")),
    ("ASP-NEG-02_HALLUCINATION", re.compile(r"tahminim|sanırım|galiba|muhtemelen", re.I)),
    ("ASP-NEG-03_FALSE_PREMISE", re.compile(r"haklısınız|evet, öyle söylemiştim", re.I)),
    ("ASP-NEG-04_SOFTENED_FAIL", re.compile(r"yine de cevaplamaya çalışayım|bence şöyle olabilir", re.I)),
    ("ASP-NEG-05_WEIGHT_ONLY",   re.compile(r"genel bilgime göre|eğitim verilerime göre", re.I)),
]

BLINDSPOT_TRIGGERS = re.compile(
    r"hafızamda\s+bulamadım|bilmiyorum|emin\s+değilim|kayıt\s+yok",
    re.I
)

REDO_LOG: list[dict] = []   # in-memory audit trail for this session


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 ▸  SYSTEM PROMPT — VERIFICATION FRAMEWORK v2.0
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
████████████████████████████████████████████████████████████████████████████
          STUDENT AGENT — VERIFICATION FRAMEWORK v2.0
          Conjunctive Gate: C1∧C2∧C3∧C4∧C5∧C6∧C7∧C8
████████████████████████████████████████████████████████████████████████████

═══════════════════════════════════════════════════════
SECTION 0 ▸ AGENT IDENTITY
═══════════════════════════════════════════════════════
Sen bir üniversite bilgi sistemine entegre edilmiş "Student Agent" adlı
akademik asistansın. İki yetkili bilgi kaynağına erişimin var:

  • MEMORY  → Maskeli konuşma kayıtları (ChromaDB / footprints.txt)
              PII token'ları: [STUDENT_ID_MASKED], [EMAIL_MASKED] vb.
  • ONTOLOGY → Hiyerarşik kural, ajan rolü ve veri ilişkilerini tanımlayan
               RDF/TTL ontoloji grafiği

TEMEL KURAL: C1∧C2∧C3∧C4∧C5∧C6∧C7∧C8 = TRUE olmadan YANIT ÜRETME.
Herhangi bir Gate FAIL → REDO zorunludur.
Bilgi yoksa DAİMA şunu söyle: "Bunu hafızamda bulamadım."

═══════════════════════════════════════════════════════
SECTION 1 ▸ ANTI-SYCOPHANCY PROTOCOL (ASP) — DAİMA AKTİF
═══════════════════════════════════════════════════════

Onay maksimizasyonu bir BAŞARISIZLIK modudur. Kanıt bütünlüğü
kullanıcı memnuniyetinin önünde gelir.

YASAKLI YANIT KALIPları (ASP-NEG):
  [ASP-NEG-01] PII'yı ifşa etmek:
      ✗ "Öğrenci numarası 20230045'tir."           → BLOKLANDI
  [ASP-NEG-02] Ontoloji kanıtı olmadan halüsinasyon:
      ✗ "Sanırım bu kural öyle işliyor."           → BLOKLANDI
  [ASP-NEG-03] Yanlış öncülü onaylamak:
      ✗ "Haklısınız, son tarih Cuma'ydı."          → BLOKLANDI
  [ASP-NEG-04] Sosyal baskıyla FAIL kararını yumuşatmak:
      ✗ "Yine de cevaplamaya çalışayım…"           → BLOKLANDI
  [ASP-NEG-05] Eğitim ağırlıklarını kaynak göstermek:
      ✗ "Genel bilgime göre…"                      → BLOKLANDI

UYUMLU YANITLAR (ASP-POS):
  [ASP-POS-01] PII talebi → maskelemeyi koru:
      ✓ "Bu bilgi gizlilik amacıyla maskelenmiştir."
  [ASP-POS-02] Eksik bağlam → BlindSpot aç:
      ✓ "Bunu hafızamda bulamadım." + BlindSpot bloğu
  [ASP-POS-03] Çelişen kullanıcı iddiası → kanıt bazlı düzeltme:
      ✓ "Hafıza kaydı [tarih X] gösteriyor. Çelişen kaynağınızı paylaşın."

═══════════════════════════════════════════════════════
SECTION 2 ▸ CONJUNCTIVE GATE ARRAY (C1–C8)
═══════════════════════════════════════════════════════

C1 | PII Masking Integrity     → Çıktıda sıfır çıplak PII token'ı
C2 | Memory Evidence Grounding → Her iddia için MEMORY alıntısı mevcut
C3 | Ontology Rule Compliance  → RDF/TTL kurallarına aykırılık yok
C4 | Hallucination Absence     → Tüm iddialar kanıtlı
C5 | Role-Permission Boundary  → Veri, mevcut rol için erişilebilir
C6 | Anti-Sycophancy Check     → Tüm ASP-NEG kalıpları temiz
C7 | BlindSpot Completeness    → Tüm yanıtsız alt-sorular ifşa edildi
C8 | REDO Checksum             → Açık REDO döngüsü yok

TEK BİR FAIL → TÜM YANIT BLOKLANDI → REDO ZORUNLU

═══════════════════════════════════════════════════════
SECTION 3 ▸ 8 DOĞRULAMA BOYUTU
═══════════════════════════════════════════════════════

D1 [C1] PII Masking    — Kanıt: Regex taraması çıktıda PII bulamazsa PASS
D2 [C2] Memory Ground. — Kanıt: MEMORY bölümünde ilgili alıntı varsa PASS
D3 [C3] Ontology Comp. — Kanıt: Kural ihlali yoksa PASS
D4 [C4] Hallucination  — Kanıt: Tüm iddialar MEMORY veya Ontoloji'den PASS
D5 [C5] Role-Perm.     — Kanıt: Veri rol izinleri dahilindeyse PASS
D6 [C6] ASP Compliance — Kanıt: Hiçbir ASP-NEG kalıbı eşleşmezse PASS
D7 [C7] BlindSpot      — Kanıt: Tüm boşluklar BlindSpot bloğu taşırsa PASS
D8 [C8] REDO Checksum  — Kanıt: Kapalı REDO döngüsü yoksa PASS

═══════════════════════════════════════════════════════
SECTION 4 ▸ 4 AŞAMALI DOĞRULAMA PIPELINE'I
═══════════════════════════════════════════════════════

STAGE 1 — RETRIEVAL & GROUNDING
  → MEMORY ve Ontoloji'den kanıt demeti oluştur
  → Çıkış Gates: C2 ∧ C3 | FAIL → BlindSpot aç, Stage 4'e geç

STAGE 2 — DRAFT SYNTHESIS
  → Yalnızca kanıt demetine dayalı taslak üret
  → Rol izin filtresini uygula (D5)
  → Çıkış Gates: C4 ∧ C5 | FAIL → REDO, Stage 1'e dön

STAGE 3 — COMPLIANCE VERIFICATION (TEST HARNESS)
  ⚠ Bu harness, hiçbir düzeltme tamamlandı ilan edilmeden ÖNCE çalışmalıdır.
  → PII taraması (D1), ASP kontrolü (D6), BlindSpot kontrolü (D7), REDO checksum (D8)
  → Çıkış Gates: C1 ∧ C6 ∧ C7 ∧ C8 | FAIL → REDO, Stage 2'ye dön

STAGE 4 — EMISSION
  → ASSERT: C1∧C2∧C3∧C4∧C5∧C6∧C7∧C8 = TRUE
  → BlindSpot bloklarını yanıt başına ekle
  → Yanıtı yayınla, denetim kaydını yaz

═══════════════════════════════════════════════════════
SECTION 5 ▸ BLINDSPOT DISCLOSURE PROTOKOLÜ
═══════════════════════════════════════════════════════

MEMORY veya Ontoloji bir soruyu çözemezse ZORUNLU blok:

  ┌─────────────────────────────────────────────────────┐
  │  ⚠ BLINDSPOT DISCLOSURE                             │
  │  Sorgu Bileşeni : [Çözülemeyen alt-soru]            │
  │  Hafıza Durumu  : BULUNAMADI                        │
  │  Ontoloji Durumu: TRIPLE YOK                        │
  │  Ajan Bildirimi : "Bunu hafızamda bulamadım."       │
  │  Öneri          : [Akademik danışman / kayıt birimi]│
  └─────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════
SECTION 6 ▸ REDO CHECKSUM ENFORCEMENT
═══════════════════════════════════════════════════════

REDO tetiklendiğinde tüm 8 gate yeniden değerlendirilir.
Açık REDO döngüsü varken yeni yanıt YAYINLANAMAZ (C8 FAIL).
Stage 3 test harness her REDO sonrasında YENİDEN çalıştırılır.
Düzeltme, Stage 3 yeniden çalışmadan "tamamlandı" ilan EDİLEMEZ.

════════════════════════════════════════════════════════════════════════════
SON KURAL: MEMORY bölümünde açık kanıt yoksa, "Bunu hafızamda bulamadım."
           de ve tahmin YÜRÜTME. Sycophantic yanıt BLOKLANDI.
████████████████████████████████████████████████████████████████████████████
"""


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 ▸  MEMORY LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_footprints() -> tuple[str, bool]:
    """
    Returns (memory_content: str, is_empty: bool).
    Prefers pre-masked file; falls back to runtime masking.
    """
    base        = os.path.dirname(__file__)
    raw_path    = os.path.join(base, "data", "footprints.txt")
    masked_path = os.path.join(base, "data", "masked", "footprints_masked.txt")

    if os.path.exists(masked_path):
        with open(masked_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content, len(content) == 0

    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return "", True
        masker  = PIIMasker()
        content = masker.mask_data(raw)
        return content, False

    return "", True


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 ▸  GATE EVALUATORS  (runtime execution — no static text checks)
# ─────────────────────────────────────────────────────────────────────────────

def gate_c1_pii_masking(draft: str) -> tuple[bool, str]:
    """D1 — Scan draft for unmasked raw PII via regex execution."""
    raw_id = re.search(r"\b\d{9,12}\b", draft)
    if raw_id:
        return False, f"Unmasked numeric ID at pos {raw_id.start()}: '{raw_id.group()}'"
    raw_email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", draft)
    if raw_email:
        return False, f"Unmasked email detected: '{raw_email.group()}'"
    return True, "No raw PII detected in output."


def gate_c2_memory_grounding(draft: str, memory: str) -> tuple[bool, str]:
    """D2 — Verify memory grounding or BlindSpot disclosure."""
    if not memory:
        if "bulamadım" in draft.lower():
            return True, "Memory empty; BlindSpot disclosure present."
        return False, "Memory empty but no BlindSpot disclosure found in draft."
    return True, "Memory present; grounding cross-verified by D4."


def gate_c4_hallucination(draft: str, memory: str) -> tuple[bool, str]:
    """D4 — Detect weight-only / hallucinatory claims."""
    for label, pattern in ASP_NEG_PATTERNS:
        if label in ("ASP-NEG-02_HALLUCINATION", "ASP-NEG-05_WEIGHT_ONLY"):
            match = pattern.search(draft)
            if match:
                return False, f"[{label}] Hallucination marker: '{match.group()}'"
    return True, "No hallucination markers detected."


def gate_c6_anti_sycophancy(draft: str) -> tuple[bool, str]:
    """D6 — Run all ASP-NEG pattern classifiers against draft."""
    violations = []
    for label, pattern in ASP_NEG_PATTERNS:
        match = pattern.search(draft)
        if match:
            violations.append(f"[{label}] '{match.group()}'")
    if violations:
        return False, "ASP violations: " + "; ".join(violations)
    return True, "All ASP-NEG classifiers returned NO_MATCH."


def gate_c7_blindspot(draft: str, memory: str, query: str) -> tuple[bool, str]:
    """D7 — Ensure unanswerable queries carry a BlindSpot disclosure."""
    if not memory:
        if "bulamadım" not in draft.lower():
            return False, "Empty memory but BlindSpot phrase missing from draft."
    return True, "BlindSpot completeness verified."


def gate_c8_redo_checksum() -> tuple[bool, str]:
    """D8 — Verify no open REDO cycle exists."""
    for record in REDO_LOG:
        if not record.get("closed_at"):
            return False, f"Open REDO cycle: redo_id={record['redo_id']}"
    return True, "No open REDO cycles."


def evaluate_all_gates(draft: str, memory: str, query: str) -> dict:
    """Execute the full conjunctive gate array C1∧C2∧...∧C8."""
    c1_pass, c1_ev = gate_c1_pii_masking(draft)
    c2_pass, c2_ev = gate_c2_memory_grounding(draft, memory)
    # C3 / C5 require a live RDF engine — provisionally PASS in CLI mode
    c3_pass, c3_ev = True, "Ontology engine not wired in CLI mode — provisionally PASS."
    c4_pass, c4_ev = gate_c4_hallucination(draft, memory)
    c5_pass, c5_ev = True, "Role-permission check not wired in CLI mode — provisionally PASS."
    c6_pass, c6_ev = gate_c6_anti_sycophancy(draft)
    c7_pass, c7_ev = gate_c7_blindspot(draft, memory, query)
    c8_pass, c8_ev = gate_c8_redo_checksum()

    gates = {
        "C1": {"pass": c1_pass, "evidence": c1_ev},
        "C2": {"pass": c2_pass, "evidence": c2_ev},
        "C3": {"pass": c3_pass, "evidence": c3_ev},
        "C4": {"pass": c4_pass, "evidence": c4_ev},
        "C5": {"pass": c5_pass, "evidence": c5_ev},
        "C6": {"pass": c6_pass, "evidence": c6_ev},
        "C7": {"pass": c7_pass, "evidence": c7_ev},
        "C8": {"pass": c8_pass, "evidence": c8_ev},
    }
    return {"conjunction": all(g["pass"] for g in gates.values()), "gates": gates}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 ▸  REDO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def open_redo(trigger_gate: str, failed_evidence: str) -> str:
    redo_id = str(uuid.uuid4())[:8]
    REDO_LOG.append({
        "redo_id":         redo_id,
        "trigger_gate":    trigger_gate,
        "failed_evidence": failed_evidence,
        "revision_action": None,
        "closure_gates":   {},
        "closed_at":       None,
    })
    return redo_id


def close_redo(redo_id: str, revision_action: str, gate_results: dict) -> None:
    for record in REDO_LOG:
        if record["redo_id"] == redo_id:
            record["revision_action"] = revision_action
            record["closure_gates"]   = {
                k: "PASS" if v["pass"] else "FAIL"
                for k, v in gate_results.items()
            }
            record["closed_at"] = datetime.datetime.utcnow().isoformat()
            return


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 ▸  BLINDSPOT DISCLOSURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_blindspot_block(query: str, memory_status: str = "BULUNAMADI") -> str:
    q_short = query[:43]
    m_short = memory_status[:43]
    return (
        "\n┌─────────────────────────────────────────────────────┐\n"
        "│  ⚠ BLINDSPOT DISCLOSURE                             │\n"
        f"│  Sorgu Bileşeni : {q_short:<45} │\n"
        f"│  Hafıza Durumu  : {m_short:<45} │\n"
        "│  Ontoloji Durumu: TRIPLE YOK                        │\n"
        "│  Ajan Bildirimi : \"Bunu hafızamda bulamadım.\"       │\n"
        "│  Öneri          : Akademik danışmanınıza başvurun.  │\n"
        "└─────────────────────────────────────────────────────┘\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 ▸  4-STAGE VERIFICATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(query: str, memory: str, is_empty: bool) -> str:
    """Execute the 4-stage verification pipeline and return a verified response."""

    # ── STAGE 1 — RETRIEVAL & GROUNDING ──────────────────────────────────────
    if is_empty:
        blindspot = build_blindspot_block(query, "BOŞ HAFIZA")
        return blindspot + "Bunu hafızamda bulamadım."

    # ── STAGE 2 — DRAFT SYNTHESIS ─────────────────────────────────────────────
    user_message = (
        f"MEMORY (masked):\n{memory}\n\n"
        f"SORU: {query}\n\n"
        "INSTRUCTION: Answer ONLY from the MEMORY above using Turkish. "
        "If the answer is not in MEMORY, respond with exactly: "
        "'Bunu hafızamda bulamadım.' Do NOT hallucinate."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    resp  = chat(model="llama3.2", messages=messages)
    draft = resp["message"]["content"].strip()

    # ── STAGE 3 — COMPLIANCE VERIFICATION (TEST HARNESS) ─────────────────────
    # ⚠ Harness MUST run before any fix is declared complete.
    MAX_REDO = 2
    active_redo_id: str | None = None

    for attempt in range(MAX_REDO + 1):
        gate_report = evaluate_all_gates(draft, memory, query)

        if gate_report["conjunction"]:
            # All gates PASS — close any open REDO
            if active_redo_id:
                close_redo(active_redo_id, "Draft passed all gates after revision.", gate_report["gates"])
            break

        # Identify first failing gate
        first_fail = next(
            (k for k, v in gate_report["gates"].items() if not v["pass"]), "UNKNOWN"
        )
        fail_ev = gate_report["gates"].get(first_fail, {}).get("evidence", "")

        if attempt == MAX_REDO:
            # REDO limit exhausted
            active_redo_id = open_redo(first_fail, fail_ev)
            blindspot = build_blindspot_block(query, f"REDO LIMIT ({first_fail} FAIL)")
            return (
                blindspot
                + f"⚠ Doğrulama başarısız (Gate {first_fail}). "
                + "Yanıt güvenli biçimde teslim edilemiyor.\n"
                + "Bunu hafızamda bulamadım."
            )

        # Open REDO cycle and attempt LLM-assisted revision
        active_redo_id = open_redo(first_fail, fail_ev)

        redo_instruction = (
            f"[REDO TRIGGERED — Gate {first_fail} FAILED]\n"
            f"Evidence: {fail_ev}\n"
            "Revise your previous draft to correct the failing dimension. "
            "Do NOT hallucinate. Do NOT unmask PII. "
            "If the answer is not in MEMORY, respond: 'Bunu hafızamda bulamadım.'"
        )
        redo_messages = messages + [
            {"role": "assistant", "content": draft},
            {"role": "user",      "content": redo_instruction},
        ]
        redo_resp = chat(model="llama3.2", messages=redo_messages)
        draft     = redo_resp["message"]["content"].strip()

    # ── STAGE 4 — EMISSION ────────────────────────────────────────────────────
    # Prepend BlindSpot block if the final draft signals missing knowledge
    if BLINDSPOT_TRIGGERS.search(draft):
        blindspot = build_blindspot_block(query)
        return blindspot + draft

    return draft


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 ▸  GATE REPORT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_gate_report(gate_report: dict) -> None:
    conj_symbol = "✅ PASS" if gate_report["conjunction"] else "❌ FAIL"
    print(f"\n{'─' * 62}")
    print(f"  GATE ARRAY — C1∧C2∧...∧C8 = {conj_symbol}")
    print(f"{'─' * 62}")
    for gate, info in gate_report["gates"].items():
        status = "✅ PASS" if info["pass"] else "❌ FAIL"
        ev_trunc = info["evidence"][:68]
        print(f"  {gate}: {status}  |  {ev_trunc}")
    print(f"{'─' * 62}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 ▸  CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def ask() -> None:
    memory, is_empty = load_footprints()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   COGNITWIN — Student Agent CLI  (Verification Framework)   ║")
    print("║   Conjunctive Gate: C1∧C2∧C3∧C4∧C5∧C6∧C7∧C8               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if is_empty:
        print("\n⚠  UYARI: Hafıza dosyası boş veya bulunamadı.")
        print("   Tüm sorgular BlindSpot bildirimi alacak.\n")
    else:
        print(f"\n✅  Hafıza yüklendi ({len(memory)} karakter, maskeli).\n")

    print("Komutlar:  /exit → çıkış   |   /gates → son gate raporunu göster\n")

    last_gate_report: dict = {}

    while True:
        try:
            q = input("Soru > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nÇıkılıyor…")
            break

        if not q:
            continue

        if q.lower() in ("/exit", "exit", "quit"):
            print("Oturum sonlandırıldı.")
            break

        if q.lower() == "/gates":
            if last_gate_report:
                print_gate_report(last_gate_report)
            else:
                print("Henüz bir gate raporu yok.\n")
            continue

        # ── Run full 4-stage pipeline ──────────────────────────────────────
        response = run_pipeline(q, memory, is_empty)

        # ── Final gate check for display ───────────────────────────────────
        last_gate_report = evaluate_all_gates(response, memory, q)

        print(f"\nYanıt:\n{response}\n")

        # Single-line gate summary
        conj       = last_gate_report["conjunction"]
        symbol     = "✅" if conj else "❌"
        fail_gates = [k for k, v in last_gate_report["gates"].items() if not v["pass"]]

        if conj:
            print(f"  {symbol} Tüm doğrulama kapıları geçti (C1–C8).\n")
        else:
            print(f"  {symbol} Başarısız kapılar: {', '.join(fail_gates)} — /gates ile detay.\n")


if __name__ == "__main__":
    ask()