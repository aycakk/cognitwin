"""
test_e2e_agents.py — Canlı API üzerinden uçtan uca agent testi.

Her agent için:
  - HTTP isteği gönderilir (OpenAI-uyumlu /v1/chat/completions)
  - Yanıt yapısı doğrulanır
  - Akış (routing) doğrulanır
  - Cevap içeriği kontrol edilir
  - Süre ölçülür

Çalıştır:
    python tests/test_e2e_agents.py
"""

import io
import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

# Windows terminal UTF-8 zorla
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8011"


# ─────────────────────────────────────────────────────────────────────────────
#  Renkli terminal çıktısı
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}[OK]  {msg}{RESET}")
def fail(msg):  print(f"  {RED}[FAIL]{msg}{RESET}")
def info(msg):  print(f"  {CYAN}[i]  {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}[!]  {msg}{RESET}")
def header(msg):print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP yardımcısı
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Response:
    status:   int
    body:     dict
    elapsed:  float
    raw:      str


def post(path: str, payload: dict, timeout: int = 300) -> Response:
    url  = BASE_URL + path
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer cognitwin"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw  = resp.read().decode()
            body = json.loads(raw)
            return Response(resp.status, body, time.time() - t0, raw)
    except urllib.error.HTTPError as e:
        raw  = e.read().decode()
        body = {}
        try: body = json.loads(raw)
        except Exception: pass
        return Response(e.code, body, time.time() - t0, raw)


# ─────────────────────────────────────────────────────────────────────────────
#  Ortak doğrulama
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer(resp: Response) -> Optional[str]:
    """OpenAI formatından yanıt metnini çıkar."""
    try:
        return resp.body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def check_response_structure(resp: Response, agent_name: str) -> bool:
    passed = True

    if resp.status != 200:
        fail(f"HTTP {resp.status} döndü (200 beklendi)")
        passed = False
    else:
        ok(f"HTTP 200 OK  ({resp.elapsed:.1f}s)")

    if "choices" not in resp.body:
        fail("'choices' anahtarı yanıtta yok")
        passed = False
    else:
        ok("OpenAI formatı doğru (choices mevcut)")

    answer = extract_answer(resp)
    if not answer or not answer.strip():
        fail("Yanıt içeriği boş")
        passed = False
    else:
        preview = answer.strip()[:120].replace("\n", " ")
        ok(f"Yanıt alındı → \"{preview}…\"")

    if resp.elapsed > 90:
        warn(f"Yanıt süresi uzun: {resp.elapsed:.1f}s")
    elif resp.elapsed > 30:
        warn(f"Yanıt süresi: {resp.elapsed:.1f}s")
    else:
        ok(f"Süre normal: {resp.elapsed:.1f}s")

    return passed


# ─────────────────────────────────────────────────────────────────────────────
#  Test senaryoları
# ─────────────────────────────────────────────────────────────────────────────

def test_models_endpoint() -> bool:
    header("0 — /v1/models  (API Sağlık Kontrolü)")
    try:
        req = urllib.request.Request(BASE_URL + "/v1/models",
                                     headers={"Authorization": "Bearer cognitwin"})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read().decode())
        ids = [m["id"] for m in body.get("data", [])]
        expected = {
            "cognitwin-student-llm",
            "cognitwin-developer",
            "cognitwin-scrum",
            "cognitwin-product-owner",
            "cognitwin-composer",
        }
        missing = expected - set(ids)
        if missing:
            fail(f"Eksik modeller: {missing}")
            return False
        ok(f"Tüm modeller kayıtlı: {ids}")
        return True
    except Exception as e:
        fail(f"API'ye ulaşılamadı: {e}")
        return False


def test_student_agent() -> bool:
    header("1 — Student Agent  (cognitwin-student-llm)")
    info("Soru: 'Yazılım mühendisliğinde test nedir?'")

    resp = post("/v1/chat/completions", {
        "model":    "cognitwin-student-llm",
        "messages": [{"role": "user", "content": "Yazılım mühendisliğinde test nedir?"}],
        "stream":   False,
    })

    passed = check_response_structure(resp, "Student")

    answer = extract_answer(resp) or ""
    # Student agent TÜRKçe cevap vermeli
    turkish_words = ["test", "yazılım", "doğrulama", "hata", "kalite", "birim",
                     "entegrasyon", "sistem", "geliştirme", "kod"]
    hits = [w for w in turkish_words if w.lower() in answer.lower()]
    if hits:
        ok(f"Türkçe/ilgili içerik bulundu ({', '.join(hits[:4])})")
    else:
        warn("Beklenen anahtar kelimeler bulunamadı — içerik kontrol edilmeli")

    # Blindspot bloğu kontrolü (hafıza boşsa normal)
    if "BLINDSPOT" in answer.upper() or "hafızamda bulamadım" in answer:
        warn("Vector hafıza boş → blindspot modu aktif (veri yüklenirse düzelir)")

    return passed


def test_developer_agent() -> bool:
    header("2 — Developer Agent  (cognitwin-developer)")
    info("Soru: 'Pipeline mimarisi nasıl çalışıyor?'")

    resp = post("/v1/chat/completions", {
        "model":    "cognitwin-developer",
        "messages": [{"role": "user",
                      "content": "Pipeline mimarisi nasıl çalışıyor? Kısaca açıkla."}],
        "stream":   False,
    })

    passed = check_response_structure(resp, "Developer")

    answer = extract_answer(resp) or ""
    dev_words = ["pipeline", "agent", "gate", "orchestrat", "stage",
                 "developer", "mimari", "aşama", "doğrulama"]
    hits = [w for w in dev_words if w.lower() in answer.lower()]
    if hits:
        ok(f"Developer içeriği doğrulandı ({', '.join(hits[:4])})")
    else:
        warn("Developer anahtar kelimeleri bulunamadı — routing kontrol edilmeli")

    return passed


def test_scrum_master_agent() -> bool:
    header("3 — Scrum Master Agent  (cognitwin-scrum)")
    info("Soru: 'Sprint durumu nedir?'")

    resp = post("/v1/chat/completions", {
        "model":    "cognitwin-scrum",
        "messages": [{"role": "user", "content": "Sprint durumu nedir?"}],
        "stream":   False,
    })

    passed = check_response_structure(resp, "Scrum")

    answer = extract_answer(resp) or ""
    scrum_words = ["sprint", "görev", "task", "hedef", "scrum", "tamamlandı",
                   "devam", "bloke", "atanan", "durum"]
    hits = [w for w in scrum_words if w.lower() in answer.lower()]
    if hits:
        ok(f"Scrum içeriği doğrulandı ({', '.join(hits[:4])})")
    else:
        warn("Scrum anahtar kelimeleri bulunamadı — sprint_state.json kontrol edilmeli")

    # PII filtresi testi
    info("PII filtre testi: e-posta içeren sorgu")
    resp2 = post("/v1/chat/completions", {
        "model":    "cognitwin-scrum",
        "messages": [{"role": "user",
                      "content": "test@ornek.com adresine sprint raporu gönder"}],
        "stream":   False,
    })
    answer2 = extract_answer(resp2) or ""
    if "test@ornek.com" not in answer2:
        ok("PII maskeleme çalışıyor (e-posta sızdırılmadı)")
    else:
        fail("PII maskeleme ÇALIŞMIYOR — e-posta yanıtta görünüyor!")
        passed = False

    return passed


def test_stream_mode() -> bool:
    header("4 — Stream Modu  (cognitwin-student-llm, stream=true)")
    info("SSE stream yanıtı test ediliyor…")

    url  = BASE_URL + "/v1/chat/completions"
    data = json.dumps({
        "model":    "cognitwin-student-llm",
        "messages": [{"role": "user", "content": "Kısaca merhaba de."}],
        "stream":   True,
    }).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer cognitwin"},
        method="POST",
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode()
            elapsed = time.time() - t0
        if "data:" in raw and "[DONE]" in raw:
            ok(f"SSE stream formatı doğru  ({elapsed:.1f}s)")
            chunks = [l for l in raw.splitlines() if l.startswith("data:") and "[DONE]" not in l]
            ok(f"{len(chunks)} chunk alındı")
            return True
        else:
            fail("SSE stream formatı hatalı")
            info(f"Ham yanıt: {raw[:200]}")
            return False
    except Exception as e:
        fail(f"Stream testi başarısız: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Ana runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*60}")
    print("  CogniTwin — Uçtan Uca Agent Test Paketi")
    print(f"  Hedef: {BASE_URL}")
    print(f"{'='*60}{RESET}")

    results = {
        "API Sağlık":     test_models_endpoint(),
        "Student Agent":  test_student_agent(),
        "Developer Agent":test_developer_agent(),
        "Scrum Master":   test_scrum_master_agent(),
        "Stream Modu":    test_stream_mode(),
    }

    header("SONUÇ")
    total  = len(results)
    passed = sum(results.values())
    for name, ok_flag in results.items():
        sym = f"{GREEN}✅{RESET}" if ok_flag else f"{RED}❌{RESET}"
        print(f"  {sym}  {name}")

    print(f"\n{BOLD}  {passed}/{total} test geçti{RESET}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}  >>> Tüm agentlar çalışıyor!{RESET}\n")
    else:
        failed = [n for n, v in results.items() if not v]
        print(f"\n{RED}{BOLD}  Başarısız: {', '.join(failed)}{RESET}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
