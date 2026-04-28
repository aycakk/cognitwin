# HR + n8n Entegrasyonu (Devam)

Bu doküman CogniTwin HR workspace'i ile n8n webhook otomasyonunun çalışma şeklini açıklar.

## 1) Çalıştırma

1. `.env.example` dosyasını `.env` olarak kopyalayın.
2. HR için gerekli anahtarları kontrol edin:
   `COGNITWIN_HR_KEY`, `N8N_ENABLED`, `N8N_WEBHOOK_URL` veya `N8N_WEBHOOK_BASE_URL`.
3. Stack'i başlatın:
   `docker compose up -d --build`
4. Portal: `http://localhost:8080`
5. HR workspace: `http://localhost:3902`
6. n8n UI: `http://localhost:5678`
7. Docker Desktop içinde container adını doğrulayın: `cognitwin-n8n`

Terminal doğrulaması:

- `docker compose config --services` çıktısında `n8n` görünmelidir.
- `docker compose ps` çıktısında `cognitwin-n8n` görünmelidir.

## 2) HR Workspace Ayrımı

- Student workspace sadece `cognitwin-student-llm` modelini görür.
- Agile workspace sadece Agile modellerini görür.
- HR workspace sadece `cognitwin-hr` modelini görür.
- Backend yönlendirme `src/pipeline/router.py` ile ayrıdır; HR modu Agile içine gömülü değildir.

## 3) Yapılandırılmış HR Yanıtı

Backend, HR LLM metninden aşağıdaki alanları çıkarır ve `AgentResponse.metadata["structured_response"]` içinde tutar:

- `decision`
- `candidate_name`
- `job_title`
- `score`
- `strengths`
- `missing_skills`
- `risks`
- `shortlist_status`
- `automation_targets`
- `recommended_actions`
- `follow_up_actions`
- `should_trigger_automation`
- `recruiter_summary`
- `token_cost`
- `remaining_budget`

LibreChat tarafında kullanıcı normal metin yanıtı görür; yapılandırılmış veri backend içindir.

## 4) n8n Tetikleme Kuralları

Tetikleme her yanıtta yapılmaz.

- Önce `intent` için izinli aksiyon listesi kontrol edilir (`INTENT_AUTOMATION_MAP`).
- Sonra `recommended_actions/automation_targets` alanları doğrulanır.
- Sadece izinli aksiyonlar için payload üretilir.
- Webhook çağrısı asenkron yapılır (kullanıcı yanıtı bloklanmaz).

Örnek aksiyonlar:

- `shortlist_to_sheets`
- `notify_slack`
- `send_outreach_email`
- `create_calendar_event`
- `log_to_ats`

## 5) Webhook Konfigürasyonu

Global (tek URL) yaklaşımı:

- `N8N_WEBHOOK_URL=https://...`

Per-action yaklaşımı:

- `N8N_SHORTLIST_WEBHOOK`
- `N8N_SLACK_WEBHOOK`
- `N8N_OUTREACH_WEBHOOK`
- `N8N_INTERVIEW_WEBHOOK`
- `N8N_MATCH_WEBHOOK`

Fallback:

- `N8N_WEBHOOK_BASE_URL=http://n8n:5678/webhook`

Not:

- Backend container içinden n8n'ye `http://n8n:5678` adı ile gider (aynı docker ağı).
- Host makineden n8n arayüzü `http://localhost:5678` ile açılır.

## 6) Hata Davranışı

- n8n kapalıysa (`N8N_ENABLED=false`) tetikleme sessizce atlanır.
- n8n erişilemezse hata loglanır; recruiter yanıtı yine döner.
- Tetikleme denemeleri `audit` içine `automation_dispatch` veya `automation_dispatch_failed` olarak yazılır.

## 7) n8n Workflow Import

Hazır workflow dosyası: `infra/n8n/hr_workflow.json`

1. n8n UI'ı açın: `http://localhost:5678`
2. Sol menüden **Workflows → Import from File** seçin.
3. `infra/n8n/hr_workflow.json` dosyasını yükleyin.
4. Workflow açıldıktan sonra **Activate / Publish** butonuna tıklayın.

### Webhook URL farkı — önemli

| URL | Ne zaman kullanılır |
|---|---|
| `http://localhost:5678/webhook-test/hr-action` | Yalnızca **manuel n8n testi** için. n8n "Listen for test event" modunda olmalı. Backend'den çağrılamaz. |
| `http://localhost:5678/webhook/hr-action` | **LibreChat / backend production testi**. Workflow Published/Active olmalı. |
| `http://n8n:5678/webhook/hr-action` | **Docker ortamında backend**. Aynı Docker network içindeki container'lardan kullanılır. |

Backend için `.env` ayarı:

```
N8N_ENABLED=true
# Host machine (local backend):
HR_N8N_WEBHOOK_URL=http://localhost:5678/webhook/hr-action
# Docker backend:
# HR_N8N_WEBHOOK_URL=http://n8n:5678/webhook/hr-action
```

Placeholder Set node'larını gerçek servislerle değiştirin (Sheets, Slack, Gmail, Calendar, ATS).

## 8) Slack Bildirimi Kurulumu

`notify_slack` artık gerçek bir Slack Incoming Webhook'u çağırır.

### Gerekli adımlar

1. **Slack Incoming Webhook oluşturun:**
   - `https://api.slack.com/apps` → uygulamanızı açın → **Incoming Webhooks** → **Add New Webhook to Workspace**
   - Hedef kanalı seçin → kopyalayın: `https://hooks.slack.com/services/T.../B.../...`

2. **n8n'de workflow değişkeni olarak tanımlayın (URL'yi dosyaya kaydetmeden):**
   - n8n UI → **Settings (sol alt)** → **Variables**
   - `SLACK_HR_WEBHOOK_URL` adıyla yeni değişken ekleyin, değer olarak Slack URL'sini yapıştırın.
   - Bu değişken `$vars.SLACK_HR_WEBHOOK_URL` ifadesiyle workflow içinde okunur.

3. **Workflow'u tekrar aktif edin** (değişken eklendikten sonra).

> **Önemli:** Slack webhook URL'sini `.env`, `hr_workflow.json` veya git'e izlenen hiçbir dosyaya yazmayın.

### Gönderilen mesaj formatı

```
*Yeni İK Bildirimi*

Aday: <candidate_name>
Pozisyon: <job_title>
Karar: <decision>
Puan: <score>
Kaynak: CognitWin HR Agent
```

### LibreChat test komutu

HR workspace'te şunu yazın:

> "Ahmet Yılmaz adayını Backend Developer pozisyonu için değerlendir ve Slack'e bildir."

Beklenen yanıt sonu:
```
---
OTOMASYON: Slack bildirimi n8n otomasyonuna iletildi.
```

n8n **Executions** sekmesinde yeni execution göreceksiniz ve hedef Slack kanalına mesaj gelecektir.

## 9) Hızlı Test Komutları

### `.env` ayarı (local test)

```
N8N_ENABLED=true
HR_N8N_WEBHOOK_URL=http://localhost:5678/webhook-test/hr-action
```

### curl (Linux / macOS / Git Bash)

```bash
curl -s -X POST http://localhost:5678/webhook-test/hr-action \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "shortlist_to_sheets",
    "intent": "shortlist",
    "recruiter_id": "recruiter-test01",
    "session_id": "test-session-001",
    "decision": "Önerilir",
    "candidate_name": "Ahmet Yılmaz",
    "candidate_id": "",
    "job_title": "Backend Developer",
    "job_id": "",
    "score": 87.5,
    "strengths": ["Python", "FastAPI"],
    "missing_skills": ["Kubernetes"],
    "risks": "Yok",
    "shortlist_status": "oluşturuldu",
    "source": "cognitwin_hr_agent",
    "triggered_at": "2026-04-28T10:00:00+00:00",
    "token_cost": 50,
    "remaining_budget": 950,
    "text_response": "Kısa liste oluşturuldu.",
    "extra": {}
  }'
```

### PowerShell (Windows)

```powershell
$body = @{
    action_type      = "shortlist_to_sheets"
    intent           = "shortlist"
    recruiter_id     = "recruiter-test01"
    session_id       = "test-session-001"
    decision         = "Önerilir"
    candidate_name   = "Ahmet Yılmaz"
    candidate_id     = ""
    job_title        = "Backend Developer"
    job_id           = ""
    score            = 87.5
    strengths        = @("Python", "FastAPI")
    missing_skills   = @("Kubernetes")
    risks            = "Yok"
    shortlist_status = "oluşturuldu"
    source           = "cognitwin_hr_agent"
    triggered_at     = "2026-04-28T10:00:00+00:00"
    token_cost       = 50
    remaining_budget = 950
    text_response    = "Kısa liste oluşturuldu."
    extra            = @{}
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:5678/webhook-test/hr-action" `
    -Method POST `
    -ContentType "application/json; charset=utf-8" `
    -Body $body
```

### Backend'den tetikleme testi (Python)

```bash
# .env içinde N8N_ENABLED=true ve HR_N8N_WEBHOOK_URL ayarlıyken:
python -c "
import os
os.environ['N8N_ENABLED'] = 'true'
os.environ['HR_N8N_WEBHOOK_URL'] = 'http://localhost:5678/webhook-test/hr-action'
from src.pipeline.hr.hr_schemas import N8nWebhookPayload
from src.pipeline.hr.n8n_client import trigger_automation
p = N8nWebhookPayload(
    action_type='shortlist_to_sheets',
    intent='shortlist',
    recruiter_id='rec-test',
    session_id='sess-test',
    candidate_name='Test Aday',
    job_title='Backend Dev',
    decision='Önerilir',
    score=82.0,
)
trigger_automation(p)
import time; time.sleep(1)
print('Tetikleme gönderildi — n8n execution geçmişini kontrol edin.')
"
```

### Beklenen davranış

- n8n **Executions** sekmesinde yeni bir execution görünür.
- Execution detayında **HR Webhook** node'u gelen payload'ı gösterir.
- **Route by action_type** node'u `shortlist_to_sheets` branch'ini seçer.
- **Shortlist → Sheets (placeholder)** node'u çalışır ve `status: placeholder` yazar.
- Backend loglarında (varsa): `n8n webhook OK action=shortlist_to_sheets`

## 10) Backend'den HR -> n8n Testi

1. n8n'de workflow aktif ve webhook-test modu açık olsun.
2. Backend loglarında `automation_dispatch` kaydını kontrol edin.
3. HR workspace'te tetikleyici bir istek gönderin (örn. "Bu adayları kısa listele ve Slack'e bildir").
4. `N8N_ENABLED=true` olduğundan emin olun.
