# CogniTwin LibreChat Yapılandırması

Bu klasör LibreChat'in CogniTwin backend'ine bağlanması için gerekli endpoint
yapılandırma dosyalarını içerir.

---

## Giriş Noktası — CogniTwin Portal

Kullanıcılar önce **`http://<sunucu>:8080`** adresini açar.

Burada iki seçenek görünür:
- **Öğrenci Workspace** → `http://<sunucu>:3900`
- **Proje / Agile Workspace** → `http://<sunucu>:3901`

Seçim yapıldığında doğru LibreChat instance'ına otomatik yönlendirilir.
Yönlendirme JavaScript ile hostname'den türetilir; ekstra yapılandırma gerekmez.

---

## İki Ayrı LibreChat Instance

CogniTwin iki farklı LibreChat instance çalıştırır.  Her instance yalnızca kendi
kullanıcı kitlesine ait model(ler)i görünür kılar.

| Instance       | Port   | Hedef Kullanıcı       | Görünen Modeller                                    |
|----------------|--------|-----------------------|-----------------------------------------------------|
| **Student**    | 3900   | Öğrenciler            | `cognitwin-student-llm`                             |
| **Agile**      | 3901   | Proje kullanıcıları   | Composer · Product Owner · Scrum Master · Developer |

**Öğrencilere verin:** `http://<sunucu>:3900`
**Proje kullanıcılarına verin:** `http://<sunucu>:3901`

---

## Erişim Denetimi — İki Katman

```
Kullanıcı HTTP isteği
        │
        ▼
LibreChat UI katmanı (YAML config)
  • librechat.student.yaml → sadece student endpoint var
  • librechat.agile.yaml   → sadece agile endpoint var
  • Tüm built-in endpoint'ler (openAI, anthropic, google…) disabled: true
        │
        ▼
CogniTwin backend katmanı (HTTP 403)
  • Bearer token → rol çözümleme (student / agile / admin)
  • Yanlış rol + model → HTTP 403 Forbidden
  • Kullanıcı model adını tahmin edip elle yazsa bile reddedilir
```

---

## Yapılandırma Dosyaları

| Dosya                       | Hangi Instance        | İçerik                                  |
|-----------------------------|-----------------------|-----------------------------------------|
| `librechat.student.yaml`    | Student (port 3900)   | Sadece CogniTwin Student endpoint       |
| `librechat.agile.yaml`      | Agile (port 3901)     | Sadece CogniTwin Agile endpoint         |
| `librechat.yaml`            | ⚠️ Kullanılmıyor      | Eski birleşik config — yalnızca referans|

---

## Hızlı Başlangıç

```bash
# 1. Değişkenleri ayarla
cp .env.example .env
# .env içindeki key'leri değiştir (üretim için zorunlu)

# 2. Stack'i başlat
docker compose up -d --build

# 3. Öğrenciler için
open http://localhost:3900

# 4. Proje kullanıcıları için
open http://localhost:3901
```

---

## API Key Yönetimi

`.env` dosyasındaki değişkenler:

```
COGNITWIN_STUDENT_KEY=cognitwin-student   # Öğrenci LibreChat'e gönderilir
COGNITWIN_AGILE_KEY=cognitwin-agile       # Agile LibreChat'e gönderilir
COGNITWIN_ADMIN_KEY=cognitwin-admin       # Tüm modellere erişim
```

**Üretimde** bu değerleri güçlü, rastgele string'lerle değiştirin.
Agile key'i yalnızca proje kullanıcılarıyla paylaşın.

---

## Stack'i Sıfırlama (Değişiklikler Sonrası)

Yapılandırma değişikliklerinin etkili olması için container'ları yeniden başlatın:

```bash
docker compose down --remove-orphans
docker compose up -d --build
```

> **Windows notu:** Log yazma hatası alırsanız
> `infra/librechat/docker-compose.override.yml.example` dosyasını
> `docker-compose.override.yml` olarak kopyalayın.

---

## `infra/librechat/docker-compose.yml` Hakkında

Bu klasördeki `docker-compose.yml` **kullanılmıyor** — yalnızca referans amaçlıdır.
Tüm servisler kök dizindeki `docker-compose.yml` tarafından yönetilir.
