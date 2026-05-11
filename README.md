# vision-service

Vision/OCR-Microservice. Wird vom **llm-router** als Spoke mit den
Capabilities `vision` und `ocr` angesprochen — analog zum
[reranker-service](https://github.com/janpow77/reranker-service).

Buendelt:

| Backend | Capability | Default | Zweck |
|---------|-----------|---------|-------|
| **Donut** (`donut-cord-v2`) | `vision` | enabled | OCR-frei, strukturierte Rechnungs-Extraktion (CORD-v2) |
| **Tesseract** | `ocr` | enabled | Klassisches OCR, deutsch + englisch |
| **EasyOCR** | `ocr` | optional | Deep-Learning-OCR, mehrsprachig, GPU-aware |
| **Chandra** | `ocr` | optional | Best-Effort-Wrapper falls Modul vorhanden |

Apps (audit_designer, flowinvoice, workshop) reden **nicht direkt** mit
diesem Service — sie sprechen den llm-router, der dann an den passenden
vision-service-Spoke proxiet.

## Architektur

```
   audit_designer   flowinvoice   workshop
        |               |             |
        |---- X-App-Id -|---- X-Api-Key
                       v
              llm-router (CCX23, :7842)
                       |
        +--------------+--------------+
        v              v              v
     ollama       reranker-service   vision-service
   (chat, embed)  (cross-encoder)    (Donut/Tesseract/EasyOCR)
       NUC            NUC/evo         NUC/evo/Desktop
```

## API

OpenAI-kompatibel mit `/v1/`-Prefix. Body ist `multipart/form-data` (Bild +
Form-Felder), damit Binaries direkt durchgereicht werden.

### Strukturiertes Vision-Parsing (Donut)

```http
POST /v1/vision/parse
Content-Type: multipart/form-data
X-Api-Key: <optional>

--boundary
Content-Disposition: form-data; name="image"; filename="invoice.png"
Content-Type: image/png

<binary>
--boundary
Content-Disposition: form-data; name="model"

donut-cord-v2
--boundary--
```

Antwort:

```json
{
  "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
  "raw": "<s_total>123.45</s_total><s_date>2026-05-11</s_date>",
  "fields": {
    "total_amount": "123.45",
    "invoice_date": "2026-05-11"
  },
  "json": {"total": "123.45", "date": "2026-05-11"},
  "duration_ms": 1872,
  "device": "cuda"
}
```

### Klassisches OCR (Tesseract/EasyOCR/Chandra)

```http
POST /v1/ocr
Content-Type: multipart/form-data

image=<binary>
backend=auto|tesseract|easyocr|chandra
lang=deu+eng | de,en  (optional, backend-spezifisch)
```

Antwort:

```json
{
  "backend": "tesseract",
  "text": "Rechnung Nr. 4711 vom 11.05.2026 ...",
  "confidence": 0.93,
  "duration_ms": 247,
  "languages": "deu+eng"
}
```

### Endpoints

| Methode | Pfad | Zweck |
|---------|------|-------|
| GET | `/health` | Liveness inkl. Backend-Status + GPU-Info |
| GET | `/v1/models` | OpenAI-Style Modell-Liste (`capabilities: ["vision"\|"ocr"]`) |
| POST | `/v1/vision/parse` | Donut (strukturiert) |
| POST | `/v1/ocr` | OCR-Backend mit `auto`-Routing |

## Konfiguration

| ENV | Default | Zweck |
|-----|---------|-------|
| `VISION_HOST` | `0.0.0.0` | Listen-Adresse |
| `VISION_PORT` | `8005` | Port |
| `VISION_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |
| `VISION_DONUT_MODEL` | `naver-clova-ix/donut-base-finetuned-cord-v2` | Donut-Modell |
| `VISION_ENABLE_DONUT` | `true` | Donut-Backend aktiv |
| `VISION_ENABLE_TESSERACT` | `true` | Tesseract-Backend aktiv |
| `VISION_ENABLE_EASYOCR` | `false` | EasyOCR-Backend aktiv (Extra noetig) |
| `VISION_ENABLE_CHANDRA` | `false` | Chandra-Backend aktiv (Modul noetig) |
| `VISION_TESSERACT_LANGS` | `deu,eng` | Tesseract-Sprachen |
| `VISION_EASYOCR_LANGS` | `de,en` | EasyOCR-Sprachen |
| `VISION_OCR_DEFAULT_BACKEND` | `tesseract` | Was `backend=auto` waehlt |
| `VISION_API_KEY` | _leer_ | Wenn gesetzt → `X-Api-Key` Header verlangt |
| `VISION_MAX_IMAGE_BYTES` | `10485760` (10 MB) | Limit pro Image |
| `VISION_MAX_PDF_PAGES` | `50` | Limit Seiten pro PDF |
| `VISION_EAGER_LOAD` | `true` | Donut beim Startup vorladen |
| `HF_HOME` | `/data/hf_cache` | HuggingFace-Cache-Verzeichnis |

## Modelle

- **Donut** wird im Docker-Build vorgeladen (Default:
  `naver-clova-ix/donut-base-finetuned-cord-v2`, ~800 MB VRAM, ~750 MB Disk).
- **Tesseract** wird als System-Paket installiert (`tesseract-ocr` +
  Sprachpakete `deu`/`eng`).
- **EasyOCR** wird nur installiert, wenn das Image mit dem
  `easyocr`-Extra gebaut wird (`pip install .[easyocr]`); andernfalls bleibt
  das Backend `available=false` und wird in `/v1/models` nicht beworben.
- **Chandra** ist Best-Effort: wenn ein Python-Modul `chandra_ocr` oder
  `chandra` mit `read_text(image)`-API verfuegbar ist, wird es genutzt.

## Deployment

### Build + Push

```bash
docker build -t ghcr.io/janpow77/vision-service:v0.1 .
docker push ghcr.io/janpow77/vision-service:v0.1
```

### Auf dem NUC starten

```bash
sudo mkdir -p /var/lib/vision-service/data
sudo chown 1000:1000 /var/lib/vision-service/data
cp .env.example /etc/vision-service/env
# /etc/vision-service/env editieren -> VISION_API_KEY setzen
docker compose --env-file /etc/vision-service/env up -d
```

### GPU-Variante (z.B. evo-x2 mit RTX 5070 Ti)

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

Voraussetzung: nvidia-container-toolkit installiert + GPU-Build des Images
(im Default-CPU-Image faellt Donut intern auf CPU zurueck).

## llm-router-Integration

Nach Deploy einen Spoke im llm-router anlegen:

```yaml
spokes:
  - name: nuc-vision
    base_url: http://100.102.132.11:8005
    type: openai
    capabilities: [vision, ocr]
    enabled: true
    priority: 10

routes:
  - model_glob: "donut*"
    spoke_id: <id-of-nuc-vision>
  - model_glob: "tesseract"
    spoke_id: <id-of-nuc-vision>
  - model_glob: "easyocr"
    spoke_id: <id-of-nuc-vision>
```

Apps sprechen ab dann `https://llm-router.intern/v1/vision/parse` bzw.
`/v1/ocr` mit ihren gewohnten Headern (`X-App-Id`, `X-Api-Key`) — Router
routet automatisch.

## Smoke-Test

```bash
# Direkt am Service (Tailscale)
curl -X POST http://100.102.132.11:8005/v1/ocr \
  -F image=@invoice.png \
  -F backend=tesseract

# Donut
curl -X POST http://100.102.132.11:8005/v1/vision/parse \
  -F image=@invoice.png \
  -F model=donut-cord-v2

# Health
curl http://100.102.132.11:8005/health
```

## Tests

```bash
pip install -e .[dev]
pytest
```

Tests mocken Donut, Tesseract, EasyOCR und Chandra — es werden **keine** Modelle
heruntergeladen.
