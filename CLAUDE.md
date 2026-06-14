# vision-service

Vision/OCR-Microservice, der vom **llm-router** als Spoke mit den Capabilities `vision` und `ocr` angesprochen wird. Bündelt mehrere Backends (Donut, Tesseract, EasyOCR, optional Chandra) hinter einer OpenAI-kompatiblen `/v1/`-API; Apps (audit_designer, flowinvoice, workshop) reden nicht direkt mit dem Service, sondern über den Router.

## Tech-Stack

- **Sprache:** Python (`requires-python >= 3.11`; Docker baut auf `python:3.11-slim`)
- **Web-Framework:** FastAPI + Uvicorn (`uvicorn[standard]`)
- **Vision/OCR:** transformers (`>=4.40,<4.50`, gepinnt wegen NameError-Bug) + torch/torchvision (CPU-Wheels im Build), Donut (`naver-clova-ix/donut-base-finetuned-cord-v2`), Tesseract via `pytesseract`, EasyOCR (optional), Chandra (optional, Best-Effort)
- **Bild/PDF:** Pillow, pdfplumber, pdf2image (poppler)
- **Port:** `8005` (`VISION_PORT`)
- **Deployment:** Docker Multi-Stage (baked Donut-Modell), Compose; Image `ghcr.io/janpow77/vision-service`; GitHub Actions Build/Push nach GHCR

## Setup & Befehle

```bash
# Lokales Setup (editierbar) + Dev-Tools
pip install -e .[dev]

# Tests (mocken alle Backends — kein Modell-Download)
pytest

# Lint/Format (ruff ist als dev-Extra deklariert)
ruff check .

# Lokal starten (Modul aus src/)
uvicorn vision_service.main:app --host 0.0.0.0 --port 8005

# Docker-Build + Push (CPU-Image mit vorgeladenem Donut)
docker build -t ghcr.io/janpow77/vision-service:v0.1 .
docker push ghcr.io/janpow77/vision-service:v0.1

# Compose (CPU-Default)
docker compose --env-file /etc/vision-service/env up -d

# Compose mit GPU-Override
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

Smoke-Test gegen einen laufenden Service (Beispiel-IP aus README):

```bash
curl http://100.102.132.11:8005/health
curl -X POST http://100.102.132.11:8005/v1/ocr -F image=@invoice.png -F backend=tesseract
curl -X POST http://100.102.132.11:8005/v1/vision/parse -F image=@invoice.png -F model=donut-cord-v2
```

## Struktur

- `src/vision_service/` — Python-Paket (Layout `src/`, gefunden über `[tool.setuptools.packages.find]`)
  - `main.py` — FastAPI-App + Endpunkte: `GET /health`, `GET /v1/models`, `POST /v1/vision/parse`, `POST /v1/ocr`, `GET /`. Lifespan lädt Donut bei `eager_load` vor; API-Key-Check via `X-Api-Key`-Header (nur wenn `VISION_API_KEY` gesetzt).
  - `config.py` — `Config`-Dataclass (Singleton via `get_config()`), komplett ENV-gesteuert (`VISION_*`); `resolve_device()` löst `auto` → `cuda`/`cpu` auf.
  - `registry.py` — zentrale Backend-Registry: `collect_backend_info()`, `pick_ocr_backend()` (Auto-Routing), `run_ocr()`, `run_vision()`. Eine Stelle für `/health`, `/v1/models` und OCR-Routing.
  - `models/` — ein Loader pro Backend (jeweils `get_*()` + `available()` + `run()`): `donut.py`, `tesseract.py`, `easyocr.py`, `chandra.py`.
- `tests/` — API-Smoke-Tests (`test_api.py`) mit gemockten Backends (`conftest.py`).
- `compose.yaml` / `compose.gpu.yaml` — Deployment (Bind-Mount `/var/lib/vision-service/data` für HF-Cache).
- `Dockerfile` — Multi-Stage: `deps` (Wheels + Donut-Preload) → `runtime` (Tesseract/poppler/libgl, Modell aus baked Cache).
- `.env.example` — ENV-Vorlage (Produktiv-Pfad `/etc/vision-service/env`).
- `.github/workflows/image.yml` — GHCR Build & Push (CPU-only, baked Donut).
- `graphify-out/` — generierter Code-Graph (`graph.json`, `graph.html`).

## Konventionen

- **Lint/Format:** Ruff (`line-length = 100`, `target-version = py311`, Regeln `E,F,W,I,B,UP`, `E501` ignoriert).
- **Tests:** pytest; Backends werden gemockt, es werden keine Modelle geladen (siehe `tests/conftest.py`).
- **Konfiguration:** ausschließlich über `VISION_*`-Umgebungsvariablen (siehe `config.py` / `.env.example`), keine Config-Dateien.
- **API-Konvention:** OpenAI-kompatibler `/v1/`-Prefix, `GET /health` liefert `{"status":"ok"}`, `/v1/models` listet Backends mit `capabilities` — abgestimmt auf die Spoke-Registrierung im llm-router. `/health` bleibt auch bei gesetztem API-Key offen.
- **Sprache:** Code-Strings und Doku auf Deutsch mit echten Umlauten; Fehlermeldungen teils deutsch (z.B. „nicht unterstuetzt").
