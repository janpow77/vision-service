# Architektur — vision-service

_Automatisch generiert von graphify-kira aus dem Code-Graphen. Nicht von Hand editieren — wird beim nächsten Lauf überschrieben._

**Umfang:** 116 Knoten, 220 Kanten, 9 größere Module, 0 zirkuläre Abhängigkeiten.

## Modulkarte

- **API Testing** (24): `test_api.py`
- **OCR Backends** (22): `config.py`, `chandra.py`, `easyocr.py`, `tesseract.py`, `registry.py`
- **Donut Backend** (18): `__init__.py`, `config.py`, `main.py`, `donut.py`, `registry.py`
- **Document Parser** (14): `config.py`, `donut.py`
- **Vision Parsing** (13): `main.py`
- **EasyOCR Backend** (7): `easyocr.py`
- **Tesseract Backend** (6): `tesseract.py`
- **Pytest Setup** (5): `conftest.py`
- **Chandra Backend** (4): `chandra.py`

## Zentrale Bausteine (God Nodes)

_Hohe Zentralität ist nicht automatisch ein Defekt (zentrale Stores/Modelle sind oft legitim). Konkrete Refactoring-Prioritäten siehe Optimierungs-Report._

- `get_config() (src/vision_service/config.py)` — Grad 24 (ein 23/aus 1)
- `_client() (tests/test_api.py)` — Grad 14 (ein 14/aus 0)
- `Config (src/vision_service/config.py)` — Grad 2 (ein 2/aus 0)
- `main.py (src/vision_service/main.py)` — Grad 20 (ein 1/aus 19)
- `registry.py (src/vision_service/registry.py)` — Grad 17 (ein 2/aus 15)
- `test_api.py (tests/test_api.py)` — Grad 17 (ein 1/aus 16)
- `DonutBackend (src/vision_service/models/donut.py)` — Grad 11 (ein 3/aus 8)
- `collect_backend_info() (src/vision_service/registry.py)` — Grad 11 (ein 6/aus 5)
- `config.py (src/vision_service/config.py)` — Grad 12 (ein 7/aus 5)
- `resolve_device() (src/vision_service/config.py)` — Grad 9 (ein 9/aus 0)

## Schnittstellen / Brücken (Betweenness)

- `collect_backend_info() (src/vision_service/registry.py)` — Betweenness 0.017
- `DonutBackend (src/vision_service/models/donut.py)` — Betweenness 0.017
- `get_donut() (src/vision_service/models/donut.py)` — Betweenness 0.014
- `EasyOCRBackend (src/vision_service/models/easyocr.py)` — Betweenness 0.008
- `get_easyocr() (src/vision_service/models/easyocr.py)` — Betweenness 0.006
- `TesseractBackend (src/vision_service/models/tesseract.py)` — Betweenness 0.006
- `registry.py (src/vision_service/registry.py)` — Betweenness 0.005
- `get_tesseract() (src/vision_service/models/tesseract.py)` — Betweenness 0.005
- `main.py (src/vision_service/main.py)` — Betweenness 0.005
- `pick_ocr_backend() (src/vision_service/registry.py)` — Betweenness 0.005

## Hinweis für Änderungen

Vor dem Ändern eines zentralen Bausteins die Abhängigen prüfen — am schnellsten über den **graphify-MCP** (globaler Graph): „Was hängt an `<datei>`?". Brücken-Knoten stabil halten.

