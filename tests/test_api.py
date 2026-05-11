"""API-Smoke-Tests fuer vision-service (mit gemockten Backends).

Echte Modelle werden in CI nie geladen — siehe ``conftest.py``.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image


def _client():
    from fastapi.testclient import TestClient

    from vision_service.main import app

    return TestClient(app)


def _png_bytes(size=(64, 32), color=(255, 255, 255)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_contract():
    """llm-router-Konvention: GET /health → status=ok + backends-Liste."""
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "backends" in data
    ids = {b["id"] for b in data["backends"]}
    assert {"donut-cord-v2", "tesseract", "easyocr", "chandra"} <= ids
    # Donut + Tesseract sind in conftest auf available=True gesetzt
    by_id = {b["id"]: b for b in data["backends"]}
    assert by_id["donut-cord-v2"]["available"] is True
    assert by_id["tesseract"]["available"] is True


def test_models_list_capabilities():
    """llm-router-Konvention: /v1/models gibt OpenAI-Liste mit ``capabilities``."""
    c = _client()
    r = c.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    by_id = {m["id"]: m for m in data["data"]}
    assert "donut-cord-v2" in by_id
    assert by_id["donut-cord-v2"]["capabilities"] == ["vision"]
    assert "tesseract" in by_id
    assert by_id["tesseract"]["capabilities"] == ["ocr"]


def test_vision_parse_donut_mock():
    c = _client()
    files = {"image": ("test.png", _png_bytes(), "image/png")}
    data = {"model": "donut-cord-v2"}
    r = c.post("/v1/vision/parse", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "naver-clova-ix/donut-base-finetuned-cord-v2"
    assert body["fields"]["total_amount"] == "42.00"
    assert "raw" in body and body["raw"].startswith("<s_total>")


def test_vision_parse_unknown_model_400():
    c = _client()
    files = {"image": ("test.png", _png_bytes(), "image/png")}
    r = c.post("/v1/vision/parse", files=files, data={"model": "trocr-large"})
    assert r.status_code == 400
    assert "nicht unterstuetzt" in r.json()["detail"].lower() or "not supported" in r.json()["detail"].lower()


def test_ocr_auto_uses_tesseract_default():
    c = _client()
    files = {"image": ("test.png", _png_bytes(), "image/png")}
    r = c.post("/v1/ocr", files=files, data={"backend": "auto"})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "tesseract"
    assert body["text"].startswith("Rechnung")
    assert 0.0 <= body["confidence"] <= 1.0


def test_ocr_explicit_tesseract_with_lang():
    c = _client()
    files = {"image": ("test.png", _png_bytes(), "image/png")}
    r = c.post("/v1/ocr", files=files, data={"backend": "tesseract", "lang": "deu"})
    assert r.status_code == 200
    body = r.json()
    assert body["languages"] == "deu"


def test_ocr_easyocr_unavailable_503():
    c = _client()
    files = {"image": ("test.png", _png_bytes(), "image/png")}
    r = c.post("/v1/ocr", files=files, data={"backend": "easyocr"})
    assert r.status_code == 503


def test_ocr_unknown_backend_400():
    c = _client()
    files = {"image": ("test.png", _png_bytes(), "image/png")}
    r = c.post("/v1/ocr", files=files, data={"backend": "trolltastic"})
    assert r.status_code == 400


def test_image_too_large_413(monkeypatch):
    """Limit-Test via Config-Singleton-Override."""
    from vision_service import config as cfg_mod

    small = cfg_mod.Config(max_image_bytes=100)
    monkeypatch.setattr(cfg_mod, "_singleton", small)
    c = _client()
    big_png = _png_bytes(size=(512, 512))  # weit ueber 100 Bytes
    files = {"image": ("big.png", big_png, "image/png")}
    r = c.post("/v1/ocr", files=files, data={"backend": "tesseract"})
    assert r.status_code == 413


def test_broken_image_400():
    c = _client()
    files = {"image": ("nope.png", b"not-an-image", "image/png")}
    r = c.post("/v1/ocr", files=files, data={"backend": "tesseract"})
    assert r.status_code == 400


def test_root_endpoint_describes_service():
    c = _client()
    r = c.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "vision-service"
    assert "/v1/vision/parse" in body["endpoints"]
    assert "/v1/ocr" in body["endpoints"]


def test_easyocr_enabled_picks_in_models_list(monkeypatch):
    """Wenn EasyOCR ``available=True`` gemockt ist, taucht es in /v1/models auf."""
    from vision_service.models import easyocr as easy_mod

    easy_mod._singleton.enabled = True  # FakeEasy-Schalter
    c = _client()
    r = c.get("/v1/models")
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert "easyocr" in by_id
    assert by_id["easyocr"]["capabilities"] == ["ocr"]


def test_api_key_required(monkeypatch):
    """Wenn VISION_API_KEY gesetzt ist, muss der Header geprueft werden."""
    from vision_service import config as cfg_mod

    monkeypatch.setattr(
        cfg_mod, "_singleton", cfg_mod.Config(api_key="topsecret"),
    )
    c = _client()
    r = c.get("/v1/models")
    assert r.status_code == 401
    r2 = c.get("/v1/models", headers={"X-Api-Key": "topsecret"})
    assert r2.status_code == 200
    # /health bleibt **offen** (Liveness-Konvention)
    r3 = c.get("/health")
    assert r3.status_code == 200


def test_donut_fields_extraction_regex():
    """Direkter Test der heuristischen Tag-Extraktion."""
    from vision_service.models.donut import DonutBackend

    seq = (
        "<s_seller_name>ACME GmbH</s_seller_name>"
        "<s_total>123,45</s_total>"
        "<s_date>2026-05-11</s_date>"
    )
    fields = DonutBackend._extract_fields(seq)
    assert fields["total_amount"] == "123,45"
    assert fields["invoice_date"] == "2026-05-11"
    assert fields["supplier_name"] == "ACME GmbH"
