"""Pytest-Setup: mockt schwere Modelle (Donut/EasyOCR/Chandra) und tesseract.

Wichtig: Diese Mocks werden vor dem Import von ``vision_service.main`` und
allen Modell-Submodulen aktiv, damit echte HF-Downloads in CI **nie**
passieren. Tests, die ein konkretes Backend brauchen, koennen die jeweilige
Singleton-Instanz danach noch tunen (siehe ``test_api.py``).
"""
from __future__ import annotations

import importlib

import pytest


def _reset_singletons() -> None:
    """Reset module-level singletons so each test fixture-config wirkt."""
    for modname in (
        "vision_service.models.donut",
        "vision_service.models.tesseract",
        "vision_service.models.easyocr",
        "vision_service.models.chandra",
        "vision_service.config",
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, "_singleton"):
            setattr(mod, "_singleton", None)


@pytest.fixture(autouse=True)
def _patch_backends(monkeypatch):
    _reset_singletons()
    # Donut: available()=True, run() liefert deterministisches Mock-Ergebnis
    from vision_service.models import donut as donut_mod

    class FakeDonut:
        def available(self):
            return True

        def warmup(self, *_a, **_kw):
            return None

        def loaded_models(self):
            return ["naver-clova-ix/donut-base-finetuned-cord-v2"]

        def device(self):
            return "cpu"

        def run(self, image, model_name=None):
            return {
                "model": model_name or "naver-clova-ix/donut-base-finetuned-cord-v2",
                "raw": "<s_total>42.00</s_total><s_date>2026-05-11</s_date>",
                "fields": {"total_amount": "42.00", "invoice_date": "2026-05-11"},
                "json": {"total": "42.00", "date": "2026-05-11"},
                "duration_ms": 7,
                "device": "cpu",
            }

    monkeypatch.setattr(donut_mod, "_singleton", FakeDonut(), raising=False)
    monkeypatch.setattr(donut_mod, "get_donut", lambda: donut_mod._singleton)

    # Tesseract: available()=True, run() liefert konstanten Text
    from vision_service.models import tesseract as tess_mod

    class FakeTesseract:
        def available(self):
            return True

        def languages(self):
            return "deu+eng"

        def run(self, image, lang=None):
            return {
                "backend": "tesseract",
                "text": "Rechnung Nr. 4711",
                "confidence": 0.91,
                "duration_ms": 3,
                "languages": lang or "deu+eng",
            }

    monkeypatch.setattr(tess_mod, "_singleton", FakeTesseract(), raising=False)
    monkeypatch.setattr(tess_mod, "get_tesseract", lambda: tess_mod._singleton)

    # EasyOCR: standardmaessig deaktiviert (matched Default-Config)
    from vision_service.models import easyocr as easy_mod

    class FakeEasy:
        def __init__(self, enabled: bool = False):
            self.enabled = enabled

        def available(self):
            return self.enabled

        def run(self, image, lang=None):
            return {
                "backend": "easyocr",
                "text": "EasyMockText",
                "confidence": 0.88,
                "duration_ms": 5,
                "languages": ["de", "en"],
            }

    monkeypatch.setattr(easy_mod, "_singleton", FakeEasy(), raising=False)
    monkeypatch.setattr(easy_mod, "get_easyocr", lambda: easy_mod._singleton)

    # Chandra: standardmaessig deaktiviert
    from vision_service.models import chandra as chan_mod

    class FakeChandra:
        def __init__(self, enabled: bool = False):
            self.enabled = enabled

        def available(self):
            return self.enabled

        def run(self, image, lang=None):
            return {
                "backend": "chandra",
                "text": "ChandraMockText",
                "confidence": 0.85,
                "duration_ms": 9,
                "languages": lang or "",
            }

    monkeypatch.setattr(chan_mod, "_singleton", FakeChandra(), raising=False)
    monkeypatch.setattr(chan_mod, "get_chandra", lambda: chan_mod._singleton)
    yield
    _reset_singletons()
