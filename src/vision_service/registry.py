"""Backend-Registry: zentraler Zugriff auf alle Vision/OCR-Backends.

Wir wollen genau **eine** Stelle, an der Backends gesammelt werden — sowohl
fuer ``/health`` (Liveness) als auch fuer ``/v1/models`` (Capability-Discovery
durch llm-router) und die OCR-Auto-Routing-Logik (``backend=auto``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import get_config
from .models.chandra import get_chandra
from .models.donut import get_donut
from .models.easyocr import get_easyocr
from .models.tesseract import get_tesseract


@dataclass(frozen=True)
class BackendInfo:
    id: str
    available: bool
    capability: str  # "vision" | "ocr"
    description: str


def collect_backend_info() -> list[BackendInfo]:
    """Statische + dynamische Backend-Liste.

    Ein Backend gilt als ``available`` wenn sein ``available()``-Check ``True``
    liefert (Modul-Import-Probe). Bei Donut bedeutet ``available=True`` noch
    nicht, dass das Modell schon geladen ist; das passiert lazy oder im
    eager-load-Startup.
    """
    return [
        BackendInfo(
            id="donut-cord-v2",
            available=get_donut().available(),
            capability="vision",
            description="Donut Document-Understanding (naver-clova-ix/donut-base-finetuned-cord-v2)",
        ),
        BackendInfo(
            id="tesseract",
            available=get_tesseract().available(),
            capability="ocr",
            description="Tesseract 5.x via pytesseract",
        ),
        BackendInfo(
            id="easyocr",
            available=get_easyocr().available(),
            capability="ocr",
            description="JaidedAI EasyOCR (optional)",
        ),
        BackendInfo(
            id="chandra",
            available=get_chandra().available(),
            capability="ocr",
            description="Chandra OCR (optional)",
        ),
    ]


def pick_ocr_backend(requested: str) -> str:
    """Resolves ``backend=auto|tesseract|easyocr|chandra`` zu konkretem Backend.

    Reihenfolge fuer ``auto``: bevorzugt das in der Config gesetzte
    Default-Backend, faellt sonst auf das erste verfuegbare zurueck.
    """
    requested = (requested or "auto").lower()
    cfg = get_config()
    info = {b.id: b for b in collect_backend_info() if b.capability == "ocr"}

    if requested != "auto":
        if requested not in info:
            raise ValueError(f"Unbekanntes OCR-Backend: {requested}")
        if not info[requested].available:
            raise RuntimeError(f"OCR-Backend {requested} ist nicht verfuegbar")
        return requested

    # auto-Routing
    preferred = cfg.ocr_default_backend
    if preferred in info and info[preferred].available:
        return preferred
    for candidate in ("tesseract", "easyocr", "chandra"):
        if info[candidate].available:
            return candidate
    raise RuntimeError("Kein OCR-Backend verfuegbar")


def run_ocr(backend_id: str, image: Any, lang: str | None = None) -> dict:
    if backend_id == "tesseract":
        return get_tesseract().run(image, lang=lang)
    if backend_id == "easyocr":
        return get_easyocr().run(image, lang=lang)
    if backend_id == "chandra":
        return get_chandra().run(image, lang=lang)
    raise ValueError(f"Backend nicht OCR-faehig: {backend_id}")


def run_vision(model: str, image: Any) -> dict:
    """Aktuell nur Donut als Vision-Modell."""
    if model in ("donut", "donut-cord-v2") or model.startswith("naver-clova-ix/donut"):
        return get_donut().run(image, model_name=None if model.startswith("donut") else model)
    raise ValueError(f"Vision-Modell nicht unterstuetzt: {model}")
