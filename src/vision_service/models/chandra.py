"""Chandra OCR Backend (optional, Best-Effort).

Chandra OCR ist im flowinvoice-Stack als ``app.services.chandra_ocr`` gewrapped.
Hier nutzen wir es nur, wenn das Python-Modul ``chandra_ocr`` oder eine
kompatible Schnittstelle vorhanden ist. Andernfalls bleibt das Backend
``available=False`` und wird vom Service nicht beworben.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ..config import get_config

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

log = logging.getLogger(__name__)


class ChandraBackend:
    def __init__(self) -> None:
        self._mod: Any = None
        self._available_checked: bool | None = None

    def available(self) -> bool:
        cfg = get_config()
        if not cfg.enable_chandra:
            return False
        if self._available_checked is not None:
            return self._available_checked
        # Wir probieren bewusst mehrere Importnamen, da Chandra-Distribution
        # uneinheitlich ist. Wenn keiner klappt → not available.
        for name in ("chandra_ocr", "chandra"):
            try:
                self._mod = __import__(name)
                self._available_checked = True
                return True
            except Exception:  # noqa: BLE001
                continue
        self._available_checked = False
        return False

    def run(self, image: "Image", lang: str | None = None) -> dict:
        if not self.available():
            raise RuntimeError("Chandra-Backend ist nicht verfuegbar")
        t0 = time.monotonic()
        # Defensive Fallback-API: viele Wrapper bieten `read_text(image)`.
        text = ""
        confidence = 0.0
        if hasattr(self._mod, "read_text"):
            try:
                text = self._mod.read_text(image)
            except Exception as exc:  # noqa: BLE001
                log.warning("Chandra.read_text failed: %s", exc)
        else:
            raise RuntimeError(
                "Chandra-Modul vorhanden, aber keine kompatible read_text()-Funktion gefunden",
            )
        return {
            "backend": "chandra",
            "text": text or "",
            "confidence": float(confidence),
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "languages": lang or "",
        }


_singleton: ChandraBackend | None = None


def get_chandra() -> ChandraBackend:
    global _singleton
    if _singleton is None:
        _singleton = ChandraBackend()
    return _singleton
