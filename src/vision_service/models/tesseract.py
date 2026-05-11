"""Tesseract-OCR Backend.

Wrapt ``pytesseract.image_to_data`` und liefert Text + mittlere Konfidenz.
Tesseract muss als System-Binary verfuegbar sein (im Dockerfile via apt).
"""
from __future__ import annotations

import logging
import shutil
import time
from typing import TYPE_CHECKING

from ..config import get_config

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

log = logging.getLogger(__name__)


class TesseractBackend:
    """Klassische OCR via libtesseract.

    Konfidenz wird aus der ``conf``-Spalte von ``image_to_data`` aggregiert
    (Mittelwert ueber alle Worte mit conf >= 0).
    """

    def __init__(self) -> None:
        self._available_checked: bool | None = None

    # ------------------------------------------------------------------
    def available(self) -> bool:
        cfg = get_config()
        if not cfg.enable_tesseract:
            return False
        if self._available_checked is not None:
            return self._available_checked
        try:
            import pytesseract  # noqa: F401

            self._available_checked = shutil.which("tesseract") is not None
            if not self._available_checked:
                log.warning("tesseract-Binary nicht im PATH gefunden")
        except Exception as exc:  # noqa: BLE001
            log.warning("pytesseract nicht verfuegbar: %s", exc)
            self._available_checked = False
        return self._available_checked

    # ------------------------------------------------------------------
    def languages(self) -> str:
        return "+".join(get_config().tesseract_languages) or "eng"

    # ------------------------------------------------------------------
    def run(self, image: "Image", lang: str | None = None) -> dict:
        if not self.available():
            raise RuntimeError("Tesseract ist nicht verfuegbar")
        import pytesseract
        from pytesseract import Output

        languages = lang or self.languages()
        t0 = time.monotonic()
        data = pytesseract.image_to_data(image, lang=languages, output_type=Output.DICT)
        words = data.get("text", [])
        confs_raw = data.get("conf", [])

        # tesseract conf ist int oder string; -1 = ungueltig
        valid_confs: list[float] = []
        for c in confs_raw:
            try:
                cf = float(c)
            except (TypeError, ValueError):
                continue
            if cf >= 0:
                valid_confs.append(cf)
        avg_conf = (sum(valid_confs) / len(valid_confs) / 100.0) if valid_confs else 0.0
        text = " ".join(w for w in words if isinstance(w, str) and w.strip())
        duration_ms = int((time.monotonic() - t0) * 1000)
        return {
            "backend": "tesseract",
            "text": text,
            "confidence": round(avg_conf, 4),
            "duration_ms": duration_ms,
            "languages": languages,
        }


_singleton: TesseractBackend | None = None


def get_tesseract() -> TesseractBackend:
    global _singleton
    if _singleton is None:
        _singleton = TesseractBackend()
    return _singleton
