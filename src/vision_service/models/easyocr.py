"""EasyOCR Backend (optional).

EasyOCR zieht ~1 GB Detector/Recognizer-Modelle bei der ersten Initialisierung
und braucht torch + opencv. Deshalb ist es als optionales Extra
(``pip install vision-service[easyocr]``) installiert und per ENV abschaltbar.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from ..config import get_config, resolve_device

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

log = logging.getLogger(__name__)


class EasyOCRBackend:
    """Deep-Learning-OCR via JaidedAI EasyOCR."""

    def __init__(self) -> None:
        self._reader: Any = None
        self._lock = threading.Lock()
        self._available_checked: bool | None = None

    # ------------------------------------------------------------------
    def available(self) -> bool:
        cfg = get_config()
        if not cfg.enable_easyocr:
            return False
        if self._available_checked is not None:
            return self._available_checked
        try:
            import easyocr  # noqa: F401

            self._available_checked = True
        except Exception as exc:  # noqa: BLE001
            log.info("EasyOCR nicht installiert: %s", exc)
            self._available_checked = False
        return self._available_checked

    # ------------------------------------------------------------------
    def _reader_for(self, lang: list[str] | None = None) -> Any:
        if self._reader is not None:
            return self._reader
        with self._lock:
            if self._reader is not None:
                return self._reader
            import easyocr

            cfg = get_config()
            langs = lang or cfg.easyocr_languages
            device = resolve_device(cfg.device)
            log.info("Lade EasyOCR (lang=%s, device=%s)…", langs, device)
            self._reader = easyocr.Reader(
                langs,
                gpu=device.startswith("cuda"),
                verbose=False,
            )
            return self._reader

    # ------------------------------------------------------------------
    def run(self, image: "Image", lang: str | None = None) -> dict:
        if not self.available():
            raise RuntimeError("EasyOCR ist nicht installiert oder deaktiviert")
        import numpy as np

        # Lang-String "de,en" → ["de", "en"]
        langs: list[str] | None = None
        if lang:
            langs = [s.strip() for s in lang.split(",") if s.strip()]
        reader = self._reader_for(langs)
        arr = np.asarray(image.convert("RGB"))
        t0 = time.monotonic()
        result = reader.readtext(arr, detail=1, paragraph=False)
        # result: list[(bbox, text, conf)]
        texts: list[str] = []
        confs: list[float] = []
        for _bbox, text, conf in result:
            if isinstance(text, str) and text.strip():
                texts.append(text)
            try:
                confs.append(float(conf))
            except (TypeError, ValueError):
                pass
        avg = (sum(confs) / len(confs)) if confs else 0.0
        return {
            "backend": "easyocr",
            "text": " ".join(texts),
            "confidence": round(avg, 4),
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "languages": langs or get_config().easyocr_languages,
        }


_singleton: EasyOCRBackend | None = None


def get_easyocr() -> EasyOCRBackend:
    global _singleton
    if _singleton is None:
        _singleton = EasyOCRBackend()
    return _singleton
