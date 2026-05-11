"""Konfiguration via Umgebungsvariablen.

Defaults zielen auf einen CPU-Lauf auf der NUC. Fuer evo-x2/Desktop einfach
``VISION_DEVICE=cuda`` setzen; transformers nimmt dann automatisch das
default-CUDA-Geraet. ``auto`` waehlt cuda wenn verfuegbar, sonst cpu.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class Config:
    # Server-Bind
    host: str = field(default_factory=lambda: os.environ.get("VISION_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("VISION_PORT", "8005")))

    # Modell-Cache + Device. ``auto`` → cuda wenn verfuegbar, sonst cpu.
    model_cache_dir: str = field(
        default_factory=lambda: os.environ.get("HF_HOME", "/data/hf_cache"),
    )
    device: str = field(default_factory=lambda: os.environ.get("VISION_DEVICE", "auto"))

    # Donut: Default-Modell fuer /v1/vision/parse
    donut_default_model: str = field(
        default_factory=lambda: os.environ.get(
            "VISION_DONUT_MODEL",
            "naver-clova-ix/donut-base-finetuned-cord-v2",
        ),
    )
    donut_task_prompt: str = field(
        default_factory=lambda: os.environ.get("VISION_DONUT_TASK_PROMPT", "<s_cord-v2>"),
    )
    enable_donut: bool = field(default_factory=lambda: _env_bool("VISION_ENABLE_DONUT", True))

    # Tesseract: Sprachen (kommagetrennt → "deu+eng" intern)
    tesseract_languages: list[str] = field(
        default_factory=lambda: _env_list("VISION_TESSERACT_LANGS", ["deu", "eng"]),
    )
    enable_tesseract: bool = field(
        default_factory=lambda: _env_bool("VISION_ENABLE_TESSERACT", True),
    )

    # EasyOCR: optional, schwer; nur wenn extra installed.
    easyocr_languages: list[str] = field(
        default_factory=lambda: _env_list("VISION_EASYOCR_LANGS", ["de", "en"]),
    )
    enable_easyocr: bool = field(
        default_factory=lambda: _env_bool("VISION_ENABLE_EASYOCR", False),
    )

    # Chandra: optional. Wenn `chandra` o.ae. Module verfuegbar, hier aktivieren.
    enable_chandra: bool = field(default_factory=lambda: _env_bool("VISION_ENABLE_CHANDRA", False))

    # OCR-Default-Backend bei ``backend=auto``.
    ocr_default_backend: str = field(
        default_factory=lambda: os.environ.get("VISION_OCR_DEFAULT_BACKEND", "tesseract"),
    )

    # Auth: optional. Wenn API_KEY gesetzt → ``X-Api-Key``-Header verlangt.
    api_key: str | None = field(default_factory=lambda: os.environ.get("VISION_API_KEY"))

    # Limits — Schutz vor Memory-Spikes.
    max_image_bytes: int = field(
        default_factory=lambda: int(os.environ.get("VISION_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
    )
    max_pdf_pages: int = field(
        default_factory=lambda: int(os.environ.get("VISION_MAX_PDF_PAGES", "50")),
    )
    max_pdf_bytes: int = field(
        default_factory=lambda: int(os.environ.get("VISION_MAX_PDF_BYTES", str(40 * 1024 * 1024))),
    )

    # Eager-Load: Donut beim Startup laden (~3s) oder Lazy beim ersten Request.
    eager_load: bool = field(default_factory=lambda: _env_bool("VISION_EAGER_LOAD", True))


_singleton: Config | None = None


def get_config() -> Config:
    global _singleton
    if _singleton is None:
        _singleton = Config()
    return _singleton


def resolve_device(cfg_device: str) -> str:
    """Resolves ``auto`` to ``cuda`` if available, else ``cpu``.

    Wir importieren torch lazy, weil Tests den Import gerne mocken und der
    Config-Modul-Import nicht torch hard requiren soll.
    """
    if cfg_device != "auto":
        return cfg_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
