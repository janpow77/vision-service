"""FastAPI-App fuer vision-service.

Endpunkte (OpenAI-kompatibel mit ``/v1/``-Prefix, damit llm-router proxien kann):

  POST /v1/vision/parse  — multipart-image + form ``model`` → strukturierte JSON
  POST /v1/ocr           — multipart-image + form ``backend`` → Text + Confidence
  GET  /health           — Liveness + verfuegbare Backends + GPU-Status
  GET  /v1/models        — OpenAI-Style Modell-Liste mit ``capabilities``

Die Service-Konventionen (``/v1``-Prefix, ``GET /health`` liefert
``{"status":"ok"}``, ``/v1/models`` mit ``capabilities``) sind so abgestimmt,
dass der llm-router den Service als Spoke registrieren kann.
"""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from . import __version__
from .config import get_config, resolve_device
from .registry import collect_backend_info, pick_ocr_backend, run_ocr, run_vision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger("vision-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    device = resolve_device(cfg.device)
    log.info(
        "vision-service %s — host=%s:%d device=%s eager_load=%s",
        __version__, cfg.host, cfg.port, device, cfg.eager_load,
    )
    if cfg.eager_load and cfg.enable_donut:
        # Donut wird beim Startup vorgeladen, damit der erste Request schnell
        # ist. Bei deaktiviertem Donut (z. B. ENV) wird der Block uebersprungen.
        try:
            from .models.donut import get_donut

            get_donut().warmup()
        except Exception as exc:  # noqa: BLE001
            log.error("Donut-Warmup fehlgeschlagen: %s", exc)
    yield
    log.info("vision-service stoppt.")


app = FastAPI(
    title="vision-service",
    version=__version__,
    lifespan=lifespan,
    description="Vision/OCR-Spoke fuer llm-router (Donut + Tesseract + EasyOCR + optional Chandra)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers


def _require_api_key(request: Request) -> None:
    cfg = get_config()
    if not cfg.api_key:
        return
    provided = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if not provided or provided != cfg.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")


def _load_image(file: UploadFile, raw: bytes) -> Image.Image:
    cfg = get_config()
    if len(raw) > cfg.max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image too large ({len(raw)} > {cfg.max_image_bytes} bytes)",
        )
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode → wirft hier bei kaputten Bildern
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"unsupported image: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc
    return img


# ---------------------------------------------------------------------------
# Endpoints


@app.get("/health")
async def health() -> dict:
    cfg = get_config()
    device = resolve_device(cfg.device)
    backends = collect_backend_info()
    gpu_available = False
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu_available = False
    return {
        "status": "ok",
        "version": __version__,
        "device": device,
        "gpu_available": gpu_available,
        "backends": [
            {
                "id": b.id,
                "available": b.available,
                "capability": b.capability,
            }
            for b in backends
        ],
        "default_donut_model": cfg.donut_default_model,
        "default_ocr_backend": cfg.ocr_default_backend,
    }


@app.get("/v1/models")
async def list_models(_=Depends(_require_api_key)) -> dict:
    """OpenAI-kompatible Modell-Liste fuer llm-router Spoke-Discovery."""
    data: list[dict] = []
    for b in collect_backend_info():
        if not b.available:
            continue
        data.append(
            {
                "id": b.id,
                "object": "model",
                "owned_by": "vision-service",
                "capabilities": [b.capability],
                "description": b.description,
            },
        )
    return {"object": "list", "data": data}


@app.post("/v1/vision/parse")
async def vision_parse(
    image: Annotated[UploadFile, File(description="JPEG/PNG/TIFF/BMP/WEBP image")],
    model: Annotated[str, Form(description="z.B. donut-cord-v2")] = "donut-cord-v2",
    _: Annotated[None, Depends(_require_api_key)] = None,
) -> dict:
    """Strukturiertes Vision-Parsing (Donut).

    Antwort:
      ``{ "model": "...", "raw": "<s_total>…", "fields": {...},
          "json": {...}|null, "duration_ms": int, "device": "cuda|cpu" }``
    """
    raw = await image.read()
    pil = _load_image(image, raw)
    try:
        result = run_vision(model, pil)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Vision parse failed")
        raise HTTPException(status_code=500, detail=f"vision parse failed: {exc}") from exc
    return result


@app.post("/v1/ocr")
async def ocr(
    image: Annotated[UploadFile, File(description="Bild")],
    backend: Annotated[
        str,
        Form(description="auto | tesseract | easyocr | chandra"),
    ] = "auto",
    lang: Annotated[
        str | None,
        Form(description="ISO/Tesseract-Sprachcodes; tesseract: 'deu+eng', easyocr: 'de,en'"),
    ] = None,
    _: Annotated[None, Depends(_require_api_key)] = None,
) -> dict:
    """Klassisches OCR (Tesseract/EasyOCR/Chandra).

    Antwort: ``{ "backend": "...", "text": "...", "confidence": 0.93,
              "duration_ms": int, "languages": "..." }``
    """
    raw = await image.read()
    pil = _load_image(image, raw)
    try:
        backend_id = pick_ocr_backend(backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        return run_ocr(backend_id, pil, lang=lang)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("OCR failed (%s)", backend_id)
        raise HTTPException(status_code=500, detail=f"ocr failed: {exc}") from exc


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "vision-service",
        "version": __version__,
        "endpoints": ["/health", "/v1/models", "/v1/vision/parse", "/v1/ocr"],
    }
