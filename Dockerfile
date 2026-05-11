# vision-service — Vision/OCR-Spoke fuer den llm-router.
#
# Image-Strategie:
# - Multi-Stage, am Ende nur Runtime + Wheels + vorgeladenes HF-Donut-Modell
# - CPU-only Torch (~250 MB statt ~2 GB GPU-Variante). GPU-Builds setzen
#   den Image-Tag ueber compose.gpu.yaml und installieren CUDA-Layer ueber
#   ein eigenes Basis-Image (out-of-scope fuer dieses Repo).
# - Donut-Modell wird im Build-Step nach /opt/hf_cache geladen, damit der
#   Container beim ersten Start keinen HuggingFace-Download macht (~800 MB).
# - Tesseract, poppler (pdf2image) und libgl1/libglib2 (Pillow + easyocr opencv)
#   sind im Runtime-Image installiert.

FROM python:3.11-slim AS deps

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU-only Torch (~250 MB). Bei GPU-Build override via build-arg.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX_URL} torch==2.3.1+cpu torchvision==0.18.1+cpu \
    || pip install torch==2.3.1 torchvision==0.18.1

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install .

# Donut-Modell vorladen — erspart den Cold-Start-Download.
ARG PRELOAD_DONUT=naver-clova-ix/donut-base-finetuned-cord-v2
ENV HF_HOME=/opt/hf_cache
RUN python -c "from transformers import DonutProcessor, VisionEncoderDecoderModel; \
    DonutProcessor.from_pretrained('${PRELOAD_DONUT}'); \
    VisionEncoderDecoderModel.from_pretrained('${PRELOAD_DONUT}'); \
    print('preload ok:', '${PRELOAD_DONUT}')"


# --------- Runtime ---------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/hf_cache \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_DISABLE_TELEMETRY=1

# System-Pakete fuer OCR + Bildverarbeitung
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tesseract-ocr \
        tesseract-ocr-deu \
        tesseract-ocr-eng \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 vision

# Site-Packages aus dem deps-Stage uebernehmen.
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Vorgeladenes Modell ins baked-Cache-Verzeichnis kopieren. Bind-Mount auf
# /data/hf_cache ueberschreibt das im Bedarfsfall.
COPY --from=deps /opt/hf_cache /opt/hf_cache_baked

COPY src/ /app/src/
ENV PYTHONPATH=/app/src

RUN mkdir -p /data/hf_cache /opt/vision && chown -R vision:vision /data /opt/vision

USER vision
WORKDIR /opt/vision

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=120s \
    CMD curl -fsS http://127.0.0.1:8005/health || exit 1

# Beim Start: HF-Cache aus baked-Image kopieren, falls leer (idempotent).
ENTRYPOINT ["/bin/bash", "-c", "if [ -z \"$(ls -A /data/hf_cache 2>/dev/null)\" ] && [ -d /opt/hf_cache_baked ]; then cp -r /opt/hf_cache_baked/. /data/hf_cache/; fi; exec uvicorn vision_service.main:app --host 0.0.0.0 --port ${VISION_PORT:-8005}"]
