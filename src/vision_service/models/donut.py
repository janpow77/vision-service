"""Donut (Document Understanding Transformer) Backend.

Donut ist ein OCR-freier Vision-Encoder-Decoder, der direkt strukturierte
Ausgaben (CORD-v2 XML-aehnlich) produziert. Default-Modell:
``naver-clova-ix/donut-base-finetuned-cord-v2`` (~800 MB VRAM).

Diese Klasse kapselt Lazy-Load + Threadsafety. Der grosse Modell-Download
passiert beim ersten Lauf oder im Docker-Build (siehe ``Dockerfile``).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from ..config import get_config, resolve_device

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

log = logging.getLogger(__name__)


class DonutBackend:
    """Hochaufgeloester Document-Parser via Donut.

    ``run(image, model_name=None)`` → dict mit:
      - ``raw``: Donut-Sequenz vor Tag-Parsing
      - ``fields``: Heuristisch gemappte Felder (total/date/...)
      - ``json``: Donut-natives ``token2json``-Ergebnis (falls verfuegbar)
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.Lock()
        self._device: str | None = None
        self._available_checked: bool | None = None

    # ------------------------------------------------------------------
    def available(self) -> bool:
        cfg = get_config()
        if not cfg.enable_donut:
            return False
        if self._available_checked is not None:
            return self._available_checked
        try:
            import torch  # noqa: F401
            from transformers import DonutProcessor  # noqa: F401
            from transformers import VisionEncoderDecoderModel  # noqa: F401

            self._available_checked = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Donut nicht verfuegbar: %s", exc)
            self._available_checked = False
        return self._available_checked

    # ------------------------------------------------------------------
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(get_config().device)
        return self._device

    # ------------------------------------------------------------------
    def loaded_models(self) -> list[str]:
        return list(self._cache.keys())

    # ------------------------------------------------------------------
    def warmup(self, model_name: str | None = None) -> None:
        """Eager-Load Hook fuer das Lifespan-Startup."""
        cfg = get_config()
        name = model_name or cfg.donut_default_model
        if not self.available():
            return
        self._load(name)

    # ------------------------------------------------------------------
    def _load(self, model_name: str) -> tuple[Any, Any]:
        if model_name in self._cache:
            return self._cache[model_name]
        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            cfg = get_config()
            device = self.device()
            t0 = time.monotonic()
            log.info(
                "Lade Donut-Modell %s (device=%s, HF_HOME=%s)…",
                model_name, device, cfg.model_cache_dir,
            )
            processor = DonutProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(
                model_name, torch_dtype=torch.float32,
            )
            model.to(device)
            model.eval()
            self._cache[model_name] = (processor, model)
            log.info("Donut %s in %.1fs geladen", model_name, time.monotonic() - t0)
            return processor, model

    # ------------------------------------------------------------------
    def run(self, image: "Image", model_name: str | None = None) -> dict:
        if not self.available():
            raise RuntimeError("Donut-Backend ist nicht verfuegbar (transformers/torch fehlen)")
        cfg = get_config()
        name = model_name or cfg.donut_default_model
        processor, model = self._load(name)

        import torch  # local import — model is already loaded

        device = self.device()
        t0 = time.monotonic()

        pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        task_prompt = cfg.donut_task_prompt
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=getattr(model.decoder.config, "max_position_embeddings", 768),
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = (
            sequence.replace(processor.tokenizer.eos_token, "")
            .replace(processor.tokenizer.pad_token, "")
        )
        sequence = sequence.replace(task_prompt, "", 1).strip()

        # token2json ist die offizielle Donut-Routine; bei aelteren transformers
        # Versionen kann sie fehlen → robust fallback.
        token_json: Any = None
        try:
            token_json = processor.token2json(sequence)
        except Exception as exc:  # noqa: BLE001
            log.debug("token2json fehlgeschlagen, fallback auf regex: %s", exc)

        fields = self._extract_fields(sequence)
        duration_ms = int((time.monotonic() - t0) * 1000)

        return {
            "model": name,
            "raw": sequence,
            "fields": fields,
            "json": token_json,
            "duration_ms": duration_ms,
            "device": device,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_fields(sequence: str) -> dict[str, str]:
        """Heuristisches Mapping CORD-v2-Tags → Rechnungsfelder.

        Bewusst defensiv: Wenn nichts passt, leeres Dict. Konsument
        (flowinvoice) hat seine eigenen, strengeren Validatoren.
        """
        patterns: dict[str, list[str]] = {
            "total_amount": [
                r"<s_total[^>]*>\s*([\d.,]+)",
                r"<s_total_price[^>]*>\s*([\d.,]+)",
            ],
            "subtotal_amount": [r"<s_sub_?total[^>]*>\s*([\d.,]+)"],
            "tax_amount": [r"<s_(?:tax|vat)[^>]*>\s*([\d.,]+)"],
            "invoice_number": [
                r"<s_(?:invoice_)?num[^>]*>\s*([^<]+)",
            ],
            "invoice_date": [r"<s_date[^>]*>\s*([^<]+)"],
            "supplier_name": [r"<s_(?:seller|supplier)_name[^>]*>\s*([^<]+)"],
            "customer_name": [r"<s_(?:buyer|customer)_name[^>]*>\s*([^<]+)"],
        }
        result: dict[str, str] = {}
        for key, regs in patterns.items():
            for pat in regs:
                m = re.search(pat, sequence, re.IGNORECASE)
                if m:
                    result[key] = m.group(1).strip()
                    break
        return result


_singleton: DonutBackend | None = None
_singleton_lock = threading.Lock()


def get_donut() -> DonutBackend:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = DonutBackend()
    return _singleton
