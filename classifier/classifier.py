import json
import time
import re
import os
import hashlib
import logging
from typing import Optional

import anthropic
from anthropic import APIStatusError, APITimeoutError, APIConnectionError

from .prompts import SYSTEM_PROMPT, build_few_shot_messages, build_inference_message

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 120
TEMPERATURE = 0.0
TIMEOUT_SECS = 15.0
MAX_RETRIES = 2
RETRY_BACKOFF = [1.0, 2.5]

VALID_PHASES = {"A", "B", "C", "D", "E"}
VALID_ACTS = {
    "validacion", "pregunta_abierta", "pregunta_cerrada",
    "reflejo", "interpretacion", "directivo",
    "silencio_contencion", "confrontacion", "otro",
}


class ClassificationResult:
    def __init__(
        self,
        fase: str,
        actos_verbales: list[str],
        confianza: float,
        llm_available: bool = True,
        raw_response: str = "",
    ):
        self.fase = fase
        self.actos_verbales = actos_verbales
        self.confianza = confianza
        self.llm_available = llm_available
        self.raw_response = raw_response

    def to_dict(self) -> dict:
        return {
            "fase": self.fase,
            "actos_verbales": self.actos_verbales,
            "confianza": round(self.confianza, 3),
            "llm_available": self.llm_available,
        }


_cache: dict[str, ClassificationResult] = {}


def _cache_key(operator_text: str, patient_context: str) -> str:
    raw = f"{operator_text.strip()}|||{patient_context.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse_response(text: str) -> tuple[str, list[str], float]:
    """Parses LLM JSON response. Tolerant to markdown fences and surrounding text."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]+\}', cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {text[:200]}")
        data = json.loads(match.group())

    fase = str(data.get("fase", "")).strip().upper()
    if fase not in VALID_PHASES:
        raise ValueError(f"Invalid fase: {fase!r}")

    raw_acts = data.get("actos_verbales", ["otro"])
    if isinstance(raw_acts, str):
        raw_acts = [raw_acts]
    acts = [a.strip().lower() for a in raw_acts if isinstance(a, str)]
    acts = [a for a in acts if a in VALID_ACTS]
    if not acts:
        acts = ["otro"]

    try:
        confianza = float(data.get("confianza", 0.8))
        confianza = max(0.0, min(1.0, confianza))
    except (TypeError, ValueError):
        confianza = 0.8

    return fase, acts, confianza


class PAPClassifier:
    """Few-shot LLM classifier for PAP operator turns."""

    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=key) if key else None
        self.few_shot_messages = build_few_shot_messages()
        self.use_cache = use_cache

    def classify(
        self,
        operator_text: str,
        patient_context: str = "",
    ) -> ClassificationResult:
        if not operator_text or not operator_text.strip():
            return self._fallback("Empty operator text")

        if self.use_cache:
            key = _cache_key(operator_text, patient_context)
            if key in _cache:
                logger.debug("Cache hit for turn")
                return _cache[key]

        result = self._call_with_retries(operator_text, patient_context)

        if self.use_cache:
            _cache[_cache_key(operator_text, patient_context)] = result

        return result

    def classify_batch(
        self,
        turns: list[dict],
        verbose: bool = False,
    ) -> list[ClassificationResult]:
        """Classifies a list of turns. Each element must have 'operador_texto'
        and optionally 'contexto_paciente_previo'."""
        results = []
        total = len(turns)
        for i, turn in enumerate(turns):
            if verbose and i % 50 == 0:
                print(f"  Clasificando turno {i+1}/{total}...", flush=True)
            r = self.classify(
                turn.get("operador_texto", ""),
                turn.get("contexto_paciente_previo", ""),
            )
            results.append(r)
        return results

    def _call_with_retries(
        self, operator_text: str, patient_context: str
    ) -> ClassificationResult:
        if self.client is None:
            return self._fallback("No API key configured")

        messages = list(self.few_shot_messages)
        messages.append(build_inference_message(operator_text, patient_context))

        last_error = ""
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    timeout=TIMEOUT_SECS,
                )
                raw_text = response.content[0].text
                fase, acts, conf = _parse_response(raw_text)
                return ClassificationResult(
                    fase=fase,
                    actos_verbales=acts,
                    confianza=conf,
                    llm_available=True,
                    raw_response=raw_text,
                )

            except (APITimeoutError, APIConnectionError) as e:
                last_error = f"Network/timeout error: {e}"
                logger.warning(f"Attempt {attempt+1} failed: {last_error}")

            except APIStatusError as e:
                last_error = f"API status {e.status_code}: {e.message}"
                logger.warning(f"Attempt {attempt+1} failed: {last_error}")
                # Don't retry auth or quota errors
                if e.status_code in (401, 403, 429):
                    break

            except (ValueError, json.JSONDecodeError) as e:
                last_error = f"Parse error: {e}"
                logger.warning(f"Attempt {attempt+1} parse failed: {last_error}")

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.error(f"Attempt {attempt+1} unexpected: {last_error}")

            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)

        return self._fallback(last_error)

    def _fallback(self, reason: str) -> ClassificationResult:
        # Returns a safe default so the API returns 200 with llm_available=False
        # instead of 500 when the LLM is unavailable.
        logger.error(f"Classifier fallback: {reason}")
        return ClassificationResult(
            fase="A",
            actos_verbales=["otro"],
            confianza=0.0,
            llm_available=False,
            raw_response="",
        )
