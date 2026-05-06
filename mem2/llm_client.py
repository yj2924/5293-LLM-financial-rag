"""LLM client abstraction.

Default backend is a local Qwen 2.5 7B Instruct via HuggingFace transformers,
but the LLMClient protocol allows swapping to any API client (OpenAI, Anthropic,
vLLM, etc.) without touching pipeline code.

Public surface:
    LLMClient (Protocol)         -- the interface pipelines depend on
    QwenLocalClient              -- default local backend
    EchoClient                   -- testing / dry-run backend
    OpenAICompatibleClient       -- hits any /v1/chat/completions endpoint
                                    (real OpenAI, vLLM, Ollama, LM Studio)
    load_default_llm()           -- returns the configured default
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Interface every LLM backend must satisfy."""

    model_name: str

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Run inference. Return the generated string (only the assistant reply)."""
        ...

    def count_tokens(self, text: str) -> int:
        """Token count for `text` under this model's tokenizer."""
        ...


# ---------------------------------------------------------------------------
# Local Qwen backend
# ---------------------------------------------------------------------------


class QwenLocalClient:
    """Local Qwen 2.5 7B Instruct via transformers.

    Uses HF chat template; deterministic by default (temperature=0 -> greedy).
    Optional disk cache: identical (prompt, gen kwargs) returns cached answer
    without re-running inference. Cache survives restart, so dev iteration
    on prompts/pipelines doesn't re-spend GPU time.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        cache_dir: Optional[Path] = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"torch_dtype must be one of {list(dtype_map)}")

        self.model_name = model_name
        logger.info("Loading LLM: %s (dtype=%s, device_map=%s)", model_name, torch_dtype, device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_map[torch_dtype],
            device_map=device_map,
        )
        self.model.eval()

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("LLM cache enabled: %s", self.cache_dir)

    # -- cache helpers ------------------------------------------------------

    def _cache_key(self, prompt: str, **kwargs: Any) -> str:
        payload = json.dumps(
            {"model": self.model_name, "prompt": prompt, **kwargs},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.json"

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        gen_params = {"max_new_tokens": max_new_tokens, "temperature": temperature}

        if self.cache_dir is not None:
            key = self._cache_key(prompt, **gen_params)
            cache_file = self._cache_path(key)
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))["answer"]
                except (KeyError, json.JSONDecodeError):
                    logger.warning("Corrupt cache entry %s, regenerating", cache_file)

        # Build chat-format input. Qwen 2.5 expects the official chat template.
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        import torch

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Strip prompt tokens from the front of the output.
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if self.cache_dir is not None:
            self._cache_path(key).write_text(
                json.dumps({"answer": answer, "prompt": prompt, **gen_params}, ensure_ascii=False),
                encoding="utf-8",
            )

        return answer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Echo / dry-run backend (tests, plumbing checks without GPU)
# ---------------------------------------------------------------------------


class EchoClient:
    """Trivial backend that echoes the prompt. Useful for testing pipelines
    without paying GPU/API cost."""

    model_name = "echo"

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, **kwargs: Any) -> str:
        return f"[ECHO max_new_tokens={max_new_tokens} temp={temperature}]\n{prompt[-500:]}"

    def count_tokens(self, text: str) -> int:
        # Rough proxy: ~4 chars/token for English. Good enough for budget checks in tests.
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP backend (covers real OpenAI, vLLM, Ollama, LM Studio)
# ---------------------------------------------------------------------------


class OpenAICompatibleClient:
    """Minimal client for any /v1/chat/completions endpoint.

    Uses stdlib `urllib` so we do NOT take a dependency on the `openai` package.

    Configure via constructor args or env vars:
        MEM2_LLM_BASE_URL   e.g. https://api.openai.com/v1, http://localhost:8000/v1
        MEM2_LLM_API_KEY    optional bearer token
        MEM2_LLM_MODEL      override model name

    `count_tokens` falls back to a 4-chars-per-token heuristic since we cannot
    assume the API exposes a tokenizer. Good enough for budget checks; mem3
    should not rely on it for fine-grained token accounting.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        request_timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("LLM cache enabled: %s", self.cache_dir)

    # ---- cache helpers (same shape as QwenLocalClient) ----

    def _cache_key(self, prompt: str, **kwargs: Any) -> str:
        payload = json.dumps(
            {"model": self.model_name, "base_url": self.base_url, "prompt": prompt, **kwargs},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.json"

    # ---- public API ----

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        gen_params = {"max_new_tokens": max_new_tokens, "temperature": temperature}

        if self.cache_dir is not None:
            key = self._cache_key(prompt, **gen_params)
            cache_file = self._cache_path(key)
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))["answer"]
                except (KeyError, json.JSONDecodeError):
                    logger.warning("Corrupt cache entry %s, regenerating", cache_file)

        import urllib.request

        body = json.dumps(
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        try:
            answer = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response shape from {self.base_url}: {payload}") from e

        if self.cache_dir is not None:
            self._cache_path(key).write_text(
                json.dumps({"answer": answer, "prompt": prompt, **gen_params}, ensure_ascii=False),
                encoding="utf-8",
            )

        return answer

    def count_tokens(self, text: str) -> int:
        # Char-based proxy. Safe upper bound for English ~ len(text)/4.
        # Pipelines use this only for context budgeting; absolute token
        # accounting (cost, rate limits) should rely on the API response.
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_default_llm(
    backend: str = "qwen_local",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    cache_dir: Optional[Path] = Path("results/llm_cache"),
    **backend_kwargs: Any,
) -> LLMClient:
    """Build an LLMClient. `backend` switches implementations.

    Backends:
        qwen_local      local Qwen 2.5 via transformers (default).
        echo            no-op echo client; set `MEM2_LLM_BACKEND=echo` for CI / dry runs.
        openai_compat   any /v1/chat/completions endpoint. Reads
                        MEM2_LLM_BASE_URL (required), MEM2_LLM_API_KEY (optional),
                        MEM2_LLM_MODEL (overrides `model_name`).
    """
    backend = os.environ.get("MEM2_LLM_BACKEND", backend)

    if backend == "qwen_local":
        return QwenLocalClient(model_name=model_name, cache_dir=cache_dir, **backend_kwargs)
    if backend == "echo":
        return EchoClient()
    if backend == "openai_compat":
        base_url = os.environ.get("MEM2_LLM_BASE_URL")
        if not base_url:
            raise ValueError(
                "MEM2_LLM_BASE_URL is required for openai_compat backend, "
                "e.g. https://api.openai.com/v1 or http://localhost:8000/v1"
            )
        return OpenAICompatibleClient(
            model_name=os.environ.get("MEM2_LLM_MODEL", model_name),
            base_url=base_url,
            api_key=os.environ.get("MEM2_LLM_API_KEY"),
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown LLM backend: {backend!r}")
