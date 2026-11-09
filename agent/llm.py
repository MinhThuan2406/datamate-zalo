"""
LLM client for the harness: OpenRouter (OpenAI-compatible), pinned to one provider.

- Provider pinning: every request carries
  ``extra_body={"provider": {"only": ["primeintellect/bf16"], "allow_fallbacks": False}}``.
  A response whose ``provider`` field does not name the pinned provider raises
  ``ProviderMismatchError`` (never cached, never retried). A response with no
  ``provider`` field is accepted with a warning (the ``only`` routing still applies).
- Retries: up to ``max_attempts`` (default 5) with exponential backoff and jitter on
  429, 5xx, timeouts, connection errors and malformed/empty 200 responses. Other
  API errors (400/401/403/404...) fail at once with ``LLMError``.
- Disk cache: ``cache/llm/<sha256>.json``. The key is the sha256 of the canonical JSON of
  (cache version, model, provider routing, messages, sampling params, reasoning). A hit
  returns the stored response without a network call. Empty responses are not cached.
- Thread-safe: the OpenAI client is shared; counters are guarded by a lock; cache writes
  are atomic (temp file + os.replace).

Configuration comes from env vars (see .env.example); ``.env`` is loaded with python-dotenv
without overriding real environment variables. The API key is never stored in the config
object, never logged and never included in the cache key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import openai
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger("agent.llm")

CACHE_VERSION = 1
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.5-9b"
PINNED_PROVIDER = "primeintellect/bf16"
DEFAULT_MAX_TOKENS = 8192
MAX_TOKENS_LIMIT = 40960
REASONING_LEVELS = ("off", "low", "medium", "high")
DEFAULT_PRICE_PROMPT_PER_M = 0.033       # USD per 1M prompt tokens (PrimeIntellect bf16)
DEFAULT_PRICE_COMPLETION_PER_M = 0.048   # USD per 1M completion tokens (includes reasoning tokens)
API_KEY_ENV = "OPENROUTER_API_KEY"

SAMPLING_OVERRIDES = ("temperature", "top_p", "top_k", "max_tokens", "reasoning")

RETRYABLE_ERRORS: Tuple[type, ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.APIResponseValidationError,
)


class LLMError(RuntimeError):
    """A call failed for good: non-retryable error, retries exhausted, or no API key."""


class ProviderMismatchError(LLMError):
    """The response was served by a provider other than the pinned one."""


class TransientResponseError(Exception):
    """A 200 response that is unusable (no choices, provider error body, finish_reason=error)."""


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def reasoning_payload(level: Optional[str]) -> Optional[Dict[str, Any]]:
    """Maps LLM_REASONING to OpenRouter's ``reasoning`` request field. None = do not send.

    Measured on the PrimeIntellect endpoint: unset and ``{"enabled": false}`` give 0
    reasoning tokens; ``{"effort": "low"}`` turns thinking on (reasoning field returned).
    """
    if level is None:
        return None
    if level not in REASONING_LEVELS:
        raise ValueError(f"reasoning must be one of {REASONING_LEVELS} or unset, got {level!r}")
    if level == "off":
        return {"enabled": False}
    return {"effort": level}


def _env_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_str(env: Mapping[str, str], name: str) -> Optional[str]:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    provider: Optional[str] = PINNED_PROVIDER      # OpenRouter provider slug; None disables pinning
    allow_fallbacks: bool = False
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning: Optional[str] = None                # off | low | medium | high | None (not sent)
    timeout_s: float = 180.0
    max_attempts: int = 5
    backoff_base_s: float = 2.0
    backoff_cap_s: float = 60.0
    cache_enabled: bool = True
    cache_dir: Path = BASE_DIR / "cache" / "llm"
    price_prompt_per_m: float = DEFAULT_PRICE_PROMPT_PER_M
    price_completion_per_m: float = DEFAULT_PRICE_COMPLETION_PER_M

    def __post_init__(self) -> None:
        reasoning_payload(self.reasoning)  # validates
        if not (0 < self.max_tokens <= MAX_TOKENS_LIMIT):
            raise ValueError(f"max_tokens must be in 1..{MAX_TOKENS_LIMIT}, got {self.max_tokens}")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, *, cache_dir: Optional[Path] = None) -> "LLMConfig":
        if env is None:
            load_dotenv(BASE_DIR / ".env", override=False)
            env = os.environ
        provider = _env_str(env, "LLM_PROVIDER") or PINNED_PROVIDER
        allow_fallbacks = _env_bool(_env_str(env, "LLM_ALLOW_FALLBACKS"), False)
        if allow_fallbacks:
            logger.warning("LLM_ALLOW_FALLBACKS=true: requests may be served by another provider")
        top_p = _env_str(env, "LLM_TOP_P")
        top_k = _env_str(env, "LLM_TOP_K")
        reasoning = _env_str(env, "LLM_REASONING")
        kwargs: Dict[str, Any] = dict(
            model=_env_str(env, "LLM_MODEL") or DEFAULT_MODEL,
            base_url=_env_str(env, "LLM_BASE_URL") or DEFAULT_BASE_URL,
            provider=provider,
            allow_fallbacks=allow_fallbacks,
            temperature=float(_env_str(env, "LLM_TEMPERATURE") or 0.0),
            top_p=float(top_p) if top_p else None,
            top_k=int(top_k) if top_k else None,
            max_tokens=int(_env_str(env, "LLM_MAX_TOKENS") or DEFAULT_MAX_TOKENS),
            reasoning=reasoning.lower() if reasoning else None,
            timeout_s=float(_env_str(env, "LLM_TIMEOUT_S") or 180.0),
            max_attempts=int(_env_str(env, "LLM_MAX_ATTEMPTS") or 5),
            cache_enabled=_env_bool(_env_str(env, "LLM_CACHE"), True),
            price_prompt_per_m=float(_env_str(env, "LLM_PRICE_PROMPT_PER_M") or DEFAULT_PRICE_PROMPT_PER_M),
            price_completion_per_m=float(_env_str(env, "LLM_PRICE_COMPLETION_PER_M") or DEFAULT_PRICE_COMPLETION_PER_M),
        )
        if cache_dir is not None:
            kwargs["cache_dir"] = Path(cache_dir)
        return cls(**kwargs)


# -----------------------------------------------------------------------------
# Results, usage, stats
# -----------------------------------------------------------------------------


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0       # includes reasoning tokens
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    openrouter_cost_usd: float = 0.0  # OpenRouter's own cost field, for cross-checking only

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            openrouter_cost_usd=self.openrouter_cost_usd + other.openrouter_cost_usd,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "openrouter_cost_usd": self.openrouter_cost_usd,
        }


def parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        cached_prompt_tokens=int(prompt_details.get("cached_tokens") or 0),
        openrouter_cost_usd=float(raw.get("cost") or 0.0),
    )


def estimate_cost(usage: Usage, config: LLMConfig) -> float:
    return (usage.prompt_tokens * config.price_prompt_per_m
            + usage.completion_tokens * config.price_completion_per_m) / 1_000_000


@dataclass
class LLMResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    provider: Optional[str] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    reasoning_text: Optional[str] = None
    cached: bool = False
    latency_s: float = 0.0
    attempts: int = 1
    cache_key: str = ""
    cost_usd: float = 0.0            # estimated cost of this call (0 for cache hits)


class LLMStats:
    """Process-wide counters, safe to update from many threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.network_calls = 0
        self.retries = 0
        self.failures = 0
        self.usage = Usage()
        self.est_cost_usd = 0.0
        self.network_latency_s = 0.0

    def record_call(self) -> None:
        with self._lock:
            self.calls += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1

    def record_network(self, result: LLMResult) -> None:
        with self._lock:
            self.network_calls += 1
            self.usage = self.usage + result.usage
            self.est_cost_usd += result.cost_usd
            self.network_latency_s += result.latency_s

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "cache_hits": self.cache_hits,
                "network_calls": self.network_calls,
                "retries": self.retries,
                "failures": self.failures,
                "usage": self.usage.to_dict(),
                "est_cost_usd": self.est_cost_usd,
                "network_latency_s": self.network_latency_s,
            }

    def summary(self) -> str:
        s = self.snapshot()
        u = s["usage"]
        return (
            f"LLM calls: {s['calls']} (network {s['network_calls']}, cache hits {s['cache_hits']}), "
            f"retries {s['retries']}, failures {s['failures']}\n"
            f"LLM tokens: prompt {u['prompt_tokens']}, completion {u['completion_tokens']} "
            f"(reasoning {u['reasoning_tokens']}), cached prompt {u['cached_prompt_tokens']}\n"
            f"LLM est. cost: ${s['est_cost_usd']:.4f} (OpenRouter-reported ${u['openrouter_cost_usd']:.4f}); "
            f"summed network latency {s['network_latency_s']:.1f}s"
        )


# -----------------------------------------------------------------------------
# Cache key
# -----------------------------------------------------------------------------


def cache_key(payload: Dict[str, Any]) -> str:
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------

Transport = Callable[[Dict[str, Any]], Any]


class LLMClient:
    """``chat(messages, **overrides) -> LLMResult``. Overrides: temperature, top_p, top_k, max_tokens, reasoning."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_key: Optional[str] = None,
        transport: Optional[Transport] = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self.stats = LLMStats()
        self._api_key = api_key
        self._transport: Transport = transport or self._default_transport
        self._sleep = sleep
        self._jitter = jitter
        self._client: Optional[openai.OpenAI] = None
        self._client_lock = threading.Lock()
        self._warned_unverified = False

    def __repr__(self) -> str:  # never show the key
        c = self.config
        return f"LLMClient(model={c.model!r}, provider={c.provider!r}, cache={c.cache_enabled})"

    # -- public -------------------------------------------------------------

    def require_api_key(self) -> None:
        if not (self._api_key or os.environ.get(API_KEY_ENV)):
            raise LLMError(f"{API_KEY_ENV} is not set (put it in .env)")

    def chat(self, messages: list, **overrides: Any) -> LLMResult:
        kwargs, payload = self._build_request(messages, overrides)
        key = cache_key(payload)
        self.stats.record_call()

        if self.config.cache_enabled:
            cached = self._cache_read(key)
            if cached is not None:
                self.stats.record_cache_hit()
                return cached

        try:
            result = self._call_with_retries(kwargs)
            self._verify_provider(result)
        except LLMError:
            self.stats.record_failure()
            raise
        result.cache_key = key
        result.cost_usd = estimate_cost(result.usage, self.config)
        self.stats.record_network(result)
        if self.config.cache_enabled and result.text.strip():
            self._cache_write(key, result)
        return result

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    # -- request building ---------------------------------------------------

    def _build_request(self, messages: list, overrides: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        unknown = set(overrides) - set(SAMPLING_OVERRIDES)
        if unknown:
            raise TypeError(f"unsupported chat() parameters: {sorted(unknown)}")
        cfg = self.config
        temperature = overrides.get("temperature", cfg.temperature)
        top_p = overrides.get("top_p", cfg.top_p)
        top_k = overrides.get("top_k", cfg.top_k)
        max_tokens = overrides.get("max_tokens", cfg.max_tokens)
        reasoning = overrides.get("reasoning", cfg.reasoning)
        reasoning_body = reasoning_payload(reasoning)

        extra_body: Dict[str, Any] = {}
        if cfg.provider:
            extra_body["provider"] = {"only": [cfg.provider], "allow_fallbacks": cfg.allow_fallbacks}
        if top_k is not None:
            extra_body["top_k"] = top_k
        if reasoning_body is not None:
            extra_body["reasoning"] = reasoning_body

        kwargs: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p

        payload = {
            "v": CACHE_VERSION,
            "model": cfg.model,
            "provider": cfg.provider,
            "allow_fallbacks": cfg.allow_fallbacks,
            "messages": messages,
            "params": {"temperature": temperature, "top_p": top_p, "top_k": top_k,
                       "max_tokens": max_tokens, "reasoning": reasoning},
        }
        return kwargs, payload

    # -- network ------------------------------------------------------------

    def _get_client(self) -> openai.OpenAI:
        with self._client_lock:
            if self._client is None:
                key = self._api_key or os.environ.get(API_KEY_ENV)
                if not key:
                    raise LLMError(f"{API_KEY_ENV} is not set (put it in .env)")
                # max_retries=0: retries are handled here so that every attempt is counted and logged.
                self._client = openai.OpenAI(api_key=key, base_url=self.config.base_url,
                                             timeout=self.config.timeout_s, max_retries=0)
            return self._client

    def _default_transport(self, kwargs: Dict[str, Any]) -> Any:
        return self._get_client().chat.completions.create(**kwargs)

    def _call_with_retries(self, kwargs: Dict[str, Any]) -> LLMResult:
        cfg = self.config
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                raw = self._transport(kwargs)
                return self._parse(raw, latency_s=time.perf_counter() - t0, attempts=attempt)
            except RETRYABLE_ERRORS + (TransientResponseError,) as e:
                last: Exception = e
            except Exception as e:  # noqa: BLE001 - non-retryable: surface as LLMError
                raise LLMError(f"{type(e).__name__}: {e}") from e
            if attempt >= cfg.max_attempts:
                raise LLMError(
                    f"gave up after {attempt} attempts; last error {type(last).__name__}: {last}") from last
            delay = min(cfg.backoff_cap_s, cfg.backoff_base_s * (2 ** (attempt - 1)))
            delay *= 0.5 + self._jitter()  # jitter in [0.5x, 1.5x)
            self.stats.record_retry()
            logger.warning("LLM attempt %d failed (%s); retrying in %.1fs", attempt, type(last).__name__, delay)
            self._sleep(delay)

    def _parse(self, raw: Any, *, latency_s: float, attempts: int) -> LLMResult:
        data = raw.model_dump() if hasattr(raw, "model_dump") else raw
        if not isinstance(data, dict):
            raise TransientResponseError(f"unexpected response type {type(raw).__name__}")
        choices = data.get("choices") or []
        if not choices:
            detail = data.get("error") or list(data.keys())
            raise TransientResponseError(f"response without choices: {str(detail)[:300]}")
        choice = choices[0] or {}
        finish = choice.get("finish_reason")
        if finish == "error":
            raise TransientResponseError("provider finished with finish_reason=error")
        message = choice.get("message") or {}
        reasoning = message.get("reasoning")
        return LLMResult(
            text=message.get("content") or "",
            usage=parse_usage(data.get("usage")),
            provider=data.get("provider"),
            model=data.get("model"),
            finish_reason=finish,
            reasoning_text=reasoning if isinstance(reasoning, str) and reasoning else None,
            latency_s=latency_s,
            attempts=attempts,
        )

    def _verify_provider(self, result: LLMResult) -> None:
        wanted = self.config.provider
        if not wanted:
            return
        family = _norm_name(wanted.split("/")[0])
        if not result.provider:
            if not self._warned_unverified:
                self._warned_unverified = True
                logger.warning("response has no provider field; relying on provider routing alone")
            return
        if family not in _norm_name(result.provider):
            raise ProviderMismatchError(
                f"response was served by {result.provider!r}, expected {wanted!r}; refusing it")

    # -- cache --------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[LLMResult]:
        path = self._cache_path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("key") != key or not isinstance(data.get("text"), str):
            return None
        usage_raw = data.get("usage") or {}
        return LLMResult(
            text=data["text"],
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
                completion_tokens=int(usage_raw.get("completion_tokens", 0)),
                reasoning_tokens=int(usage_raw.get("reasoning_tokens", 0)),
                cached_prompt_tokens=int(usage_raw.get("cached_prompt_tokens", 0)),
                openrouter_cost_usd=float(usage_raw.get("openrouter_cost_usd", 0.0)),
            ),
            provider=data.get("provider"),
            model=data.get("model"),
            finish_reason=data.get("finish_reason"),
            reasoning_text=data.get("reasoning_text"),
            cached=True,
            latency_s=0.0,
            attempts=0,
            cache_key=key,
            cost_usd=0.0,
        )

    def _cache_write(self, key: str, result: LLMResult) -> None:
        path = self._cache_path(key)
        payload = {
            "key": key,
            "model": result.model,
            "provider": result.provider,
            "text": result.text,
            "reasoning_text": result.reasoning_text,
            "finish_reason": result.finish_reason,
            "usage": result.usage.to_dict(),
        }
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("could not write LLM cache entry: %s", e)
            try:
                tmp.unlink()
            except OSError:
                pass


__all__ = [
    "LLMClient", "LLMConfig", "LLMResult", "LLMError", "ProviderMismatchError", "Usage",
    "LLMStats", "cache_key", "estimate_cost", "parse_usage", "reasoning_payload", "PINNED_PROVIDER",
]
