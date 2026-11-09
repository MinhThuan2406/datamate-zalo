"""Tests for agent.llm: provider pinning, retries, cache, usage accounting, config. No network."""

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import httpx2
import openai

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import (  # noqa: E402
    LLMClient,
    LLMConfig,
    LLMError,
    ProviderMismatchError,
    Usage,
    cache_key,
    estimate_cost,
    parse_usage,
    reasoning_payload,
)

REQ = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Doanh thu hôm nay?"}]


def fake_response(text="ok", provider="PrimeIntellect", reasoning=None, usage=None, choices=None):
    if choices is not None:
        return {"choices": choices, "usage": None, "provider": provider}
    return {
        "id": "gen-1",
        "model": "qwen/qwen3.5-9b",
        "provider": provider,
        "choices": [{"finish_reason": "stop", "index": 0,
                     "message": {"role": "assistant", "content": text, "reasoning": reasoning}}],
        "usage": usage if usage is not None else {
            "prompt_tokens": 100, "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 5},
            "prompt_tokens_details": {"cached_tokens": 0}, "cost": 0.000002,
        },
    }


def rate_limit():
    return openai.RateLimitError("rate limited", response=httpx2.Response(429, request=REQ), body=None)


def server_error():
    return openai.InternalServerError("boom", response=httpx2.Response(500, request=REQ), body=None)


def bad_request():
    return openai.BadRequestError("bad", response=httpx2.Response(400, request=REQ), body=None)


def timeout_error():
    return openai.APITimeoutError(request=REQ)


class ScriptedTransport:
    """Returns queued outcomes (responses or exceptions); records the request kwargs."""

    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, kwargs):
        self.calls.append(kwargs)
        item = self.outcomes.pop(0) if self.outcomes else fake_response()
        if isinstance(item, BaseException):
            raise item
        return item


class LLMTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmp.name) / "llm"
        self.sleeps = []

    def tearDown(self):
        self._tmp.cleanup()

    def config(self, **kw):
        base = dict(cache_dir=self.cache_dir, backoff_base_s=1.0, backoff_cap_s=8.0, max_attempts=5)
        base.update(kw)
        return LLMConfig(**base)

    def client(self, transport, **cfg):
        return LLMClient(self.config(**cfg), transport=transport,
                         sleep=self.sleeps.append, jitter=lambda: 0.5)  # jitter 0.5 -> delay x1.0


class TestCacheKey(LLMTestCase):
    def test_key_is_stable_and_order_independent(self):
        a = {"model": "m", "messages": MESSAGES, "params": {"temperature": 0.0, "max_tokens": 8192}}
        b = {"params": {"max_tokens": 8192, "temperature": 0.0}, "messages": MESSAGES, "model": "m"}
        self.assertEqual(cache_key(a), cache_key(b))
        self.assertEqual(len(cache_key(a)), 64)

    def test_key_changes_with_any_semantic_input(self):
        base = {"model": "m", "provider": "primeintellect/bf16", "messages": MESSAGES,
                "params": {"temperature": 0.0, "reasoning": None}}
        keys = {cache_key(base)}
        keys.add(cache_key({**base, "provider": "novita"}))
        keys.add(cache_key({**base, "messages": MESSAGES + [{"role": "assistant", "content": "x"}]}))
        keys.add(cache_key({**base, "params": {"temperature": 0.7, "reasoning": None}}))
        keys.add(cache_key({**base, "params": {"temperature": 0.0, "reasoning": "low"}}))
        self.assertEqual(len(keys), 5)

    def test_key_uses_unicode_text_not_escapes(self):
        # Two different Vietnamese strings must not collide after serialisation.
        self.assertNotEqual(cache_key({"m": "Hôm"}), cache_key({"m": "Hom"}))


class TestRequestShape(LLMTestCase):
    def test_provider_pin_is_sent_and_reasoning_omitted_by_default(self):
        t = ScriptedTransport()
        self.client(t).chat(MESSAGES)
        kw = t.calls[0]
        self.assertEqual(kw["extra_body"], {"provider": {"only": ["primeintellect/bf16"], "allow_fallbacks": False}})
        self.assertNotIn("reasoning", kw["extra_body"])
        self.assertEqual(kw["temperature"], 0.0)
        self.assertNotIn("seed", kw)

    def test_reasoning_mapping(self):
        self.assertIsNone(reasoning_payload(None))
        self.assertEqual(reasoning_payload("off"), {"enabled": False})
        self.assertEqual(reasoning_payload("low"), {"effort": "low"})
        self.assertEqual(reasoning_payload("high"), {"effort": "high"})
        with self.assertRaises(ValueError):
            reasoning_payload("maximum")

    def test_reasoning_override_reaches_request_body(self):
        t = ScriptedTransport()
        self.client(t).chat(MESSAGES, reasoning="low")
        self.assertEqual(t.calls[0]["extra_body"]["reasoning"], {"effort": "low"})

    def test_unknown_override_rejected(self):
        with self.assertRaises(TypeError):
            self.client(ScriptedTransport()).chat(MESSAGES, seed=1)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            LLMConfig(max_tokens=40961)
        with self.assertRaises(ValueError):
            LLMConfig(reasoning="huge")


class TestRetries(LLMTestCase):
    def test_retries_rate_limit_and_5xx_then_succeeds(self):
        t = ScriptedTransport([rate_limit(), server_error(), fake_response("42")])
        res = self.client(t).chat(MESSAGES)
        self.assertEqual(res.text, "42")
        self.assertEqual(len(t.calls), 3)
        self.assertEqual(self.sleeps, [1.0, 2.0])  # exponential, jitter fixed at x1.0
        self.assertEqual(res.attempts, 3)

    def test_timeout_is_retried(self):
        t = ScriptedTransport([timeout_error(), fake_response("ok")])
        self.assertEqual(self.client(t).chat(MESSAGES).text, "ok")
        self.assertEqual(len(t.calls), 2)

    def test_gives_up_after_max_attempts(self):
        client = self.client(ScriptedTransport([server_error()] * 5))
        with self.assertRaises(LLMError) as ctx:
            client.chat(MESSAGES)
        self.assertIn("gave up after 5 attempts", str(ctx.exception))
        self.assertEqual(len(self.sleeps), 4)
        self.assertEqual(client.stats.snapshot()["failures"], 1)

    def test_non_retryable_status_fails_immediately(self):
        t = ScriptedTransport([bad_request(), fake_response("never")])
        with self.assertRaises(LLMError):
            self.client(t).chat(MESSAGES)
        self.assertEqual(len(t.calls), 1)
        self.assertEqual(self.sleeps, [])

    def test_empty_200_without_choices_is_retried(self):
        t = ScriptedTransport([fake_response(choices=[]), fake_response("ok")])
        self.assertEqual(self.client(t).chat(MESSAGES).text, "ok")
        self.assertEqual(len(t.calls), 2)

    def test_backoff_is_capped(self):
        t = ScriptedTransport([server_error()] * 5)
        client = self.client(t, backoff_base_s=10.0, backoff_cap_s=12.0)
        with self.assertRaises(LLMError):
            client.chat(MESSAGES)
        self.assertEqual(self.sleeps, [10.0, 12.0, 12.0, 12.0])


class TestProviderPin(LLMTestCase):
    def test_mismatched_provider_is_refused_not_cached_not_retried(self):
        t = ScriptedTransport([fake_response("from elsewhere", provider="Novita")])
        client = self.client(t)
        with self.assertRaises(ProviderMismatchError):
            client.chat(MESSAGES)
        self.assertEqual(len(t.calls), 1)
        self.assertEqual(list(self.cache_dir.glob("*.json")) if self.cache_dir.exists() else [], [])
        self.assertEqual(client.stats.snapshot()["failures"], 1)

    def test_matching_provider_name_is_accepted(self):
        res = self.client(ScriptedTransport([fake_response("ok", provider="PrimeIntellect")])).chat(MESSAGES)
        self.assertEqual(res.provider, "PrimeIntellect")

    def test_missing_provider_field_is_accepted(self):
        res = self.client(ScriptedTransport([fake_response("ok", provider=None)])).chat(MESSAGES)
        self.assertEqual(res.text, "ok")


class TestCache(LLMTestCase):
    def test_cache_hit_makes_no_network_call_and_hides_key(self):
        secret = "sk-test-secret-value-123"
        first = LLMClient(self.config(), transport=ScriptedTransport([fake_response("cached text")]),
                          api_key=secret, sleep=self.sleeps.append)
        r1 = first.chat(MESSAGES)
        self.assertFalse(r1.cached)

        t2 = ScriptedTransport()
        second = self.client(t2)
        r2 = second.chat(MESSAGES)
        self.assertEqual(t2.calls, [])
        self.assertTrue(r2.cached)
        self.assertEqual(r2.text, "cached text")
        self.assertEqual(r2.cost_usd, 0.0)
        self.assertEqual(second.stats.snapshot()["cache_hits"], 1)
        self.assertEqual(second.stats.snapshot()["network_calls"], 0)
        for path in self.cache_dir.glob("*.json"):
            self.assertNotIn(secret, path.read_text(encoding="utf-8"))

    def test_cache_file_location_and_content(self):
        self.client(ScriptedTransport([fake_response("hello")])).chat(MESSAGES)
        files = list(self.cache_dir.glob("*.json"))
        self.assertEqual(len(files), 1)
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["provider"], "PrimeIntellect")
        self.assertEqual(files[0].stem, payload["key"])

    def test_different_messages_do_not_hit_cache(self):
        t = ScriptedTransport([fake_response("a"), fake_response("b")])
        client = self.client(t)
        self.assertEqual(client.chat(MESSAGES).text, "a")
        self.assertEqual(client.chat(MESSAGES[:1]).text, "b")
        self.assertEqual(len(t.calls), 2)

    def test_cache_disabled_always_calls_network(self):
        t = ScriptedTransport([fake_response("a"), fake_response("b")])
        client = self.client(t, cache_enabled=False)
        client.chat(MESSAGES)
        self.assertEqual(client.chat(MESSAGES).text, "b")
        self.assertEqual(len(t.calls), 2)

    def test_empty_text_is_not_cached(self):
        t = ScriptedTransport([fake_response(""), fake_response("second")])
        client = self.client(t)
        self.assertEqual(client.chat(MESSAGES).text, "")
        self.assertEqual(client.chat(MESSAGES).text, "second")
        self.assertEqual(len(t.calls), 2)

    def test_corrupt_cache_entry_is_ignored(self):
        cfg = self.config()
        client = LLMClient(cfg, transport=ScriptedTransport([fake_response("fresh")]), sleep=self.sleeps.append)
        key = cache_key({"v": 1, "model": cfg.model, "provider": cfg.provider,
                         "allow_fallbacks": cfg.allow_fallbacks, "messages": MESSAGES,
                         "params": {"temperature": 0.0, "top_p": None, "top_k": None,
                                    "max_tokens": 8192, "reasoning": None}})
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        (cfg.cache_dir / f"{key}.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(client.chat(MESSAGES).text, "fresh")
        # The entry at the expected key was replaced, which proves the key matched.
        rewritten = json.loads((cfg.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        self.assertEqual(rewritten["text"], "fresh")


class TestUsageAndCost(LLMTestCase):
    def test_parse_usage_reads_reasoning_and_cached_tokens(self):
        u = parse_usage({"prompt_tokens": 27, "completion_tokens": 185,
                         "completion_tokens_details": {"reasoning_tokens": 141},
                         "prompt_tokens_details": {"cached_tokens": 10}, "cost": 8.68e-06})
        self.assertEqual((u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.cached_prompt_tokens),
                         (27, 185, 141, 10))
        self.assertAlmostEqual(u.openrouter_cost_usd, 8.68e-06)
        self.assertEqual(parse_usage(None), Usage())

    def test_estimated_cost_uses_configured_prices(self):
        cfg = self.config()
        u = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        self.assertAlmostEqual(estimate_cost(u, cfg), 0.033 + 0.048)
        cheap = self.config(price_prompt_per_m=0.0, price_completion_per_m=0.0)
        self.assertEqual(estimate_cost(u, cheap), 0.0)

    def test_stats_accumulate_across_calls(self):
        t = ScriptedTransport([fake_response("a"), fake_response("b")])
        client = self.client(t, cache_enabled=False)
        client.chat(MESSAGES)
        client.chat(MESSAGES)
        snap = client.stats.snapshot()
        self.assertEqual(snap["network_calls"], 2)
        self.assertEqual(snap["usage"]["prompt_tokens"], 200)
        self.assertEqual(snap["usage"]["reasoning_tokens"], 10)
        self.assertGreater(snap["est_cost_usd"], 0)
        self.assertIn("LLM calls: 2", client.stats.summary())

    def test_thread_safe_counters(self):
        def always_ok(kwargs):
            return fake_response("x")

        client = LLMClient(self.config(cache_enabled=False), transport=always_ok, sleep=lambda s: None)

        def worker():
            for _ in range(5):
                client.chat(MESSAGES)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(client.stats.snapshot()["network_calls"], 80)
        self.assertEqual(client.stats.snapshot()["calls"], 80)


class TestConfig(LLMTestCase):
    def test_defaults_pin_provider_without_fallbacks(self):
        cfg = LLMConfig.from_env({})
        self.assertEqual(cfg.model, "qwen/qwen3.5-9b")
        self.assertEqual(cfg.provider, "primeintellect/bf16")
        self.assertFalse(cfg.allow_fallbacks)
        self.assertEqual(cfg.temperature, 0.0)
        self.assertIsNone(cfg.reasoning)
        self.assertTrue(cfg.cache_enabled)

    def test_env_values_are_parsed(self):
        cfg = LLMConfig.from_env({
            "LLM_MODEL": "qwen/qwen3.5-9b", "LLM_PROVIDER": "primeintellect/bf16",
            "LLM_ALLOW_FALLBACKS": "false", "LLM_TEMPERATURE": "0", "LLM_MAX_TOKENS": "4096",
            "LLM_REASONING": "LOW", "LLM_CACHE": "0", "LLM_PRICE_PROMPT_PER_M": "0.05",
        })
        self.assertEqual(cfg.reasoning, "low")
        self.assertEqual(cfg.max_tokens, 4096)
        self.assertFalse(cfg.cache_enabled)
        self.assertEqual(cfg.price_prompt_per_m, 0.05)

    def test_invalid_reasoning_env_fails_fast(self):
        with self.assertRaises(ValueError):
            LLMConfig.from_env({"LLM_REASONING": "extreme"})

    def test_config_and_client_reprs_do_not_contain_key(self):
        client = LLMClient(self.config(), api_key="sk-test-repr-secret", transport=ScriptedTransport())
        self.assertNotIn("sk-test-repr-secret", repr(client))
        self.assertNotIn("sk-test-repr-secret", repr(client.config))

    def test_missing_api_key_is_a_clear_llm_error(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            client = LLMClient(self.config())  # real transport, no key
            with self.assertRaises(LLMError) as ctx:
                client.chat(MESSAGES)
            self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
            with self.assertRaises(LLMError):
                client.require_api_key()


if __name__ == "__main__":
    unittest.main()
