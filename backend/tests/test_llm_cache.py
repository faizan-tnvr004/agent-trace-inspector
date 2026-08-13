"""Tests for the completion cache.

The cache exists because corpus generation is quota-bound rather than
compute-bound: a deep run spends several calls, the free tier allows a few
hundred a day, and without a cache a bug in a late graph node means re-paying
for every call in the earlier ones.

Its correctness properties are narrow but load-bearing. It must never return a
response generated under a different model or prompt, and it must not put
fabricated timings into the corpus.
"""

from __future__ import annotations

from pathlib import Path

from harness.llm import LLMResponse, ResponseCache


def _response(text: str = "hello", model: str = "m1") -> LLMResponse:
    return LLMResponse(
        text=text,
        prompt_tokens=11,
        completion_tokens=7,
        latency_ms=250,
        model=model,
    )


def test_a_stored_response_comes_back(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("m1", "prompt", "system", _response())
    got = cache.get("m1", "prompt", "system")
    assert got is not None
    assert got.text == "hello"
    assert got.prompt_tokens == 11
    assert got.completion_tokens == 7


def test_a_miss_returns_none(tmp_path: Path) -> None:
    assert ResponseCache(tmp_path).get("m1", "prompt", "system") is None


def test_a_different_model_is_a_different_entry(tmp_path: Path) -> None:
    """Serving one model's answer for another would silently blend two models
    into one corpus."""
    cache = ResponseCache(tmp_path)
    cache.put("m1", "prompt", None, _response())
    assert cache.get("m2", "prompt", None) is None


def test_a_different_prompt_is_a_different_entry(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("m1", "prompt a", None, _response())
    assert cache.get("m1", "prompt b", None) is None


def test_a_different_system_prompt_is_a_different_entry(tmp_path: Path) -> None:
    """The system prompt changes the answer, so it has to be in the key."""
    cache = ResponseCache(tmp_path)
    cache.put("m1", "prompt", "system a", _response())
    assert cache.get("m1", "prompt", "system b") is None


def test_the_key_cannot_be_confused_by_concatenation(tmp_path: Path) -> None:
    """Fields are separated in the digest, so ("ab", "c") and ("a", "bc") are
    different keys rather than the same one."""
    cache = ResponseCache(tmp_path)
    cache.put("m1", "ab", "c", _response("first"))
    cache.put("m1", "a", "bc", _response("second"))
    assert cache.get("m1", "ab", "c").text == "first"
    assert cache.get("m1", "a", "bc").text == "second"


def test_latency_is_not_replayed(tmp_path: Path) -> None:
    """A cached call did not take 250ms, and recording that it did would put a
    fabricated timing into the corpus."""
    cache = ResponseCache(tmp_path)
    cache.put("m1", "prompt", None, _response())
    assert cache.get("m1", "prompt", None).latency_ms == 0


def test_hits_and_misses_are_counted(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.get("m1", "prompt", None)
    cache.put("m1", "prompt", None, _response())
    cache.get("m1", "prompt", None)
    assert (cache.hits, cache.misses) == (1, 1)


def test_a_corrupt_entry_is_treated_as_a_miss(tmp_path: Path) -> None:
    """A process killed mid-write must not poison the cache permanently."""
    cache = ResponseCache(tmp_path)
    cache.put("m1", "prompt", None, _response())
    next(tmp_path.glob("*.json")).write_text("{truncated")
    assert cache.get("m1", "prompt", None) is None


def test_no_temporary_files_are_left_behind(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("m1", "prompt", None, _response())
    assert list(tmp_path.glob("*.tmp")) == []
