"""A cut-off reply must be diagnosed as cut off, and retried with more room.

Three of the last forty production runs aborted with "empty completion" when the
model had in fact produced *too much*: the reply hit `max_tokens`, was cut mid
JSON, and `complete_json` then re-asked the identical question at the identical
budget until it ran out of attempts.
"""
from __future__ import annotations

import pytest

from llm.client import LLMClient, LLMError, LLMTruncated


class _FakeClient(LLMClient):
    """LLMClient with the HTTP call replaced by a scripted list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.base_url = "https://example.invalid"
        self.api_key = "test"
        self.model = "test-model"

    def _post(self, payload):  # noqa: D102 - overrides the network call
        self.calls.append(payload["max_tokens"])
        return self._responses.pop(0)


def _reply(text, stop_reason="end_turn", out_tokens=10):
    return {
        "model": "test-model",
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 5, "output_tokens": out_tokens},
    }


def test_truncated_json_raises_rather_than_reporting_empty():
    """A cut reply with no text block used to surface as 'empty completion'."""
    client = _FakeClient([_reply("", stop_reason="max_tokens")])
    with pytest.raises(LLMTruncated) as excinfo:
        client.complete(system="s", user="u", max_tokens=64, strict_length=True)
    assert "cut off" in str(excinfo.value)


def test_prose_callers_still_get_their_length_cap():
    """Without strict_length a max_tokens stop is a deliberate cap, not an error."""
    client = _FakeClient([_reply("a long answer", stop_reason="max_tokens")])
    assert client.complete(system="s", user="u", max_tokens=64) == "a long answer"


def test_complete_json_doubles_the_budget_after_a_truncation():
    """The retry must have MORE room, or it just reproduces the same cut reply."""
    client = _FakeClient([
        _reply('{"a": 1', stop_reason="max_tokens"),   # cut mid-object
        _reply('{"a": 1}', stop_reason="end_turn"),    # fits second time
    ])
    assert client.complete_json(system="s", user="u", max_tokens=100) == {"a": 1}
    assert client.calls == [100, 200], "second attempt should double the cap"


def test_repeated_truncation_still_gives_up_with_the_real_reason():
    client = _FakeClient([_reply("", stop_reason="max_tokens") for _ in range(3)])
    with pytest.raises(LLMError) as excinfo:
        client.complete_json(system="s", user="u", max_tokens=50, attempts=3)
    assert "cut off" in str(excinfo.value)
    assert client.calls == [50, 100, 200]


def test_parse_failure_that_is_not_truncation_keeps_the_same_budget():
    """Only truncation earns more tokens; a malformed-but-complete reply does not."""
    client = _FakeClient([
        _reply("not json at all", stop_reason="end_turn"),
        _reply('{"ok": true}', stop_reason="end_turn"),
    ])
    assert client.complete_json(system="s", user="u", max_tokens=80) == {"ok": True}
    assert client.calls == [80, 80]
