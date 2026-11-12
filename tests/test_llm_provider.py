"""Provider routing tests using SDK-shaped replies and no network."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import openai
import pytest
from openai.types.chat import ChatCompletion

from company_email_agent import llm
from company_email_agent.simulation.evaluate import adapter_agent_factory
from company_email_agent.simulation.persona import load_persona
from company_email_agent.simulation.runner import make_judge_lm, make_user_lm


_MESSAGES = [
    {"role": "system", "content": "Reply with plain text."},
    {"role": "user", "content": "A synthetic test."},
    {"role": "assistant", "content": "Understood."},
    {"role": "user", "content": "Reply with OK."},
]
_REAL_LOAD_ENV = llm._load_env


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    for name in (
        "LLM_PROVIDER", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL", "CS329Z_GRADING_MODEL", "CS329Z_USER_MODEL", "CS329Z_JUDGE_MODEL",
        "CS329Z_REASONING_EFFORT", "CS329Z_BUDGET_USD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CS329Z_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(llm, "_load_env", lambda: None)
    monkeypatch.setattr(llm, "_NO_TEMPERATURE", set())
    monkeypatch.setattr(llm, "_NO_EFFORT", set())


@pytest.fixture
def sdk(monkeypatch):
    create = Mock()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(openai, "OpenAI", constructor)
    return constructor, create


def completion(model, *, finish="stop", **usage):
    return ChatCompletion(
        id="fixture-completion", created=0, model=model, object="chat.completion",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "OK"},
                  "finish_reason": finish}],
        usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, **usage},
    )


def test_default_openrouter_roles_and_legacy_cache_key():
    model = llm.LM()
    assert model.provider == "openrouter"
    assert model.model == llm.LM("user").model == "openai/gpt-6-luna"
    assert llm.LM("judge").model == "openai/gpt-6-sol"
    legacy_payload = [model.model, 0.0, 4096, "low", "", _MESSAGES]
    legacy_json = json.dumps(legacy_payload, sort_keys=True, ensure_ascii=False)
    assert model._key(_MESSAGES) == hashlib.sha256(legacy_json.encode()).hexdigest()


def test_deepseek_role_defaults_and_simulation_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "DeepSeek")
    assert [llm.LM(role).model for role in (None, "user", "judge")] == ["deepseek-flash"] * 3

    persona = load_persona(Path(__file__).parent / "fixtures/personas/01-doc-incident-severity.json")
    monkeypatch.setenv("CS329Z_GRADING_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("CS329Z_USER_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("CS329Z_JUDGE_MODEL", "deepseek-v4-pro")
    _, (agent, aux) = adapter_agent_factory(None)(persona)
    assert agent.model == aux.model == "deepseek-v4-pro"
    assert make_user_lm(persona).lm.model == "deepseek-v4-pro"
    assert make_judge_lm(persona).lm.model == "deepseek-v4-pro"
    assert llm.LM("deepseek-flash").model == "deepseek-flash"


def test_local_env_configuration_and_process_precedence(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=deepseek\nCS329Z_GRADING_MODEL=deepseek-v4-pro\n", encoding="utf-8"
    )
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(llm, "_load_env", _REAL_LOAD_ENV)
    monkeypatch.setenv("CS329Z_GRADING_MODEL", "deepseek-flash")
    # Have monkeypatch undo the variable that dotenv will populate.
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.delenv("LLM_PROVIDER")
    model = llm.LM()
    assert (model.provider, model.model) == ("deepseek", "deepseek-flash")


def test_openrouter_preserves_sdk_request_and_reported_cost(monkeypatch, sdk):
    constructor, create = sdk
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter-credential")
    create.return_value = completion("openai/gpt-6-luna", cost=0.012)
    model = llm.LM(cache=False)
    constructor.assert_not_called()
    assert model(_MESSAGES) == "OK"
    constructor.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1", api_key="fixture-openrouter-credential",
        max_retries=6, timeout=120, default_headers={"X-Title": "Company Email Agent"},
    )
    assert create.call_args.kwargs["extra_body"] == {
        "usage": {"include": True}, "reasoning": {"effort": "low"},
    }
    assert llm.read_ledger()[0]["cost_usd"] == 0.012
    assert not llm.read_ledger()[0]["cost_estimated"]


def test_deepseek_sdk_request_cache_and_usage(monkeypatch, sdk):
    constructor, create = sdk
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-openrouter-fixture")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-deepseek-credential")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/compat/")
    monkeypatch.setenv("CS329Z_REASONING_EFFORT", "high")
    create.return_value = completion("deepseek-flash", prompt_cache_hit_tokens=30)
    model = llm.LM(max_tokens=32)

    assert model(_MESSAGES) == model(_MESSAGES) == "OK"
    constructor.assert_called_once_with(
        base_url="https://example.invalid/compat", api_key="fixture-deepseek-credential",
        max_retries=6, timeout=120, default_headers=None,
    )
    create.assert_called_once_with(
        model="deepseek-flash", messages=_MESSAGES, max_tokens=32, temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    row, = llm.read_ledger()
    assert row["provider"] == "deepseek" and row["cost_estimated"]
    assert row["cached_tokens"] == 30
    assert row["cost_usd"] == pytest.approx((70 * 0.30 + 30 * 0.006 + 20 * 1.20) / 1_000_000)
    assert model.cached_calls == 1


def test_cache_isolated_by_provider_and_endpoint(monkeypatch):
    # Use the same model string on both routes to isolate the routing dimension.
    monkeypatch.setitem(llm.PRICES, "deepseek-flash", (0.30, 0.006, 1.20))
    default_router = llm.LM("deepseek-flash")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.invalid/router")
    custom_router = llm.LM("deepseek-flash")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    default_deepseek = llm.LM()
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/deepseek")
    custom_deepseek = llm.LM()
    routes = [default_router, custom_router, default_deepseek, custom_deepseek]
    assert len({model._key(_MESSAGES) for model in routes}) == 4


@pytest.mark.parametrize("provider,expected", [
    ("openrouter", "OPENROUTER_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY"),
])
def test_missing_provider_key_stops_before_sdk(monkeypatch, sdk, provider, expected):
    constructor, _ = sdk
    monkeypatch.setenv("LLM_PROVIDER", provider)
    other = "OPENROUTER_API_KEY" if provider == "deepseek" else "DEEPSEEK_API_KEY"
    monkeypatch.setenv(other, "other-provider-fixture")
    with pytest.raises(RuntimeError, match=expected):
        llm.LM(cache=False)(_MESSAGES)
    constructor.assert_not_called()


def test_invalid_provider_and_cross_provider_model_fail_early(monkeypatch, sdk):
    constructor, _ = sdk
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llm.LM()
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("CS329Z_GRADING_MODEL", "openai/gpt-6-luna")
    with pytest.raises(KeyError, match="deepseek-flash"):
        llm.LM()
    constructor.assert_not_called()


def test_deepseek_truncation_is_logged_and_not_cached(monkeypatch, sdk):
    _, create = sdk
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-deepseek-credential")
    create.return_value = completion("deepseek-flash", finish="length")
    model = llm.LM()
    for _ in range(2):
        with pytest.raises(llm.OutputTruncated) as caught:
            model(_MESSAGES)
        assert caught.value.text == "OK"
    assert create.call_count == 2
    assert [row["finish_reason"] for row in llm.read_ledger()] == ["length", "length"]


def test_deepseek_sdk_error_does_not_echo_provider_body(monkeypatch, sdk):
    _, create = sdk
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-deepseek-credential")
    create.side_effect = openai.OpenAIError("untrusted-provider-error-body")
    with pytest.raises(RuntimeError, match=r"DeepSeek Chat Completions failed \(OpenAIError\)") as caught:
        llm.LM(cache=False)(_MESSAGES)
    assert "untrusted-provider-error-body" not in str(caught.value)


def test_openrouter_still_retries_without_unsupported_reasoning(monkeypatch, sdk):
    _, create = sdk
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter-credential")
    create.side_effect = [
        RuntimeError("reasoning is not supported"), completion("openai/gpt-6-luna"),
    ]
    assert llm.LM(cache=False)(_MESSAGES) == "OK"
    assert create.call_count == 2
    assert create.call_args.kwargs["extra_body"] == {"usage": {"include": True}}
