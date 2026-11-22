from drafting_agent.agent.injected_errors import build_injected_error_cases
from drafting_agent.agent.loop import run_loop
from drafting_agent.fixtures.generate_fixtures import load_fixtures
from drafting_agent.model_client import ModelClient


def _cases_by_name(ledger):
    return {c["name"]: c["text"] for c in build_injected_error_cases(ledger)}


def test_loop_approves_a_clean_draft():
    client, ledger = load_fixtures()
    texts = _cases_by_name(ledger)
    model_client = ModelClient(
        provider="stub",
        stub_responses={"generate_draft": [texts["clean baseline (control)"]]},
    )
    result = run_loop(client, ledger, model_client)
    assert result.action == "approved"
    assert result.attempts == 1


def test_loop_escalates_on_persistent_correctable_error():
    client, ledger = load_fixtures()
    texts = _cases_by_name(ledger)
    model_client = ModelClient(
        provider="stub",
        stub_responses={"generate_draft": [texts["wrong amount on a real transaction"]]},
    )
    result = run_loop(client, ledger, model_client, max_revisions=2)
    assert result.action == "escalate"
    assert result.attempts == 3
    assert "confirm" in result.reason.lower()


def test_loop_abstains_on_persistent_uncorrectable_error():
    client, ledger = load_fixtures()
    texts = _cases_by_name(ledger)
    model_client = ModelClient(
        provider="stub",
        stub_responses={"generate_draft": [texts["fabricated transaction"]]},
    )
    result = run_loop(client, ledger, model_client, max_revisions=2)
    assert result.action == "abstain"
    assert result.attempts == 3
    assert "cannot produce a trustworthy draft" in result.reason.lower()
