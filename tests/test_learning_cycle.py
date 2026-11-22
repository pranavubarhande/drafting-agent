from pathlib import Path

from drafting_agent.agent.injected_errors import build_injected_error_cases
from drafting_agent.fixtures.generate_fixtures import load_fixtures
from drafting_agent.learning.decision_log import load_capability_gaps, load_decisions
from drafting_agent.learning.learning_cycle import run_learning_cycle
from drafting_agent.model_client import ModelClient


def test_learning_cycle_logs_decisions_and_capability_gaps(tmp_path):
    _, ledger = load_fixtures()
    texts = {c["name"]: c["text"] for c in build_injected_error_cases(ledger)}
    log_path = tmp_path / "decisions.jsonl"

    model_client = ModelClient(
        provider="stub",
        stub_responses={
            "generate_draft": [
                texts["clean baseline (control)"],
                texts["fabricated transaction"],
                texts["fabricated transaction"],
                texts["fabricated transaction"],
                texts["clean baseline (control)"],
            ]
        },
    )

    report = run_learning_cycle(model_client, batch_size=3, log_path=log_path)

    assert len(report.batch1) == 3
    assert len(report.batch2) == 3

    decisions = load_decisions(log_path)
    gaps = load_capability_gaps(log_path)
    assert len(decisions) == 6
    assert len(gaps) >= 1
    assert any(d["human_decision"] == "approve_as_is" for d in decisions)
    assert any(d["human_decision"] == "decline" for d in decisions)


def test_learning_cycle_resets_log_between_runs(tmp_path):
    _, ledger = load_fixtures()
    texts = {c["name"]: c["text"] for c in build_injected_error_cases(ledger)}
    log_path = tmp_path / "decisions.jsonl"

    model_client = ModelClient(
        provider="stub",
        stub_responses={"generate_draft": [texts["clean baseline (control)"]]},
    )

    run_learning_cycle(model_client, batch_size=2, log_path=log_path)
    first_len = len(load_decisions(log_path))

    model_client2 = ModelClient(
        provider="stub",
        stub_responses={"generate_draft": [texts["clean baseline (control)"]]},
    )
    run_learning_cycle(model_client2, batch_size=2, log_path=log_path)
    second_len = len(load_decisions(log_path))

    assert first_len == second_len == 4


def test_learning_cycle_memory_does_not_leak_across_clients(tmp_path):
    from drafting_agent.agent.injected_errors import build_baseline_text

    _, ledger_one = load_fixtures("CLIENT-001")
    _, ledger_two = load_fixtures("CLIENT-002")
    log_path = tmp_path / "decisions.jsonl"

    clean_one = build_baseline_text(ledger_one)
    clean_two = build_baseline_text(ledger_two)

    model_client_one = ModelClient(provider="stub", stub_responses={"generate_draft": [clean_one]})
    run_learning_cycle(
        model_client_one, batch_size=2, log_path=log_path, client_id="CLIENT-001", reset_log=True
    )

    model_client_two = ModelClient(provider="stub", stub_responses={"generate_draft": [clean_two]})
    report_two = run_learning_cycle(
        model_client_two, batch_size=2, log_path=log_path, client_id="CLIENT-002", reset_log=False
    )

    client_one_txn_ids = {t.id for t in ledger_one.transactions}
    for txn_id in client_one_txn_ids:
        assert txn_id not in report_two.memory_block, f"client 1's {txn_id} leaked into client 2's memory"

    assert any(t.id in report_two.memory_block for t in ledger_two.transactions), (
        "sanity check: client 2's own memory should still contain client 2's transactions"
    )
