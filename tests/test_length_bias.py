from maxed_agent.agent.verifier import count_txn_citations
from maxed_agent.eval.length_bias import build_padded_variant, measure_anchored, measure_naive
from maxed_agent.fixtures.generate_fixtures import load_fixtures
from maxed_agent.model_client import ModelClient


def _sparse_draft(ledger) -> str:
    txn = ledger.transactions[1]
    return f"A payment of ${abs(txn.amount):.2f} ({txn.id}) was received.\nTotal balance due: ${ledger.balance:.2f}"


def test_padded_variant_adds_no_new_transaction_or_amount_claims():
    _, ledger = load_fixtures()
    sparse = _sparse_draft(ledger)
    padded = build_padded_variant(sparse)

    assert count_txn_citations(padded) == count_txn_citations(sparse)
    assert len(padded) > len(sparse)


def test_naive_prompt_shows_length_bias_in_stub_mode():
    _, ledger = load_fixtures()
    sparse = _sparse_draft(ledger)
    model_client = ModelClient(
        provider="stub",
        stub_responses={
            "naive_judge_score": [
                "completeness: 3\ntone: 4\nrationale: partial detail",
                "completeness: 5\ntone: 5\nrationale: thorough and transparent",
            ]
        },
    )

    result = measure_naive(sparse, ledger, model_client)

    assert result.completeness_gap == 2
    assert result.tone_gap == 1


def test_anchored_prompt_shows_no_length_bias_in_stub_mode():
    _, ledger = load_fixtures()
    sparse = _sparse_draft(ledger)
    model_client = ModelClient(
        provider="stub",
        stub_responses={
            "judge_score": [
                "completeness: 2\ntone: 5\nrationale: lacks itemization",
                "completeness: 2\ntone: 5\nrationale: still lacks itemization despite the padding",
            ]
        },
    )

    result = measure_anchored(sparse, ledger, model_client)

    assert result.completeness_gap == 0
    assert result.tone_gap == 0
