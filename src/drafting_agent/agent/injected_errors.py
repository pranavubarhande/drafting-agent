from typing import Optional, TypedDict

from drafting_agent.fixtures.generate_fixtures import Ledger


class InjectedCase(TypedDict):
    name: str
    text: str
    expected_type: Optional[str]


def build_baseline_text(ledger: Ledger) -> str:
    sentences = []
    for t in ledger.transactions[:6]:
        sentences.append(
            f"On {t.date}, a {t.type} of ${abs(t.amount):.2f} ({t.id}) was recorded: {t.description}."
        )
    body = " ".join(sentences)
    return f"{body}\nTotal balance due: ${ledger.balance:.2f}"


def build_injected_error_cases(ledger: Ledger) -> list[InjectedCase]:
    baseline = build_baseline_text(ledger)
    real_txn = ledger.transactions[0]

    wrong_amount_text = baseline.replace(
        f"${abs(real_txn.amount):.2f} ({real_txn.id})",
        f"${abs(real_txn.amount) + 75:.2f} ({real_txn.id})",
    )

    fabricated_text = (
        baseline
        + " Additionally, on 2026-07-01, a service charge of $95.00 (TXN-999) was recorded."
    )

    unsupported_text = (
        baseline
        + " You also have an outstanding processing fee of $43.00 from a prior cycle."
    )

    wrong_balance_text = baseline.replace(
        f"Total balance due: ${ledger.balance:.2f}",
        f"Total balance due: ${ledger.balance + 100:.2f}",
    )

    return [
        {"name": "wrong amount on a real transaction", "text": wrong_amount_text, "expected_type": "wrong_amount"},
        {"name": "fabricated transaction", "text": fabricated_text, "expected_type": "fabricated_transaction"},
        {"name": "unsupported claim, no evidence", "text": unsupported_text, "expected_type": "unsupported_claim"},
        {"name": "wrong total balance", "text": wrong_balance_text, "expected_type": "wrong_balance"},
        {"name": "clean baseline (control)", "text": baseline, "expected_type": None},
    ]
