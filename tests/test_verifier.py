from drafting_agent.agent.generator import FEEDBACK_PREAMBLE
from drafting_agent.agent.injected_errors import build_injected_error_cases
from drafting_agent.agent.verifier import critique_draft
from drafting_agent.fixtures.generate_fixtures import load_fixtures


def test_verifier_catches_every_injected_error():
    _, ledger = load_fixtures()
    cases = build_injected_error_cases(ledger)
    for case in cases:
        critique = critique_draft(case["text"], ledger)
        if case["expected_type"] is None:
            assert not critique.has_issues, f"false positive on: {case['name']}"
        else:
            found_types = {i.issue_type for i in critique.issues}
            assert case["expected_type"] in found_types, f"missed: {case['name']}"


def test_verifier_accepts_clean_generated_style_draft():
    _, ledger = load_fixtures()
    from drafting_agent.agent.injected_errors import build_baseline_text

    critique = critique_draft(build_baseline_text(ledger), ledger)
    assert not critique.has_issues


def test_verifier_catches_wrong_amount_in_bulleted_list_format():
    _, ledger = load_fixtures()
    txn1, txn2 = ledger.transactions[0], ledger.transactions[1]
    other_lines = "\n".join(
        f"- {t.id}: ${abs(t.amount):.2f} - {t.description}"
        for t in ledger.transactions[2:10]
    )
    draft = (
        f"- {txn1.id}: ${abs(txn1.amount) + 50:.2f} - {txn1.description}\n"
        f"- {txn2.id}: ${abs(txn2.amount):.2f} - {txn2.description}\n"
        f"{other_lines}\n"
        f"Total balance due: ${ledger.balance:.2f}"
    )
    critique = critique_draft(draft, ledger)
    wrong_amount_txns = {
        i.description.split(" is actually")[0] for i in critique.issues if i.issue_type == "wrong_amount"
    }
    assert txn1.id in wrong_amount_txns, "wrong amount on one line was masked by correct amounts on other lines"


def test_verifier_catches_leaked_revision_feedback():
    _, ledger = load_fixtures()
    from drafting_agent.agent.injected_errors import build_baseline_text

    draft = FEEDBACK_PREAMBLE + "some prior issue\n\n" + build_baseline_text(ledger)
    critique = critique_draft(draft, ledger)
    found_types = {i.issue_type for i in critique.issues}
    assert "leaked_internal_text" in found_types


def test_verifier_handles_thousand_separator_amounts():
    _, ledger = load_fixtures()
    draft = f"Total balance due: ${ledger.balance:,.2f}"
    critique = critique_draft(draft, ledger)
    assert not critique.has_issues, "comma thousand-separator in the balance line was not parsed correctly"


def test_verifier_handles_negative_amount_notations():
    _, ledger = load_fixtures()
    payment = next(t for t in ledger.transactions if t.amount < 0)
    for notation in (
        f"-${abs(payment.amount):.2f} ({payment.id})",
        f"(${abs(payment.amount):.2f}) ({payment.id})",
        f"${abs(payment.amount):.2f} ({payment.id})",
    ):
        critique = critique_draft(f"{notation}\nTotal balance due: ${ledger.balance:.2f}", ledger)
        wrong_amount_issues = [i for i in critique.issues if i.issue_type == "wrong_amount"]
        assert not wrong_amount_issues, f"false positive on negative notation: {notation!r}"


def test_verifier_catches_all_issues_present_simultaneously():
    _, ledger = load_fixtures()
    real_txn = ledger.transactions[0]
    draft = (
        f"{real_txn.id}: ${abs(real_txn.amount) + 999:.2f} {real_txn.description}\n"
        f"TXN-999: $95.00 a service charge that never happened\n"
        f"An unrelated processing fee of $43.00 was also applied\n"
        f"Total balance due: ${ledger.balance + 1:.2f}"
    )
    critique = critique_draft(draft, ledger)
    found_types = {i.issue_type for i in critique.issues}
    assert found_types == {"wrong_amount", "fabricated_transaction", "unsupported_claim", "wrong_balance"}, (
        f"expected all four issue types in one draft, got {found_types}"
    )


def test_verifier_handles_amount_immediately_before_sentence_period():
    _, ledger = load_fixtures()
    txn = ledger.transactions[0]
    draft = (
        f"A charge of ${abs(txn.amount):.2f} was recorded for {txn.id}. "
        f"Total balance due: ${ledger.balance:.2f}"
    )
    critique = critique_draft(draft, ledger)
    assert not critique.has_issues, "amount immediately followed by a sentence-ending period was misparsed"


def test_verifier_catches_wrong_amount_in_markdown_table_row():
    _, ledger = load_fixtures()
    txn = ledger.transactions[0]
    draft = (
        f"| id | amount |\n"
        f"| {txn.id} | ${abs(txn.amount) + 10:.2f} |\n"
        f"Total balance due: ${ledger.balance:.2f}"
    )
    critique = critique_draft(draft, ledger)
    found_types = {i.issue_type for i in critique.issues}
    assert "wrong_amount" in found_types, "wrong amount in a markdown table row was not caught"
