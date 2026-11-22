from drafting_agent.fixtures.generate_fixtures import load_fixtures
from drafting_agent.learning.decision_log import DecisionRecord, log_decision, now
from drafting_agent.learning.escalation_calibration import calibrate_abstain_escalate


def test_escalate_correction_success_rate_and_abstain_breakdown(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    _, ledger = load_fixtures()
    txn = ledger.transactions[0]

    clean_final_text = (
        f"A payment of ${abs(txn.amount):.2f} ({txn.id}) was received.\n"
        f"Total balance due: ${ledger.balance:.2f}"
    )
    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text=f"A payment of ${abs(txn.amount) + 50:.2f} ({txn.id}) was received.\nTotal balance due: ${ledger.balance:.2f}",
            loop_action="escalate",
            attempts=3,
            human_decision="approve_with_edit",
            final_text=clean_final_text,
            edit_note="corrected wrong amount",
        ),
        path=log_path,
    )

    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text="A charge of $95.00 (TXN-999) was recorded.\nTotal balance due: $1337.50",
            loop_action="abstain",
            attempts=3,
            human_decision="decline",
            final_text="",
        ),
        path=log_path,
    )

    report = calibrate_abstain_escalate(log_path)

    assert report.escalate_count == 1
    assert report.escalate_corrections_clean == 1
    assert report.escalate_correction_success_rate == 1.0
    assert report.abstain_count == 1
    assert report.abstain_reason_breakdown.get("fabricated_transaction") == 1


def test_no_escalate_cases_reports_none_not_a_crash(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    report = calibrate_abstain_escalate(log_path)
    assert report.escalate_count == 0
    assert report.escalate_correction_success_rate is None
