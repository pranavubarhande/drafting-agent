from maxed_agent.agent.verifier import Critique, Issue
from maxed_agent.agent.loop import LoopResult
from maxed_agent.learning.decision_log import load_decisions
from maxed_agent.learning.human_sim import simulate_human


def test_sparse_approved_draft_is_declined_not_rubber_stamped(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    result = LoopResult(
        action="approved",
        draft_text="Total balance due: $100.00",
        critique=Critique(issues=[]),
        attempts=1,
    )

    simulate_human("CLIENT-001", result, index=0, log_path=log_path)

    decisions = load_decisions(log_path)
    assert len(decisions) == 1
    assert decisions[0]["human_decision"] == "decline"
    assert decisions[0]["final_text"] == ""


def test_detailed_approved_draft_is_approved(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    result = LoopResult(
        action="approved",
        draft_text="A payment of $50.00 (TXN-001) was received. Total balance due: $100.00",
        critique=Critique(issues=[]),
        attempts=1,
    )

    simulate_human("CLIENT-001", result, index=1, log_path=log_path)

    decisions = load_decisions(log_path)
    assert len(decisions) == 1
    assert decisions[0]["human_decision"] in ("approve_as_is", "approve_with_edit")
    assert decisions[0]["final_text"]


def test_escalate_corrects_every_wrong_amount_not_just_the_first(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    draft_text = (
        "TXN-001: $500.00 (wrong)\n"
        "TXN-002: $999.00 (also wrong)\n"
        "Total balance due: $100.00"
    )
    result = LoopResult(
        action="escalate",
        draft_text=draft_text,
        critique=Critique(
            issues=[
                Issue(
                    issue_type="wrong_amount",
                    description="TXN-001 is actually 450.00, but the draft states [500.0] for it.",
                    sentence="TXN-001: $500.00 (wrong)",
                    expected="450.00",
                ),
                Issue(
                    issue_type="wrong_amount",
                    description="TXN-002 is actually 450.00, but the draft states [999.0] for it.",
                    sentence="TXN-002: $999.00 (also wrong)",
                    expected="450.00",
                ),
            ]
        ),
        attempts=3,
        reason="needs human confirmation",
    )

    simulate_human("CLIENT-001", result, index=0, log_path=log_path)

    decisions = load_decisions(log_path)
    final_text = decisions[0]["final_text"]
    assert "$500.00" not in final_text
    assert "$999.00" not in final_text
    assert final_text.count("$450.00") == 2
