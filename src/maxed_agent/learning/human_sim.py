from pathlib import Path

from maxed_agent.agent.loop import LoopResult
from maxed_agent.agent.verifier import AMOUNT_RE, BALANCE_RE, Issue, count_txn_citations
from maxed_agent.learning.decision_log import (
    DEFAULT_LOG_PATH,
    CapabilityGapRecord,
    DecisionRecord,
    log_capability_gap,
    log_decision,
    now,
)

MIN_TXN_CITATIONS_FOR_APPROVAL = 1


def _apply_correction(text: str, issue: Issue) -> str:
    if not issue.expected:
        return text
    if issue.issue_type == "wrong_balance":
        return BALANCE_RE.sub(f"Total balance due: ${issue.expected}", text, count=1)
    if issue.issue_type == "wrong_amount" and issue.sentence:
        corrected_sentence = AMOUNT_RE.sub(f"${issue.expected}", issue.sentence, count=1)
        return text.replace(issue.sentence, corrected_sentence, 1)
    return text


def simulate_human(
    client_id: str,
    result: LoopResult,
    index: int = 0,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    if result.action == "approved":
        if count_txn_citations(result.draft_text) < MIN_TXN_CITATIONS_FOR_APPROVAL:
            log_decision(
                DecisionRecord(
                    timestamp=now(),
                    client_id=client_id,
                    draft_text=result.draft_text,
                    critique_issues=[],
                    loop_action=result.action,
                    attempts=result.attempts,
                    human_decision="decline",
                    final_text="",
                    edit_note=(
                        "declined: draft does not cite any specific transaction, "
                        "too sparse to send to a client even though the verifier found no false claims"
                    ),
                ),
                path=log_path,
            )
            return

        if index % 3 == 0 and result.draft_text.strip():
            final_text = "Thank you for your continued business.\n\n" + result.draft_text
            human_decision = "approve_with_edit"
            edit_note = "added a courtesy opening line"
        else:
            final_text = result.draft_text
            human_decision = "approve_as_is"
            edit_note = ""
        log_decision(
            DecisionRecord(
                timestamp=now(),
                client_id=client_id,
                draft_text=result.draft_text,
                critique_issues=[],
                loop_action=result.action,
                attempts=result.attempts,
                human_decision=human_decision,
                final_text=final_text,
                edit_note=edit_note,
            ),
            path=log_path,
        )
        return

    if result.action == "escalate":
        final_text = result.draft_text
        for issue in result.critique.issues:
            final_text = _apply_correction(final_text, issue)
        log_decision(
            DecisionRecord(
                timestamp=now(),
                client_id=client_id,
                draft_text=result.draft_text,
                critique_issues=[i.description for i in result.critique.issues],
                loop_action=result.action,
                attempts=result.attempts,
                human_decision="approve_with_edit",
                final_text=final_text,
                edit_note="corrected: " + "; ".join(i.description for i in result.critique.issues),
            ),
            path=log_path,
        )
        return

    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id=client_id,
            draft_text=result.draft_text,
            critique_issues=[i.description for i in result.critique.issues],
            loop_action=result.action,
            attempts=result.attempts,
            human_decision="decline",
            final_text="",
        ),
        path=log_path,
    )
    issue = result.critique.issues[0] if result.critique.issues else None
    log_capability_gap(
        CapabilityGapRecord(
            timestamp=now(),
            client_id=client_id,
            gap_type="missing_record",
            description=result.reason,
            related_issue=issue.description if issue else "",
        ),
        path=log_path,
    )
