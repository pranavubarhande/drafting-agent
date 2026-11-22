from collections import Counter
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from drafting_agent.agent.verifier import UNCORRECTABLE, critique_draft
from drafting_agent.fixtures.generate_fixtures import Ledger, load_fixtures
from drafting_agent.learning.decision_log import DEFAULT_LOG_PATH, load_decisions


class EscalationCalibrationReport(BaseModel):
    escalate_count: int
    escalate_corrections_clean: int
    abstain_count: int
    abstain_reason_breakdown: dict[str, int]

    @property
    def escalate_correction_success_rate(self) -> Optional[float]:
        if self.escalate_count == 0:
            return None
        return self.escalate_corrections_clean / self.escalate_count


def calibrate_abstain_escalate(log_path: Path = DEFAULT_LOG_PATH) -> EscalationCalibrationReport:
    decisions = load_decisions(log_path)
    escalate_decisions = [d for d in decisions if d["loop_action"] == "escalate"]
    abstain_decisions = [d for d in decisions if d["loop_action"] == "abstain"]

    ledger_cache: dict[str, Ledger] = {}

    def ledger_for(client_id: str) -> Ledger:
        if client_id not in ledger_cache:
            _, ledger_cache[client_id] = load_fixtures(client_id)
        return ledger_cache[client_id]

    clean_count = 0
    for d in escalate_decisions:
        critique = critique_draft(d["final_text"], ledger_for(d["client_id"]))
        if not critique.has_issues:
            clean_count += 1

    breakdown: Counter = Counter()
    for d in abstain_decisions:
        critique = critique_draft(d["draft_text"], ledger_for(d["client_id"]))
        uncorrectable_types = [i.issue_type for i in critique.issues if i.issue_type in UNCORRECTABLE]
        bucket = uncorrectable_types[0] if uncorrectable_types else "unknown"
        breakdown[bucket] += 1

    return EscalationCalibrationReport(
        escalate_count=len(escalate_decisions),
        escalate_corrections_clean=clean_count,
        abstain_count=len(abstain_decisions),
        abstain_reason_breakdown=dict(breakdown),
    )
