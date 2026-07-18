from typing import Literal, Optional

from pydantic import BaseModel

from maxed_agent.agent.generator import generate_draft
from maxed_agent.agent.verifier import CORRECTABLE, UNCORRECTABLE, Critique, critique_draft
from maxed_agent.fixtures.generate_fixtures import Client, Ledger
from maxed_agent.model_client import ModelClient

Action = Literal["approved", "abstain", "escalate"]


class LoopResult(BaseModel):
    action: Action
    draft_text: str
    critique: Critique
    reason: str = ""
    attempts: int


def run_loop(
    client: Client,
    ledger: Ledger,
    model_client: ModelClient,
    max_revisions: int = 2,
    memory_block: Optional[str] = None,
) -> LoopResult:
    feedback: Optional[str] = None
    draft_text = ""
    critique = Critique()
    attempt = 0

    for attempt in range(1, max_revisions + 2):
        draft_text = generate_draft(
            client, ledger, model_client, feedback=feedback, memory_block=memory_block
        )
        critique = critique_draft(draft_text, ledger)
        if not critique.has_issues:
            return LoopResult(action="approved", draft_text=draft_text, critique=critique, attempts=attempt)
        if attempt <= max_revisions:
            feedback = "\n".join(f"- {issue.description}" for issue in critique.issues)
            continue
        break

    uncorrectable = [i for i in critique.issues if i.issue_type in UNCORRECTABLE]
    if uncorrectable:
        return LoopResult(
            action="abstain",
            draft_text=draft_text,
            critique=critique,
            reason=f"Cannot produce a trustworthy draft from these records: {uncorrectable[0].description}",
            attempts=attempt,
        )

    correctable = [i for i in critique.issues if i.issue_type in CORRECTABLE]
    return LoopResult(
        action="escalate",
        draft_text=draft_text,
        critique=critique,
        reason=f"Needs human confirmation: {correctable[0].description}",
        attempts=attempt,
    )
