import re
from typing import Optional

from pydantic import BaseModel

from drafting_agent.agent.verifier import count_txn_citations, critique_draft
from drafting_agent.fixtures.generate_fixtures import Ledger
from drafting_agent.model_client import ModelClient

JUDGE_SYSTEM_PROMPT = """You are scoring a client-facing accounting message on two dimensions, each from 1 to 5.

completeness: does the message give the client enough detail to understand their balance, given
the transactions it had available. 1 means only a bare total with no itemized detail at all. 5
means the relevant transactions are itemized clearly.

tone: is the message professional, clear, and appropriately direct for a business client. 1 means
unprofessional, confusing, or inappropriate. 5 means clear and professional throughout.

Some facts have already been verified by code. Trust them completely, do not re-derive or second
guess them, and do not let them affect the tone score.

Respond with exactly this format and nothing else, on three separate lines:
completeness: <integer 1 to 5>
tone: <integer 1 to 5>
rationale: <one sentence>"""

COMPLETENESS_RE = re.compile(r"completeness:\s*(\d)", re.IGNORECASE)
TONE_RE = re.compile(r"tone:\s*(\d)", re.IGNORECASE)
RATIONALE_RE = re.compile(r"rationale:\s*(.+)", re.IGNORECASE)

GROUNDING_ISSUE_TYPES = {"wrong_balance", "missing_balance", "wrong_amount", "fabricated_transaction", "unsupported_claim", "leaked_internal_text"}
INVENTED_NUMBER_ISSUE_TYPES = {"fabricated_transaction", "unsupported_claim"}


class JudgeScore(BaseModel):
    factual_grounding: int
    absence_of_invented_numbers: int
    completeness: int
    tone: int
    rationale: str

    @property
    def average(self) -> float:
        return (self.factual_grounding + self.absence_of_invented_numbers + self.completeness + self.tone) / 4


def _degrade_score(issue_count: int) -> int:
    return max(1, 5 - min(4, issue_count))


def _build_judge_prompt(draft_text: str, ledger: Ledger, verified_issue_count: int) -> str:
    coverage = count_txn_citations(draft_text) / len(ledger.transactions) if ledger.transactions else 0.0
    issues_line = "none" if verified_issue_count == 0 else f"{verified_issue_count} factual issue(s) found"
    return (
        f"Verified facts (already checked by code, trust them):\n"
        f"- factual issues found in this draft: {issues_line}\n"
        f"- fraction of ledger transactions cited in this draft: {coverage:.2f}\n\n"
        f"Draft to score:\n{draft_text}"
    )


def score_draft(
    draft_text: str,
    ledger: Ledger,
    model_client: ModelClient,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    task: str = "judge_score",
) -> JudgeScore:
    critique = critique_draft(draft_text, ledger)
    grounding_issues = [i for i in critique.issues if i.issue_type in GROUNDING_ISSUE_TYPES]
    invented_issues = [i for i in critique.issues if i.issue_type in INVENTED_NUMBER_ISSUE_TYPES]

    factual_grounding = _degrade_score(len(grounding_issues))
    absence_of_invented_numbers = _degrade_score(len(invented_issues))

    prompt = _build_judge_prompt(draft_text, ledger, len(critique.issues))
    output = model_client.complete(task=task, system=system_prompt, prompt=prompt)

    completeness_match = COMPLETENESS_RE.search(output)
    tone_match = TONE_RE.search(output)
    rationale_match = RATIONALE_RE.search(output)

    completeness = int(completeness_match.group(1)) if completeness_match else 3
    tone = int(tone_match.group(1)) if tone_match else 3
    rationale = rationale_match.group(1).strip() if rationale_match else "no rationale parsed"

    return JudgeScore(
        factual_grounding=factual_grounding,
        absence_of_invented_numbers=absence_of_invented_numbers,
        completeness=completeness,
        tone=tone,
        rationale=rationale,
    )
