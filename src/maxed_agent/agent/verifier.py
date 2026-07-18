import re
from typing import Literal, Optional

from pydantic import BaseModel

from maxed_agent.fixtures.generate_fixtures import Ledger

AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")
TXN_RE = re.compile(r"TXN-\d+")
BALANCE_RE = re.compile(r"Total balance due:\s*\$\s?([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)

LEAKED_FEEDBACK_MARKER = "internal note, do not repeat it"

IssueType = Literal[
    "missing_balance",
    "wrong_balance",
    "fabricated_transaction",
    "wrong_amount",
    "unsupported_claim",
    "leaked_internal_text",
]

CORRECTABLE: set[IssueType] = {"wrong_balance", "wrong_amount"}
UNCORRECTABLE: set[IssueType] = {
    "missing_balance",
    "fabricated_transaction",
    "unsupported_claim",
    "leaked_internal_text",
}


class Issue(BaseModel):
    issue_type: IssueType
    description: str
    sentence: str
    expected: Optional[str] = None


class Critique(BaseModel):
    issues: list[Issue] = []

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0


def count_txn_citations(text: str) -> int:
    return len(TXN_RE.findall(text))


def _parse_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


def _split_sentences(text: str) -> list[str]:
    units: list[str] = []
    for line in text.splitlines():
        parts = re.split(r"(?<=[.!?])\s+", line.strip())
        units.extend(p for p in parts if p)
    return units


def critique_draft(draft_text: str, ledger: Ledger) -> Critique:
    issues: list[Issue] = []

    if LEAKED_FEEDBACK_MARKER in draft_text:
        issues.append(
            Issue(
                issue_type="leaked_internal_text",
                description="The draft repeats internal revision feedback instead of only containing the client message.",
                sentence="",
            )
        )

    balance_match = BALANCE_RE.search(draft_text)
    if not balance_match:
        issues.append(
            Issue(
                issue_type="missing_balance",
                description="No 'Total balance due' line found in the draft.",
                sentence="",
            )
        )
    else:
        claimed_balance = _parse_amount(balance_match.group(1))
        if abs(claimed_balance - ledger.balance) > 0.01:
            issues.append(
                Issue(
                    issue_type="wrong_balance",
                    description=(
                        f"Draft states total balance due as {claimed_balance:.2f}, "
                        f"but the ledger balance is {ledger.balance:.2f}."
                    ),
                    sentence=balance_match.group(0),
                    expected=f"{ledger.balance:.2f}",
                )
            )

    known_amounts = {round(abs(t.amount), 2) for t in ledger.transactions}
    known_amounts.add(round(ledger.balance, 2))

    remaining_text = draft_text
    if balance_match:
        remaining_text = draft_text[: balance_match.start()] + draft_text[balance_match.end() :]

    for sentence in _split_sentences(remaining_text):
        txn_ids = TXN_RE.findall(sentence)
        amounts = [_parse_amount(a) for a in AMOUNT_RE.findall(sentence)]

        if txn_ids:
            for txn_id in txn_ids:
                txn = ledger.find(txn_id)
                if txn is None:
                    issues.append(
                        Issue(
                            issue_type="fabricated_transaction",
                            description=f"{txn_id} is cited in the draft but does not exist in the ledger.",
                            sentence=sentence,
                        )
                    )
                    continue
                if amounts and not any(abs(a - abs(txn.amount)) <= 0.01 for a in amounts):
                    issues.append(
                        Issue(
                            issue_type="wrong_amount",
                            description=(
                                f"{txn_id} is actually {abs(txn.amount):.2f}, "
                                f"but the draft states {amounts} for it."
                            ),
                            sentence=sentence,
                            expected=f"{abs(txn.amount):.2f}",
                        )
                    )
        else:
            for a in amounts:
                if round(a, 2) not in known_amounts:
                    issues.append(
                        Issue(
                            issue_type="unsupported_claim",
                            description=(
                                f"Amount {a:.2f} does not match any ledger transaction or the "
                                f"total balance, and cites no transaction id."
                            ),
                            sentence=sentence,
                        )
                    )

    return Critique(issues=issues)
