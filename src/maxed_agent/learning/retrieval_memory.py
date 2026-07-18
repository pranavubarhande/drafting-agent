from pathlib import Path

from maxed_agent.agent.verifier import count_txn_citations
from maxed_agent.learning.decision_log import DEFAULT_LOG_PATH, load_decisions

APPROVED_DECISIONS = {"approve_as_is", "approve_with_edit"}
MIN_TXN_CITATIONS = 1


def _meets_quality_bar(final_text: str) -> bool:
    return count_txn_citations(final_text) >= MIN_TXN_CITATIONS


def get_approved_examples(client_id: str, path: Path = DEFAULT_LOG_PATH, limit: int = 3) -> list[dict]:
    decisions = load_decisions(path)
    approved = [
        d
        for d in decisions
        if d.get("client_id") == client_id
        and d.get("human_decision") in APPROVED_DECISIONS
        and d.get("final_text")
        and _meets_quality_bar(d["final_text"])
    ]
    return approved[-limit:]


def build_few_shot_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    blocks = [
        f"Example {i} (human-approved, final text):\n{ex['final_text']}"
        for i, ex in enumerate(examples, 1)
    ]
    return "\n\n".join(blocks)
