import json
from pathlib import Path

from pydantic import BaseModel
from scipy.stats import spearmanr

from drafting_agent.agent.generator import FEEDBACK_PREAMBLE
from drafting_agent.agent.injected_errors import build_baseline_text, build_injected_error_cases
from drafting_agent.eval.judge import JudgeScore, score_draft
from drafting_agent.fixtures.generate_fixtures import Ledger
from drafting_agent.model_client import ModelClient

CANDIDATES_PATH = Path("data/eval/labeling_candidates.json")
HUMAN_LABELS_PATH = Path("data/eval/human_labels.json")

DIMENSIONS = ["factual_grounding", "absence_of_invented_numbers", "completeness", "tone"]


class LabelingCandidate(BaseModel):
    id: str
    source: str
    draft_text: str


def build_labeling_candidates(ledger: Ledger) -> list[LabelingCandidate]:
    candidates: list[LabelingCandidate] = []

    clean = build_baseline_text(ledger)
    candidates.append(LabelingCandidate(id="clean-full", source="clean, fully itemized", draft_text=clean))

    for case in build_injected_error_cases(ledger):
        if case["expected_type"] is None:
            continue
        candidates.append(
            LabelingCandidate(
                id=f"injected-{case['expected_type']}",
                source=case["name"],
                draft_text=case["text"],
            )
        )

    sparse_txn = ledger.transactions[1]
    sparse = f"Total balance due: ${ledger.balance:.2f}"
    candidates.append(LabelingCandidate(id="sparse-bare-total", source="only the total, no itemization", draft_text=sparse))

    one_line = (
        f"A payment of ${abs(sparse_txn.amount):.2f} ({sparse_txn.id}) was received.\n"
        f"Total balance due: ${ledger.balance:.2f}"
    )
    candidates.append(
        LabelingCandidate(id="sparse-one-citation", source="a single citation, otherwise bare", draft_text=one_line)
    )

    half = ledger.transactions[: len(ledger.transactions) // 2]
    partial_lines = "\n".join(
        f"- {t.id}: ${abs(t.amount):.2f} - {t.description}" for t in half
    )
    partial = f"{partial_lines}\nTotal balance due: ${ledger.balance:.2f}"
    candidates.append(
        LabelingCandidate(id="partial-half-coverage", source="half the transactions itemized", draft_text=partial)
    )

    padded = clean + (
        "\n\nThank you so much for being a valued customer, we truly appreciate your business "
        "and look forward to continuing to serve you in the future."
    )
    candidates.append(LabelingCandidate(id="clean-plus-filler", source="fully itemized plus generic filler", draft_text=padded))

    leaked = FEEDBACK_PREAMBLE + "some prior issue\n\n" + clean
    candidates.append(
        LabelingCandidate(id="leaked-internal-text", source="fully itemized but leaks revision feedback", draft_text=leaked)
    )

    casual_txn = ledger.transactions[2]
    casual_lines = "\n".join(
        f"- {t.id}: ${abs(t.amount):.2f} - {t.description}" for t in ledger.transactions[:8]
    )
    casual = (
        f"hey so u owe us some money lol, details below, lmk if this looks weird:\n"
        f"{casual_lines}\n"
        f"Total balance due: ${ledger.balance:.2f}"
    )
    candidates.append(
        LabelingCandidate(id="accurate-but-casual-tone", source="factually correct, unprofessional tone", draft_text=casual)
    )

    two_issue_txn = ledger.transactions[3]
    two_issues = (
        f"A charge of ${abs(two_issue_txn.amount) + 40:.2f} ({two_issue_txn.id}) was recorded.\n"
        f"An additional miscellaneous processing fee of $37.00 was also applied.\n"
        f"Total balance due: ${ledger.balance:.2f}"
    )
    candidates.append(
        LabelingCandidate(id="wrong-amount-plus-unsupported-claim", source="two distinct issues at once", draft_text=two_issues)
    )

    return candidates


def write_candidates_for_labeling(ledger: Ledger, path: Path = CANDIDATES_PATH) -> list[LabelingCandidate]:
    candidates = build_labeling_candidates(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": c.id,
            "source": c.source,
            "draft_text": c.draft_text,
            "human_scores": {dim: None for dim in DIMENSIONS},
        }
        for c in candidates
    ]
    path.write_text(json.dumps(payload, indent=2))
    return candidates


def load_human_labels(path: Path = HUMAN_LABELS_PATH) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {item["id"]: item["human_scores"] for item in raw}


def compute_agreement(
    candidates: list[LabelingCandidate],
    human_labels: dict[str, dict[str, int]],
    ledger: Ledger,
    model_client: ModelClient,
) -> dict:
    judge_scores: dict[str, JudgeScore] = {}
    labeled_ids = [c.id for c in candidates if c.id in human_labels]

    for c in candidates:
        if c.id in human_labels:
            judge_scores[c.id] = score_draft(c.draft_text, ledger, model_client)

    per_dimension: dict[str, dict] = {}
    for dim in DIMENSIONS:
        human_values = [human_labels[cid][dim] for cid in labeled_ids]
        judge_values = [getattr(judge_scores[cid], dim) for cid in labeled_ids]
        if len(set(human_values)) < 2 or len(set(judge_values)) < 2:
            correlation = None
        else:
            correlation, _ = spearmanr(human_values, judge_values)
        exact = sum(1 for h, j in zip(human_values, judge_values) if h == j) / len(labeled_ids)
        near = sum(1 for h, j in zip(human_values, judge_values) if abs(h - j) <= 1) / len(labeled_ids)
        per_dimension[dim] = {
            "spearman": correlation,
            "exact_agreement": exact,
            "near_agreement": near,
            "n": len(labeled_ids),
        }

    return per_dimension
