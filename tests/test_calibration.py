from drafting_agent.eval.calibration import build_labeling_candidates, compute_agreement
from drafting_agent.fixtures.generate_fixtures import load_fixtures
from drafting_agent.model_client import ModelClient


def test_build_labeling_candidates_has_at_least_ten_with_unique_ids():
    _, ledger = load_fixtures()
    candidates = build_labeling_candidates(ledger)

    assert len(candidates) >= 10
    ids = [c.id for c in candidates]
    assert len(ids) == len(set(ids)), "candidate ids must be unique"


def test_compute_agreement_reports_perfect_agreement_when_labels_match_judge():
    _, ledger = load_fixtures()
    candidates = build_labeling_candidates(ledger)[:3]

    human_labels = {
        c.id: {"factual_grounding": 5, "absence_of_invented_numbers": 5, "completeness": 3, "tone": 4}
        for c in candidates
    }
    model_client = ModelClient(
        provider="stub",
        stub_responses={"judge_score": ["completeness: 3\ntone: 4\nrationale: matches"] * 3},
    )

    result = compute_agreement(candidates, human_labels, ledger, model_client)

    assert result["completeness"]["exact_agreement"] == 1.0
    assert result["tone"]["exact_agreement"] == 1.0
    assert result["completeness"]["n"] == 3


def test_compute_agreement_only_uses_labeled_candidates():
    _, ledger = load_fixtures()
    candidates = build_labeling_candidates(ledger)

    human_labels = {
        candidates[0].id: {"factual_grounding": 5, "absence_of_invented_numbers": 5, "completeness": 4, "tone": 4},
        candidates[1].id: {"factual_grounding": 1, "absence_of_invented_numbers": 1, "completeness": 2, "tone": 3},
    }
    model_client = ModelClient(
        provider="stub",
        stub_responses={
            "judge_score": [
                "completeness: 4\ntone: 4\nrationale: a",
                "completeness: 2\ntone: 3\nrationale: b",
            ]
        },
    )

    result = compute_agreement(candidates, human_labels, ledger, model_client)

    assert result["completeness"]["n"] == 2
