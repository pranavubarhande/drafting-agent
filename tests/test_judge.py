from drafting_agent.agent.injected_errors import build_baseline_text, build_injected_error_cases
from drafting_agent.eval.judge import score_draft
from drafting_agent.fixtures.generate_fixtures import load_fixtures
from drafting_agent.model_client import ModelClient


def _stub_judge_client(response: str) -> ModelClient:
    return ModelClient(provider="stub", stub_responses={"judge_score": [response]})


def test_clean_draft_scores_perfect_on_code_derived_dimensions():
    _, ledger = load_fixtures()
    draft = build_baseline_text(ledger)
    model_client = _stub_judge_client("completeness: 4\ntone: 5\nrationale: clear and direct")

    score = score_draft(draft, ledger, model_client)

    assert score.factual_grounding == 5
    assert score.absence_of_invented_numbers == 5
    assert score.completeness == 4
    assert score.tone == 5


def test_fabricated_transaction_lowers_invented_numbers_score_not_tone():
    _, ledger = load_fixtures()
    cases = {c["name"]: c["text"] for c in build_injected_error_cases(ledger)}
    model_client = _stub_judge_client("completeness: 4\ntone: 4\nrationale: reads fine")

    score = score_draft(cases["fabricated transaction"], ledger, model_client)

    assert score.absence_of_invented_numbers < 5
    assert score.factual_grounding < 5
    assert score.tone == 4, "tone must come from the model, not be affected by grounding issues"


def test_unparseable_judge_output_falls_back_to_neutral_score():
    _, ledger = load_fixtures()
    draft = build_baseline_text(ledger)
    model_client = _stub_judge_client("the model said something unexpected")

    score = score_draft(draft, ledger, model_client)

    assert score.completeness == 3
    assert score.tone == 3
    assert "no rationale parsed" in score.rationale
