from drafting_agent.agent.loop import LoopResult
from drafting_agent.agent.verifier import Critique
from drafting_agent.eval.before_after import score_before_after
from drafting_agent.fixtures.generate_fixtures import load_fixtures
from drafting_agent.learning.learning_cycle import LearningCycleReport
from drafting_agent.model_client import ModelClient


def _approved(draft_text: str) -> LoopResult:
    return LoopResult(action="approved", draft_text=draft_text, critique=Critique(issues=[]), attempts=1)


def test_score_before_after_reports_honest_movement_or_lack_of_it():
    _, ledger = load_fixtures()
    sparse = f"Total balance due: ${ledger.balance:.2f}"
    detailed = (
        f"A payment of ${abs(ledger.transactions[0].amount):.2f} ({ledger.transactions[0].id}) was received.\n"
        f"Total balance due: ${ledger.balance:.2f}"
    )
    report = LearningCycleReport(batch1=[_approved(sparse)], batch2=[_approved(detailed)], memory_block="")

    model_client = ModelClient(
        provider="stub",
        stub_responses={
            "judge_score": [
                "completeness: 2\ntone: 4\nrationale: bare total only",
                "completeness: 4\ntone: 4\nrationale: itemizes one transaction",
            ]
        },
    )

    result = score_before_after(report, ledger, model_client)

    assert result.before_average < result.after_average
    assert result.moved
