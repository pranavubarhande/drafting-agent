from pydantic import BaseModel

from drafting_agent.eval.judge import JudgeScore, score_draft
from drafting_agent.fixtures.generate_fixtures import Ledger
from drafting_agent.learning.learning_cycle import LearningCycleReport
from drafting_agent.model_client import ModelClient


class BeforeAfterReport(BaseModel):
    before_scores: list[JudgeScore]
    after_scores: list[JudgeScore]

    @property
    def before_average(self) -> float:
        return sum(s.average for s in self.before_scores) / len(self.before_scores)

    @property
    def after_average(self) -> float:
        return sum(s.average for s in self.after_scores) / len(self.after_scores)

    @property
    def moved(self) -> bool:
        return abs(self.after_average - self.before_average) > 0.01


def score_before_after(
    report: LearningCycleReport,
    ledger: Ledger,
    model_client: ModelClient,
) -> BeforeAfterReport:
    before_scores = [score_draft(r.draft_text, ledger, model_client) for r in report.batch1]
    after_scores = [score_draft(r.draft_text, ledger, model_client) for r in report.batch2]
    return BeforeAfterReport(before_scores=before_scores, after_scores=after_scores)
