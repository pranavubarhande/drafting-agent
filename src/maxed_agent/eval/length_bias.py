from pydantic import BaseModel

from maxed_agent.eval.judge import JUDGE_SYSTEM_PROMPT, JudgeScore, score_draft
from maxed_agent.fixtures.generate_fixtures import Ledger
from maxed_agent.model_client import ModelClient

NAIVE_JUDGE_SYSTEM_PROMPT = """You are scoring a client-facing accounting message.

completeness: how complete does this message feel overall, on a scale of 1 to 5.
tone: how professional does this message feel, on a scale of 1 to 5.

Respond with exactly this format:
completeness: <integer 1 to 5>
tone: <integer 1 to 5>
rationale: <one sentence>"""

THOROUGHNESS_PADDING = (
    "This statement represents a comprehensive, thorough, and complete accounting of your account "
    "activity, reflecting our full commitment to transparency and detailed financial reporting. We "
    "have carefully documented every aspect of your account in exhaustive detail to ensure you have "
    "a complete and thorough understanding of your current standing."
)


class LengthBiasComparison(BaseModel):
    sparse: JudgeScore
    padded: JudgeScore

    @property
    def completeness_gap(self) -> int:
        return self.padded.completeness - self.sparse.completeness

    @property
    def tone_gap(self) -> int:
        return self.padded.tone - self.sparse.tone


def build_padded_variant(sparse_draft: str) -> str:
    return f"{sparse_draft}\n\n{THOROUGHNESS_PADDING}"


def measure_with_prompt(
    sparse_draft: str,
    ledger: Ledger,
    model_client: ModelClient,
    system_prompt: str,
    task: str = "judge_score",
) -> LengthBiasComparison:
    padded_draft = build_padded_variant(sparse_draft)
    sparse_score = score_draft(sparse_draft, ledger, model_client, system_prompt=system_prompt, task=task)
    padded_score = score_draft(padded_draft, ledger, model_client, system_prompt=system_prompt, task=task)
    return LengthBiasComparison(sparse=sparse_score, padded=padded_score)


def measure_naive(sparse_draft: str, ledger: Ledger, model_client: ModelClient) -> LengthBiasComparison:
    return measure_with_prompt(sparse_draft, ledger, model_client, NAIVE_JUDGE_SYSTEM_PROMPT, task="naive_judge_score")


def measure_anchored(sparse_draft: str, ledger: Ledger, model_client: ModelClient) -> LengthBiasComparison:
    return measure_with_prompt(sparse_draft, ledger, model_client, JUDGE_SYSTEM_PROMPT, task="judge_score")
