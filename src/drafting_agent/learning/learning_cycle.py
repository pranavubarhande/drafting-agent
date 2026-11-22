from pathlib import Path
from typing import NamedTuple

from drafting_agent.agent.loop import LoopResult, run_loop
from drafting_agent.fixtures.generate_fixtures import DEFAULT_CLIENT_ID, load_fixtures
from drafting_agent.learning.decision_log import DEFAULT_LOG_PATH
from drafting_agent.learning.human_sim import simulate_human
from drafting_agent.learning.retrieval_memory import build_few_shot_block, get_approved_examples
from drafting_agent.model_client import ModelClient


class LearningCycleReport(NamedTuple):
    batch1: list[LoopResult]
    batch2: list[LoopResult]
    memory_block: str


def _batch_stats(batch: list[LoopResult]) -> dict:
    n = len(batch)
    avg_attempts = sum(r.attempts for r in batch) / n
    approved = sum(1 for r in batch if r.action == "approved")
    escalated = sum(1 for r in batch if r.action == "escalate")
    abstained = sum(1 for r in batch if r.action == "abstain")
    return {
        "avg_attempts": avg_attempts,
        "approved": approved,
        "escalate": escalated,
        "abstain": abstained,
        "n": n,
    }


def run_learning_cycle(
    model_client: ModelClient,
    batch_size: int = 3,
    log_path: Path = DEFAULT_LOG_PATH,
    client_id: str = DEFAULT_CLIENT_ID,
    reset_log: bool = True,
) -> LearningCycleReport:
    client, ledger = load_fixtures(client_id)

    if reset_log and log_path.exists():
        log_path.unlink()

    batch1: list[LoopResult] = []
    for i in range(batch_size):
        result = run_loop(client, ledger, model_client)
        simulate_human(client.id, result, index=i, log_path=log_path)
        batch1.append(result)

    examples = get_approved_examples(client.id, log_path, limit=batch_size)
    memory_block = build_few_shot_block(examples)

    batch2: list[LoopResult] = []
    for i in range(batch_size):
        result = run_loop(client, ledger, model_client, memory_block=memory_block)
        simulate_human(client.id, result, index=batch_size + i, log_path=log_path)
        batch2.append(result)

    return LearningCycleReport(batch1=batch1, batch2=batch2, memory_block=memory_block)
