import json
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

DEFAULT_LOG_PATH = Path("data/logs/decisions.jsonl")

RecordKind = Literal["decision", "capability_gap"]
LoopAction = Literal["approved", "abstain", "escalate"]
HumanDecision = Literal["approve_as_is", "approve_with_edit", "decline"]
GapType = Literal["missing_record", "missing_tool", "ambiguity"]


class DecisionRecord(BaseModel):
    kind: Literal["decision"] = "decision"
    timestamp: float
    client_id: str
    draft_text: str
    critique_issues: list[str] = []
    loop_action: LoopAction
    attempts: int
    human_decision: HumanDecision
    final_text: str
    edit_note: str = ""


class CapabilityGapRecord(BaseModel):
    kind: Literal["capability_gap"] = "capability_gap"
    timestamp: float
    client_id: str
    gap_type: GapType
    description: str
    related_issue: str = ""


def _append(record: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def log_decision(record: DecisionRecord, path: Path = DEFAULT_LOG_PATH) -> None:
    _append(record, path)


def log_capability_gap(record: CapabilityGapRecord, path: Path = DEFAULT_LOG_PATH) -> None:
    _append(record, path)


def load_records(path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_decisions(path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    return [r for r in load_records(path) if r.get("kind") == "decision"]


def load_capability_gaps(path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    return [r for r in load_records(path) if r.get("kind") == "capability_gap"]


def now() -> float:
    return time.time()
