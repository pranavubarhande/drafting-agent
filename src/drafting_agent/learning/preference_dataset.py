import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from drafting_agent.learning.decision_log import DEFAULT_LOG_PATH, load_decisions

DEFAULT_TRACE_PATH = Path("data/logs/trace.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/preference_dataset.jsonl")

EDITED_DECISIONS = {"approve_with_edit"}


class PreferencePair(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    source: str


def _load_generate_traces(trace_path: Path) -> list[dict]:
    if not trace_path.exists():
        return []
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    return [r for r in records if r.get("task") == "generate_draft"]


def _find_prompt(draft_text: str, generate_traces: list[dict]) -> Optional[str]:
    for t in generate_traces:
        if t.get("output") == draft_text:
            return t.get("prompt")
    return None


def build_preference_dataset(
    log_path: Path = DEFAULT_LOG_PATH,
    trace_path: Path = DEFAULT_TRACE_PATH,
) -> list[PreferencePair]:
    decisions = load_decisions(log_path)
    generate_traces = _load_generate_traces(trace_path)

    pairs: list[PreferencePair] = []
    for d in decisions:
        if d["human_decision"] not in EDITED_DECISIONS:
            continue
        if not d["final_text"] or d["final_text"] == d["draft_text"]:
            continue
        prompt = _find_prompt(d["draft_text"], generate_traces)
        if prompt is None:
            continue
        source = "escalate_corrected" if d["loop_action"] == "escalate" else "approved_with_cosmetic_edit"
        pairs.append(
            PreferencePair(prompt=prompt, chosen=d["final_text"], rejected=d["draft_text"], source=source)
        )
    return pairs


def write_preference_dataset(pairs: list[PreferencePair], path: Path = DEFAULT_OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")
