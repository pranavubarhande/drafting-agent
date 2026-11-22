import json

from drafting_agent.learning.decision_log import DecisionRecord, log_decision, now
from drafting_agent.learning.preference_dataset import build_preference_dataset, write_preference_dataset


def _write_trace(trace_path, prompt, output):
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a") as f:
        f.write(json.dumps({"task": "generate_draft", "prompt": prompt, "output": output}) + "\n")


def test_builds_pair_from_approve_with_edit(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    original_draft = "A payment of $525.00 (TXN-001) was received.\nTotal balance due: $1337.50"
    corrected = "A payment of $450.00 (TXN-001) was received.\nTotal balance due: $1337.50"
    _write_trace(trace_path, prompt="ledger context here", output=original_draft)

    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text=original_draft,
            loop_action="escalate",
            attempts=3,
            human_decision="approve_with_edit",
            final_text=corrected,
            edit_note="corrected wrong amount",
        ),
        path=log_path,
    )

    pairs = build_preference_dataset(log_path, trace_path)

    assert len(pairs) == 1
    assert pairs[0].prompt == "ledger context here"
    assert pairs[0].chosen == corrected
    assert pairs[0].rejected == original_draft
    assert pairs[0].source == "escalate_corrected"


def test_excludes_approve_as_is_and_decline(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    trace_path = tmp_path / "trace.jsonl"

    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text="clean draft",
            loop_action="approved",
            attempts=1,
            human_decision="approve_as_is",
            final_text="clean draft",
        ),
        path=log_path,
    )
    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text="bad draft",
            loop_action="abstain",
            attempts=3,
            human_decision="decline",
            final_text="",
        ),
        path=log_path,
    )

    pairs = build_preference_dataset(log_path, trace_path)
    assert pairs == []


def test_write_preference_dataset_round_trips(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    out_path = tmp_path / "preference_dataset.jsonl"

    _write_trace(trace_path, prompt="p", output="draft")
    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text="draft",
            loop_action="escalate",
            attempts=2,
            human_decision="approve_with_edit",
            final_text="fixed draft",
        ),
        path=log_path,
    )

    pairs = build_preference_dataset(log_path, trace_path)
    write_preference_dataset(pairs, out_path)

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["chosen"] == "fixed draft"
