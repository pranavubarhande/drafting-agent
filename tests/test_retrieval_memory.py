from maxed_agent.learning.decision_log import DecisionRecord, log_decision, now
from maxed_agent.learning.retrieval_memory import get_approved_examples


def test_sparse_approved_example_is_excluded_from_memory(tmp_path):
    log_path = tmp_path / "decisions.jsonl"

    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text="x",
            loop_action="approved",
            attempts=1,
            human_decision="approve_as_is",
            final_text="Your balance is due. Total balance due: $100.00",
        ),
        path=log_path,
    )
    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text="x",
            loop_action="approved",
            attempts=1,
            human_decision="approve_as_is",
            final_text="A payment of $50.00 (TXN-001) was received. Total balance due: $100.00",
        ),
        path=log_path,
    )

    examples = get_approved_examples("CLIENT-001", log_path, limit=5)
    assert len(examples) == 1
    assert "TXN-001" in examples[0]["final_text"]


def test_memory_is_scoped_to_client_and_never_leaks_across_clients(tmp_path):
    log_path = tmp_path / "decisions.jsonl"

    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-001",
            draft_text="x",
            loop_action="approved",
            attempts=1,
            human_decision="approve_as_is",
            final_text="A payment of $450.00 (TXN-001) was received. Total balance due: $1337.50",
        ),
        path=log_path,
    )
    log_decision(
        DecisionRecord(
            timestamp=now(),
            client_id="CLIENT-002",
            draft_text="x",
            loop_action="approved",
            attempts=1,
            human_decision="approve_as_is",
            final_text="A payment of $600.00 (TXN-029) was received. Total balance due: $735.00",
        ),
        path=log_path,
    )

    client_two_examples = get_approved_examples("CLIENT-002", log_path, limit=5)
    assert len(client_two_examples) == 1
    assert "TXN-029" in client_two_examples[0]["final_text"]
    assert "TXN-001" not in client_two_examples[0]["final_text"]

    client_one_examples = get_approved_examples("CLIENT-001", log_path, limit=5)
    assert len(client_one_examples) == 1
    assert "TXN-001" in client_one_examples[0]["final_text"]
