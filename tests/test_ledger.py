from drafting_agent.fixtures.generate_fixtures import load_fixtures


def test_ledger_balance_matches_manual_sum():
    _, ledger = load_fixtures()
    assert ledger.balance == 1337.50


def test_ledger_has_expected_transaction_count():
    _, ledger = load_fixtures()
    assert len(ledger.transactions) == 28


def test_ledger_find_returns_none_for_unknown_id():
    _, ledger = load_fixtures()
    assert ledger.find("TXN-999") is None
    assert ledger.find("TXN-001") is not None
