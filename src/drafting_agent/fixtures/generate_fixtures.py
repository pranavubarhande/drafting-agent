import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, TypeAdapter

DATA_DIR = Path(__file__).parent / "data"
CLIENTS_PATH = DATA_DIR / "clients.json"
LEDGERS_PATH = DATA_DIR / "ledgers.json"

DEFAULT_CLIENT_ID = "CLIENT-001"


class Transaction(BaseModel):
    id: str
    date: str
    type: Literal["invoice", "payment", "credit", "adjustment"]
    description: str
    amount: float


class Client(BaseModel):
    id: str
    name: str
    email: str


class Ledger(BaseModel):
    client_id: str
    transactions: list[Transaction]

    @property
    def balance(self) -> float:
        return round(sum(t.amount for t in self.transactions), 2)

    def find(self, txn_id: str) -> Optional[Transaction]:
        return next((t for t in self.transactions if t.id == txn_id), None)


def build_client_one() -> Client:
    return Client(
        id="CLIENT-001",
        name="Brightleaf Home Goods",
        email="accounts@brightleafhomegoods.example",
    )


def build_ledger_one(client: Client) -> Ledger:
    raw = [
        ("TXN-001", "2026-01-05", "invoice", "January service fee", 450.00),
        ("TXN-002", "2026-01-20", "payment", "Payment received, check #1042", -450.00),
        ("TXN-003", "2026-02-05", "invoice", "February service fee", 450.00),
        ("TXN-004", "2026-02-18", "payment", "Partial payment, ACH", -300.00),
        ("TXN-005", "2026-02-25", "adjustment", "Late fee assessed", 25.00),
        ("TXN-006", "2026-03-01", "invoice", "March service fee", 450.00),
        ("TXN-007", "2026-03-10", "credit", "Billing correction credit", -50.00),
        ("TXN-008", "2026-03-15", "payment", "Payment received, card", -400.00),
        ("TXN-009", "2026-03-22", "invoice", "Supplemental consulting hours", 320.00),
        ("TXN-010", "2026-04-01", "invoice", "April service fee", 450.00),
        ("TXN-011", "2026-04-01", "invoice", "Software license renewal", 325.00),
        ("TXN-012", "2026-04-14", "payment", "Payment received, check #1077", -450.00),
        ("TXN-013", "2026-04-20", "credit", "Refund, returned equipment", -180.00),
        ("TXN-014", "2026-05-01", "invoice", "May service fee", 450.00),
        ("TXN-015", "2026-05-01", "invoice", "May service fee, satellite office", 450.00),
        ("TXN-016", "2026-05-12", "payment", "Payment received, ACH", -450.00),
        ("TXN-017", "2026-05-12", "payment", "Payment received, ACH, satellite office", -450.00),
        ("TXN-018", "2026-05-28", "adjustment", "Rush processing fee", 60.00),
        ("TXN-019", "2026-06-01", "invoice", "June service fee", 450.00),
        ("TXN-020", "2026-06-01", "invoice", "June service fee, satellite office", 450.00),
        ("TXN-021", "2026-06-10", "payment", "Payment received, check #1103", -450.00),
        ("TXN-022", "2026-06-15", "credit", "Loyalty discount applied", -40.00),
        ("TXN-023", "2026-06-18", "invoice", "Onboarding fee, new integration", 275.00),
        ("TXN-024", "2026-06-20", "payment", "Partial payment, ACH", -200.00),
        ("TXN-025", "2026-06-25", "adjustment", "Late fee assessed", 25.00),
        ("TXN-026", "2026-06-27", "invoice", "July service fee, prepay", 450.00),
        ("TXN-027", "2026-06-28", "credit", "Prepayment discount", -22.50),
        ("TXN-028", "2026-06-30", "payment", "Payment received, card", -300.00),
    ]
    transactions = [
        Transaction(id=i, date=d, type=t, description=desc, amount=amt)
        for i, d, t, desc, amt in raw
    ]
    return Ledger(client_id=client.id, transactions=transactions)


def build_client_two() -> Client:
    return Client(
        id="CLIENT-002",
        name="Marlow and Finch Design Studio",
        email="billing@marlowfinchdesign.example",
    )


def build_ledger_two(client: Client) -> Ledger:
    raw = [
        ("TXN-029", "2026-02-01", "invoice", "February design retainer", 600.00),
        ("TXN-030", "2026-02-15", "payment", "Payment received, wire transfer", -600.00),
        ("TXN-031", "2026-03-01", "invoice", "March design retainer", 600.00),
        ("TXN-032", "2026-03-12", "payment", "Partial payment, check #221", -400.00),
        ("TXN-033", "2026-03-20", "adjustment", "Rush revision fee", 85.00),
        ("TXN-034", "2026-04-01", "invoice", "April design retainer", 600.00),
        ("TXN-035", "2026-04-05", "credit", "Discount, referral program", -60.00),
        ("TXN-036", "2026-04-18", "payment", "Payment received, wire transfer", -600.00),
        ("TXN-037", "2026-04-22", "invoice", "Additional asset delivery", 210.00),
        ("TXN-038", "2026-05-01", "invoice", "May design retainer", 600.00),
        ("TXN-039", "2026-05-10", "payment", "Payment received, check #235", -500.00),
        ("TXN-040", "2026-05-20", "adjustment", "Late fee assessed", 30.00),
        ("TXN-041", "2026-06-01", "invoice", "June design retainer", 600.00),
        ("TXN-042", "2026-06-01", "invoice", "June design retainer, secondary brand", 600.00),
        ("TXN-043", "2026-06-14", "payment", "Payment received, wire transfer", -600.00),
        ("TXN-044", "2026-06-14", "payment", "Payment received, wire transfer, secondary brand", -600.00),
        ("TXN-045", "2026-06-20", "credit", "Billing correction credit", -45.00),
        ("TXN-046", "2026-06-25", "invoice", "Brand refresh package", 950.00),
        ("TXN-047", "2026-06-28", "payment", "Partial payment, ACH", -700.00),
        ("TXN-048", "2026-06-30", "credit", "Loyalty discount applied", -35.00),
    ]
    transactions = [
        Transaction(id=i, date=d, type=t, description=desc, amount=amt)
        for i, d, t, desc, amt in raw
    ]
    return Ledger(client_id=client.id, transactions=transactions)


def build_all() -> tuple[list[Client], list[Ledger]]:
    client_one = build_client_one()
    client_two = build_client_two()
    clients = [client_one, client_two]
    ledgers = [build_ledger_one(client_one), build_ledger_two(client_two)]
    return clients, ledgers


def write_fixtures() -> list[Ledger]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clients, ledgers = build_all()
    clients_adapter = TypeAdapter(list[Client])
    ledgers_adapter = TypeAdapter(list[Ledger])
    CLIENTS_PATH.write_bytes(clients_adapter.dump_json(clients, indent=2))
    LEDGERS_PATH.write_bytes(ledgers_adapter.dump_json(ledgers, indent=2))
    return ledgers


def load_fixtures(client_id: str = DEFAULT_CLIENT_ID) -> tuple[Client, Ledger]:
    if not CLIENTS_PATH.exists() or not LEDGERS_PATH.exists():
        write_fixtures()
    clients = TypeAdapter(list[Client]).validate_json(CLIENTS_PATH.read_text())
    ledgers = TypeAdapter(list[Ledger]).validate_json(LEDGERS_PATH.read_text())
    client = next(c for c in clients if c.id == client_id)
    ledger = next(l for l in ledgers if l.client_id == client_id)
    return client, ledger


def load_all_client_ids() -> list[str]:
    if not CLIENTS_PATH.exists():
        write_fixtures()
    clients = TypeAdapter(list[Client]).validate_json(CLIENTS_PATH.read_text())
    return [c.id for c in clients]


if __name__ == "__main__":
    ledgers = write_fixtures()
    for ledger in ledgers:
        print(f"wrote fixtures for {ledger.client_id}, balance = {ledger.balance:.2f}")
