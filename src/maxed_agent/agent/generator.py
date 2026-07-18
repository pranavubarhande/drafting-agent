from typing import Optional

from maxed_agent.agent.verifier import LEAKED_FEEDBACK_MARKER
from maxed_agent.fixtures.generate_fixtures import Client, Ledger
from maxed_agent.model_client import ModelClient

SYSTEM_PROMPT = """You draft a message to a client explaining their current account balance due, on behalf of an accounting firm.

Rules you must follow exactly:
1. You must individually mention every transaction listed in the ledger given to you. Do not summarize or skip any of them, and do not just state the total balance on its own.
2. Every dollar amount you state must come directly from the ledger data given to you. Never invent a number.
3. Every time you mention a transaction, cite its transaction id in parentheses immediately after the dollar amount, for example: a payment of $50.00 (TXN-014).
4. Do not mention any transaction id that is not in the ledger given to you.
5. End the message with a line in exactly this format, with no other text on that line: Total balance due: $<amount>
6. Use the exact precomputed total balance given to you on that line. Do not compute it yourself.
7. Keep the tone professional and direct. Do not use em dashes."""

FEEDBACK_PREAMBLE = f"Your previous draft had the following problems ({LEAKED_FEEDBACK_MARKER}):\n"


def build_prompt(
    client: Client,
    ledger: Ledger,
    feedback: Optional[str] = None,
    memory_block: Optional[str] = None,
) -> str:
    lines = [
        f"{t.id}: {t.date} {t.type} amount={t.amount:.2f} description={t.description}"
        for t in ledger.transactions
    ]
    prompt = (
        f"Client: {client.name} ({client.email})\n"
        f"Precomputed total balance due: {ledger.balance:.2f}\n\n"
        f"Ledger:\n" + "\n".join(lines) + "\n\nWrite the client message now."
    )
    if memory_block:
        prompt = (
            "Here are examples of past messages a human already approved or corrected. "
            "Match their style and level of detail.\n\n" + memory_block + "\n\n" + prompt
        )
    if feedback:
        prompt += (
            "\n\n"
            + FEEDBACK_PREAMBLE
            + feedback
            + "\n\nNow write only the corrected client message itself. Do not repeat this "
            "feedback, do not mention that a previous draft existed, and do not include "
            "any text other than the client message."
        )
    return prompt


def generate_draft(
    client: Client,
    ledger: Ledger,
    model_client: ModelClient,
    feedback: Optional[str] = None,
    memory_block: Optional[str] = None,
) -> str:
    prompt = build_prompt(client, ledger, feedback=feedback, memory_block=memory_block)
    return model_client.complete(task="generate_draft", system=SYSTEM_PROMPT, prompt=prompt)
