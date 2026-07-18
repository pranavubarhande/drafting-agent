import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from maxed_agent.agent.injected_errors import build_injected_error_cases
from maxed_agent.agent.loop import run_loop
from maxed_agent.agent.verifier import critique_draft
from maxed_agent.fixtures.generate_fixtures import DEFAULT_CLIENT_ID, load_fixtures
from maxed_agent.learning.learning_cycle import _batch_stats, run_learning_cycle
from maxed_agent.model_client import ModelClient

load_dotenv()

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def status() -> None:
    console.print("maxed-agent scaffold OK. Generate, verify, learn, and eval commands land here next.")


@app.command()
def generate(client_id: str = DEFAULT_CLIENT_ID) -> None:
    client, ledger = load_fixtures(client_id)
    model_client = ModelClient()
    result = run_loop(client, ledger, model_client)

    console.print(f"[bold]Action:[/bold] {result.action} (attempts: {result.attempts})")
    if result.reason:
        console.print(f"[bold]Reason:[/bold] {result.reason}")
    console.print()
    console.print(result.draft_text)


@app.command(name="injected-errors")
def injected_errors(client_id: str = DEFAULT_CLIENT_ID) -> None:
    _, ledger = load_fixtures(client_id)
    cases = build_injected_error_cases(ledger)

    table = Table(title="Injected-error verifier results")
    table.add_column("Case")
    table.add_column("Injected issue")
    table.add_column("Caught?")
    table.add_column("Verifier finding")

    all_caught = True
    for case in cases:
        critique = critique_draft(case["text"], ledger)
        if case["expected_type"] is None:
            caught = not critique.has_issues
            finding = "no issues found" if not critique.has_issues else critique.issues[0].description
        else:
            caught = any(i.issue_type == case["expected_type"] for i in critique.issues)
            finding = critique.issues[0].description if critique.issues else "no issues found"
        all_caught = all_caught and caught
        table.add_row(case["name"], case["expected_type"] or "none (control)", "yes" if caught else "no", finding)

    console.print(table)
    if not all_caught:
        raise typer.Exit(code=1)


@app.command(name="learning-cycle")
def learning_cycle(batch_size: int = 3, client_id: str = DEFAULT_CLIENT_ID) -> None:
    model_client = ModelClient()
    report = run_learning_cycle(model_client, batch_size=batch_size, client_id=client_id)

    stats1 = _batch_stats(report.batch1)
    stats2 = _batch_stats(report.batch2)

    table = Table(title="Learning cycle: batch 1 (no memory) vs batch 2 (with retrieval memory)")
    table.add_column("Metric")
    table.add_column("Batch 1")
    table.add_column("Batch 2")
    table.add_row("Average attempts to settle", f"{stats1['avg_attempts']:.2f}", f"{stats2['avg_attempts']:.2f}")
    table.add_row("Approved", str(stats1["approved"]), str(stats2["approved"]))
    table.add_row("Escalated", str(stats1["escalate"]), str(stats2["escalate"]))
    table.add_row("Abstained", str(stats1["abstain"]), str(stats2["abstain"]))

    console.print(table)
    console.print()
    console.print("[bold]Retrieval memory injected into batch 2:[/bold]")
    console.print(report.memory_block if report.memory_block else "(none, batch 1 produced no approved examples)")


if __name__ == "__main__":
    app()
