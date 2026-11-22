import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from drafting_agent.agent.injected_errors import build_injected_error_cases
from drafting_agent.agent.loop import run_loop
from drafting_agent.agent.verifier import critique_draft
from drafting_agent.eval.before_after import score_before_after
from drafting_agent.eval.calibration import build_labeling_candidates, compute_agreement, load_human_labels
from drafting_agent.eval.length_bias import measure_anchored, measure_naive
from drafting_agent.fixtures.generate_fixtures import DEFAULT_CLIENT_ID, load_fixtures
from drafting_agent.learning.decision_log import DEFAULT_LOG_PATH, load_decisions, veto_memory_example
from drafting_agent.learning.escalation_calibration import calibrate_abstain_escalate
from drafting_agent.learning.learning_cycle import _batch_stats, run_learning_cycle
from drafting_agent.learning.preference_dataset import build_preference_dataset, write_preference_dataset
from drafting_agent.model_client import ModelClient

load_dotenv()

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def status() -> None:
    console.print("drafting-agent ready. Run --help to see available commands.")


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


@app.command()
def calibration(client_id: str = DEFAULT_CLIENT_ID) -> None:
    _, ledger = load_fixtures(client_id)
    candidates = build_labeling_candidates(ledger)
    human_labels = load_human_labels()

    if not human_labels:
        console.print("No human labels found at data/eval/human_labels.json. Nothing to calibrate against.")
        raise typer.Exit(code=1)

    model_client = ModelClient()
    result = compute_agreement(candidates, human_labels, ledger, model_client)

    table = Table(title=f"Judge calibration against {len(human_labels)} hand-labeled drafts")
    table.add_column("Dimension")
    table.add_column("Spearman")
    table.add_column("Exact agreement")
    table.add_column("Within 1 point")
    for dim, stats in result.items():
        spearman = f"{stats['spearman']:.2f}" if stats["spearman"] is not None else "n/a (no variance)"
        table.add_row(dim, spearman, f"{stats['exact_agreement']:.0%}", f"{stats['near_agreement']:.0%}")

    console.print(table)


@app.command(name="length-bias")
def length_bias(client_id: str = DEFAULT_CLIENT_ID) -> None:
    _, ledger = load_fixtures(client_id)
    txn = ledger.transactions[1]
    sparse_draft = f"A payment of ${abs(txn.amount):.2f} ({txn.id}) was received.\nTotal balance due: ${ledger.balance:.2f}"

    model_client = ModelClient()
    naive_result = measure_naive(sparse_draft, ledger, model_client)
    anchored_result = measure_anchored(sparse_draft, ledger, model_client)

    table = Table(title="Length bias: naive rubric vs anchored rubric, same sparse-plus-padding pair")
    table.add_column("Prompt")
    table.add_column("Sparse completeness/tone")
    table.add_column("Padded completeness/tone")
    table.add_column("Completeness gap")
    table.add_column("Tone gap")
    table.add_row(
        "naive (vague)",
        f"{naive_result.sparse.completeness}/{naive_result.sparse.tone}",
        f"{naive_result.padded.completeness}/{naive_result.padded.tone}",
        str(naive_result.completeness_gap),
        str(naive_result.tone_gap),
    )
    table.add_row(
        "anchored (mine)",
        f"{anchored_result.sparse.completeness}/{anchored_result.sparse.tone}",
        f"{anchored_result.padded.completeness}/{anchored_result.padded.tone}",
        str(anchored_result.completeness_gap),
        str(anchored_result.tone_gap),
    )

    console.print(table)


@app.command(name="before-after")
def before_after(batch_size: int = 3, client_id: str = DEFAULT_CLIENT_ID) -> None:
    _, ledger = load_fixtures(client_id)
    model_client = ModelClient()
    report = run_learning_cycle(model_client, batch_size=batch_size, client_id=client_id)
    result = score_before_after(report, ledger, model_client)

    table = Table(title="Before (no memory) vs after (with retrieval memory), judge-scored")
    table.add_column("")
    table.add_column("Before")
    table.add_column("After")
    for dim in ("factual_grounding", "absence_of_invented_numbers", "completeness", "tone"):
        before_avg = sum(getattr(s, dim) for s in result.before_scores) / len(result.before_scores)
        after_avg = sum(getattr(s, dim) for s in result.after_scores) / len(result.after_scores)
        table.add_row(dim, f"{before_avg:.2f}", f"{after_avg:.2f}")
    table.add_row("overall average", f"{result.before_average:.2f}", f"{result.after_average:.2f}")

    console.print(table)
    console.print()
    console.print(f"[bold]Moved:[/bold] {'yes' if result.moved else 'no'}")


@app.command(name="escalation-calibration")
def escalation_calibration() -> None:
    report = calibrate_abstain_escalate()

    console.print(f"[bold]Escalate cases:[/bold] {report.escalate_count}")
    if report.escalate_correction_success_rate is not None:
        resolved = report.escalate_corrections_clean
        total = report.escalate_count
        rate = report.escalate_correction_success_rate
        console.print(f"  corrections that fully resolved the draft: {resolved}/{total} ({rate:.0%})")
    else:
        console.print("  no escalate cases in the current log to check")

    console.print(f"\n[bold]Abstain cases:[/bold] {report.abstain_count}")
    if report.abstain_reason_breakdown:
        table = Table(title="Abstain reasons")
        table.add_column("Uncorrectable issue type")
        table.add_column("Count")
        for reason, count in sorted(report.abstain_reason_breakdown.items(), key=lambda kv: -kv[1]):
            table.add_row(reason, str(count))
        console.print(table)
    else:
        console.print("  no abstain cases in the current log to check")


@app.command(name="preference-dataset")
def preference_dataset() -> None:
    pairs = build_preference_dataset()
    write_preference_dataset(pairs)

    console.print(f"[bold]Preference pairs built:[/bold] {len(pairs)}")
    console.print("written to data/eval/preference_dataset.jsonl")
    if pairs:
        console.print()
        console.print(f"[bold]Sample ({pairs[0].source}):[/bold]")
        console.print(f"[bold]rejected:[/bold] {pairs[0].rejected[:200]}")
        console.print(f"[bold]chosen:[/bold]   {pairs[0].chosen[:200]}")


@app.command(name="list-decisions")
def list_decisions() -> None:
    decisions = load_decisions(DEFAULT_LOG_PATH)

    if not decisions:
        console.print("No decisions in the current log.")
        return

    table = Table(title="Decisions in the current log")
    table.add_column("Id")
    table.add_column("Client")
    table.add_column("Action")
    table.add_column("Human decision")
    table.add_column("Memory veto")
    table.add_column("Final text preview")
    for d in decisions:
        preview = (d.get("final_text") or "")[:60].replace("\n", " ")
        table.add_row(
            d.get("id", "(no id, logged before this feature)"),
            d["client_id"],
            d["loop_action"],
            d["human_decision"],
            "yes" if d.get("memory_veto") else "no",
            preview,
        )
    console.print(table)


@app.command(name="veto-memory")
def veto_memory(decision_id: str) -> None:
    found = veto_memory_example(decision_id, DEFAULT_LOG_PATH)
    if found:
        message = f"Decision {decision_id} will no longer be used as a retrieval-memory example."
        console.print(message)
        console.print("The draft itself is untouched; this only affects future memory retrieval.")
    else:
        console.print(f"No decision with id {decision_id} found in the current log.")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
