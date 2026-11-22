# drafting-agent

A self-checking drafting agent. It writes a client message explaining an account balance due,
verifies its own draft against the real ledger records, and lands on exactly one outcome: revise,
abstain, or escalate. Every human decision on a draft is captured as structured signal, and a
retrieval-memory mechanism uses that signal to guide future drafts.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

`.env` controls `MODEL_PROVIDER`:

- `stub` (default): canned responses, no credentials, no network.
- `ollama`: a local model via an OpenAI-compatible endpoint (`gemma2:9b` by default, `gemma2:2b` as a faster option).
- `gemini`: hosted, also OpenAI-compatible.

## Commands

```bash
pytest                              # full test suite, stub mode
drafting-agent generate             # one generate -> verify -> decide cycle against the fixture ledger
drafting-agent injected-errors      # corrupts drafts on purpose, reports caught vs missed
drafting-agent learning-cycle       # batch (no memory) -> decide -> ingest -> batch (with memory)
drafting-agent calibration          # judge vs data/eval/human_labels.json, agreement numbers
drafting-agent length-bias          # naive rubric vs anchored rubric on the same padded pair
drafting-agent before-after         # judge-scores learning-cycle's batch 1 vs batch 2
drafting-agent escalation-calibration  # validates the abstain/escalate rule against real decisions
drafting-agent preference-dataset   # builds a DPO-shaped dataset from the decision log
drafting-agent list-decisions       # shows every decision, its id, and its memory-veto status
drafting-agent veto-memory <id>     # excludes a decision from future retrieval memory
```

Most commands take `--client-id` (default `CLIENT-001`; `CLIENT-002` is the second fixture client).
`calibration`, `length-bias`, and `before-after` need a real model configured in `.env`, since stub
responses do not vary with the prompt. `escalation-calibration`, `preference-dataset`, and
`list-decisions` need no model; they only read `data/logs/decisions.jsonl`. Run `learning-cycle`
first if that log is empty.

## Architecture

**Fixtures.** Two synthetic clients, each with a ledger (`src/drafting_agent/fixtures/`), generated
by a deterministic script. Transaction ids are globally unique across ledgers so a leaked
cross-client reference is unambiguous to detect. Each balance is computed in code from the
transactions and is the single source of truth the verifier checks against.

**Self-checking agent.** `generator.py` prompts the model to mention every transaction with its id
cited next to its dollar amount, ending in a fixed `Total balance due: $X` line. `verifier.py` is
pure code, with no second model call: it extracts every cited transaction id and amount and checks
each against the ledger. `loop.py` revises up to twice, then escalates (a real transaction exists
but a number is off, so a human can quickly confirm a fix) or abstains (the claim has no supporting
record at all). `injected-errors` demonstrates the verifier catches a wrong amount, a fabricated
transaction, an unsupported claim, and a wrong total, with a clean control producing no false
positive.

**Learning loop.** `decision_log.py` defines structured records: a `DecisionRecord` per draft
(source context, draft, critique, human decision, final text) and a `CapabilityGapRecord` whenever
the agent abstains or escalates. `human_sim.py` simulates the human: approve, edit and approve, or
decline. `retrieval_memory.py` feeds approved examples back as few-shot guidance, scoped per client
id so one client's examples never leak into another's. `learning-cycle` runs a batch without
memory, logs decisions, builds memory from qualifying approved examples, runs a second batch with
that memory, and reports both side by side.

**Human veto on memory.** Approving a draft ("send this") is not the same as endorsing it as a
teaching example. Every decision record carries a short id and a `memory_veto` flag (default
false). `list-decisions` and `veto-memory <id>` let a human revoke a draft's eligibility for
retrieval without altering the draft itself.

**Eval harness.**
- `judge.py` scores drafts on four dimensions. `factual_grounding` and `absence_of_invented_numbers`
  are computed in code from the verifier critique; `completeness` and `tone` go to the model, with
  verified facts supplied as trusted context.
- `calibration.py` scores a 12-draft set spanning quality tiers against hand labels, reporting
  Spearman correlation and near-agreement per dimension.
- `length_bias.py` compares a naive completeness prompt against the anchored one in `judge.py` on a
  sparse draft padded with filler, to check for length bias.
- `before_after.py` judge-scores batch 1 and batch 2 of a learning cycle.

**Analysis tools.**
- `escalation_calibration.py` checks the abstain-versus-escalate rule against the decision log:
  it re-verifies human-corrected text for escalations and tallies which issue types drive abstentions.
- `preference_dataset.py` turns every `approve_with_edit` decision into a `{prompt, chosen, rejected}`
  triple (the shape a DPO pass consumes) and writes `data/eval/preference_dataset.jsonl`.

## Design notes

- "Grounded" means every dollar amount and transaction reference in a draft matches the ledger
  exactly. Requiring a transaction id next to every amount makes this checkable by regex and
  lookup rather than by asking a model to judge its own arithmetic.
- Retrieval memory was chosen as the consumption mechanism because its effect is directly
  observable in the next generation call. Alternatives are prompt-patching from failure patterns
  and fine-tuning on the preference dataset.
- Abstain versus escalate is a fixed rule (fabricated or unsupported claims abstain; a real record
  with a wrong number escalates).
- The human simulator is a fixed policy, not a model of real human judgment.

## Limitations and future work

- The verifier only catches claims that resolve to a dollar amount; a qualitative fabricated claim
  with no number would not be caught.
- The verifier checks grounding, not completeness or tone; the judge covers those, within its four
  fixed dimensions.
- Model-scored judge dimensions (completeness, tone) vary run to run because of live sampling, so
  treat them as directional. The code-derived dimensions are stable.
- Retrieval is recency-based within a client, not similarity-scored.
- Sentence-level checks in the verifier are regex-based and tuned to observed model output; other
  edge cases may exist.
- Not yet built: a DPO/LoRA fine-tune on the preference dataset, additional artifact types (such as
  an account summary reusing the same ledger and verifier shape), further judge failure-mode
  defenses (sycophancy, self-preference, position bias), and a go/no-go gate for model changes.
