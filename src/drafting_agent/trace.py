import json
import time
from pathlib import Path
from typing import Optional

DEFAULT_LOG_PATH = Path("data/logs/trace.jsonl")


def trace(
    task: str,
    prompt: str,
    output: str,
    meta: Optional[dict] = None,
    path: Path = DEFAULT_LOG_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "task": task,
        "prompt": prompt,
        "output": output,
        "meta": meta or {},
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
